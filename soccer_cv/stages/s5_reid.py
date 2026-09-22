"""Stage 5a — ReID embeddings + resolution of ambiguous crossings (same-team swaps).

Why this exists: during a mutual occlusion between two teammates the tracker
has to assign boxes from one blurred, half-visible crop. That decision is
unreliable and there is no cue at that instant that fixes it — colour is
identical, motion is discontinuous (tackles, stops). But there is abundant
evidence *around* the window: dozens of clean crops of each player before and
after. So we (1) detect ambiguous windows from the tracker's own signals
(mutual box overlap + low assignment margin), (2) compute OSNet embeddings at
`every_n_frames` for every track, and (3) for each window compare the two
hypotheses — keep vs swap — on appearance similarity of pre/post segments plus
image-space motion continuity, applying the swap when it wins by a margin and
recording the decision confidence either way.

Input : s4_track output (tracks.parquet) + ingest video.
Output: tracks.parquet   corrected track ids (same schema + `swap_conf` column)
        track_emb.parquet (frame, track_id, emb[512])
        windows.json      ambiguous windows with the decision and its margin
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..core import Stage, StageContext, Storage, frames_iter
from ..reid import ReIDEncoder

log = logging.getLogger(__name__)


def _pair_iou(a, b) -> float:
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1]); xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(xx2 - xx1, 0) * max(yy2 - yy1, 0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def find_ambiguous_windows(tr: pd.DataFrame, margin_thr: float, iou_thr: float = 0.05,
                           merge_gap: int = 15, same_team_only: bool = True) -> list[dict]:
    """Pairs of tracks that overlap while at least one has a low assignment margin."""
    events: dict[tuple[int, int], list[int]] = {}
    for f, g in tr.groupby("frame"):
        if len(g) < 2:
            continue
        rows = g[["track_id", "x1", "y1", "x2", "y2", "margin", "team_online"]].to_numpy()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if same_team_only and rows[i][6] >= 0 and rows[j][6] >= 0 and rows[i][6] != rows[j][6]:
                    continue
                if min(rows[i][5], rows[j][5]) > margin_thr:
                    continue
                if _pair_iou(rows[i][1:5], rows[j][1:5]) < iou_thr:
                    continue
                key = (int(min(rows[i][0], rows[j][0])), int(max(rows[i][0], rows[j][0])))
                events.setdefault(key, []).append(int(f))
    windows = []
    for (a, b), frames in events.items():
        frames.sort(); start = frames[0]; prev = frames[0]
        for f in frames[1:] + [None]:
            if f is None or f - prev > merge_gap:
                windows.append({"a": a, "b": b, "f0": start, "f1": prev})
                if f is not None:
                    start = f
            if f is not None:
                prev = f
    # a track can also lose its detection entirely inside the window (gap): widen by the
    # tracker's gaps adjacent to the window
    return sorted(windows, key=lambda w: w["f0"])


class ReIDStage(Stage):
    name = "s5_reid"
    config_key = "reid"

    def __init__(self, cfg: dict, shot_id: str, ingest_uri: str, encoder: ReIDEncoder | None = None):
        super().__init__(cfg)
        self.shot_id, self.ingest_uri = shot_id, ingest_uri
        self.encoder = encoder

    def run(self, ctx: StageContext) -> dict:
        from .s0_ingest import load_shots
        p = self.params
        shot = next(s for s in load_shots(self.ingest_uri) if s.shot_id == self.shot_id)
        tr = Storage.read_df(ctx.inp("tracks.parquet")).sort_values(["frame", "track_id"]).reset_index(drop=True)
        every = int(p.get("every_n_frames", 5))
        margin_thr = float(self.cfg.get("track", {}).get("margin_low", 0.15))
        W = int(p.get("swap_window_frames", 100))
        tau = float(p.get("swap_margin", 0.02))
        w_motion = float(p.get("swap_motion_weight", 0.3))

        # ---- 1. embeddings at every_n_frames
        enc = self.encoder or ReIDEncoder(device=self.cfg.get("runtime", {}).get("device", "auto"),
                                          fp16=bool(self.cfg.get("runtime", {}).get("fp16", False)))
        video = Storage.localize(Storage.join(self.ingest_uri, "video.mp4"), ctx.workdir)
        by_frame = {int(k): g for k, g in tr.groupby("frame")}
        emb_rows = []
        for i, frame in frames_iter(video, shot.start_frame, shot.end_frame):
            if (i - shot.start_frame) % every:
                continue
            g = by_frame.get(i)
            if g is None or not len(g):
                continue
            g = g[g.occl < 0.5]                              # only reasonably visible crops
            if not len(g):
                continue
            E = enc(frame, g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float))
            for tid, e in zip(g.track_id.to_numpy(), E):
                emb_rows.append((i, int(tid), e.astype(np.float32)))
            if (i - shot.start_frame) % (every * 40) == 0:
                log.info("reid %s frame %d", self.shot_id, i)
        emb = pd.DataFrame(emb_rows, columns=["frame", "track_id", "emb"])

        # ---- 2. ambiguous windows
        windows = find_ambiguous_windows(tr, margin_thr)

        # ---- 3. keep-vs-swap decision per window, applied chronologically
        tr["swap_conf"] = 1.0
        tid = tr.track_id.to_numpy().copy()
        frames = tr.frame.to_numpy()
        emb_tid = emb.track_id.to_numpy().copy(); emb_frames = emb.frame.to_numpy()
        E_all = np.stack(emb.emb.to_numpy()) if len(emb) else np.zeros((0, 512), np.float32)

        def seg_mean(t, f_lo, f_hi):
            m = (emb_tid == t) & (emb_frames >= f_lo) & (emb_frames <= f_hi)
            if m.sum() == 0:
                return None
            v = E_all[m].mean(0); return v / (np.linalg.norm(v) + 1e-9)

        def foot_vel(t, f_lo, f_hi):
            m = (tid == t) & (frames >= f_lo) & (frames <= f_hi)
            if m.sum() < 3:
                return None, None
            sub = tr[m].sort_values("frame")
            cx = ((sub.x1 + sub.x2) / 2).to_numpy(); fy = sub.y2.to_numpy(); f = sub.frame.to_numpy()
            v = np.array([np.polyfit(f, cx, 1)[0], np.polyfit(f, fy, 1)[0]])
            return np.array([cx[-1], fy[-1]]), v

        decisions = []
        n_swapped = 0
        for w in windows:
            a, b, f0, f1 = w["a"], w["b"], w["f0"], w["f1"]
            # labels may already have been swapped by an earlier window: work on current ids
            Apre, Bpre = seg_mean(a, f0 - W, f0 - 1), seg_mean(b, f0 - W, f0 - 1)
            Apost, Bpost = seg_mean(a, f1 + 1, f1 + W), seg_mean(b, f1 + 1, f1 + W)
            if any(x is None for x in (Apre, Bpre, Apost, Bpost)):
                decisions.append({**w, "decision": "undecidable", "margin": 0.0}); continue
            s_keep = float(Apre @ Apost + Bpre @ Bpost)
            s_swap = float(Apre @ Bpost + Bpre @ Apost)
            # motion continuity: extrapolate pre-window foot position/velocity to the first post frame
            m_keep = m_swap = 0.0
            pa, va = foot_vel(a, f0 - 25, f0 - 1); pb, vb = foot_vel(b, f0 - 25, f0 - 1)
            qa, _ = foot_vel(a, f1 + 1, f1 + 25); qb, _ = foot_vel(b, f1 + 1, f1 + 25)
            if all(x is not None for x in (pa, va, pb, vb, qa, qb)):
                gap = f1 + 1 - (f0 - 1)
                ea, eb = pa + va * gap, pb + vb * gap
                h = float(tr[(tid == a) & (frames == f0 - 1)].y2.iloc[0] - tr[(tid == a) & (frames == f0 - 1)].y1.iloc[0]) if ((tid == a) & (frames == f0 - 1)).any() else 80.0
                d = lambda u, v: float(np.exp(-np.linalg.norm(u - v) / max(h, 20.0)))
                m_keep, m_swap = d(ea, qa) + d(eb, qb), d(ea, qb) + d(eb, qa)
            score = (1 - w_motion) * (s_swap - s_keep) + w_motion * (m_swap - m_keep) * 0.5
            conf = float(min(1.0, abs(score) / (3 * tau)))
            if score > tau:
                # swap ids of a and b for all frames after the window
                post = frames > f1
                ma, mb = post & (tid == a), post & (tid == b)
                tid[ma], tid[mb] = b, a
                epost = emb_frames > f1
                ea_, eb_ = epost & (emb_tid == a), epost & (emb_tid == b)
                emb_tid[ea_], emb_tid[eb_] = b, a
                n_swapped += 1
                decisions.append({**w, "decision": "swap", "margin": round(score, 4), "app": round(s_swap - s_keep, 4), "motion": round(m_swap - m_keep, 4)})
            else:
                decisions.append({**w, "decision": "keep", "margin": round(score, 4), "app": round(s_swap - s_keep, 4), "motion": round(m_swap - m_keep, 4)})
            # confidence of the identity of both tracks right after the window
            for t in (a, b):
                m = (tid == t) & (frames > f1) & (frames <= f1 + W)
                tr.loc[m, "swap_conf"] = np.minimum(tr.loc[m, "swap_conf"].to_numpy(), conf)

        tr["track_id"] = tid
        emb["track_id"] = emb_tid
        Storage.write_df(ctx.out("tracks.parquet"), tr)
        emb_out = pd.DataFrame({"frame": emb.frame, "track_id": emb.track_id,
                                "emb": [json.dumps(np.round(e, 4).tolist()) for e in emb.emb]})
        Storage.write_df(ctx.out("track_emb.parquet"), emb_out)
        Storage.write_json(ctx.out("windows.json"), decisions)
        # track-level mean ReID embedding for Tier B linking
        means = []
        for t, g in emb.groupby("track_id"):
            v = np.stack(g.emb.to_numpy()).mean(0); v /= (np.linalg.norm(v) + 1e-9)
            means.append({"track_id": int(t), "reid": json.dumps(np.round(v, 4).tolist()), "n": len(g)})
        Storage.write_df(ctx.out("track_reid.parquet"), pd.DataFrame(means))
        undec = sum(d["decision"] == "undecidable" for d in decisions)
        return {"n_windows": len(windows), "n_swapped": n_swapped, "n_undecidable": undec,
                "n_emb": len(emb), "low_conf_frames": int((tr.swap_conf < 0.5).sum())}
