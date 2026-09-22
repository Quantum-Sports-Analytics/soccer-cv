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


def current_ids(w: dict, raw: np.ndarray, tid: np.ndarray, frames: np.ndarray) -> dict:
    """Windows are found on raw tracker ids; earlier relabels may have renamed them."""
    out = dict(w)
    for key in ("a", "b"):
        m = (raw == w[key]) & (frames >= w["f0"]) & (frames <= w["f1"] + 30)
        if m.any():
            out[key] = int(tid[np.where(m)[0][0]])
    return out


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def resolve_window(w: dict, tr: pd.DataFrame, tid: np.ndarray, frames: np.ndarray,
                   emb_tid: np.ndarray, emb_frames: np.ndarray, E_all: np.ndarray,
                   W: int, tau: float, w_motion: float) -> dict:
    """Assign pre-window segments to post-window segments for one ambiguous window.

    Participants = the two overlapping tracks plus any track that is *born* during
    the window (or just after it) close to the crossing — that is how a lost player
    re-appears under a new id. Cost = appearance (ReID cosine on pre/post segment
    means); motion continuity only breaks near-ties, because in tackles and stops
    it is not informative. Post-segments matched to a different pre-id are relabelled
    from the window end onwards; a new-born track matched to a pre-id is relabelled
    from its birth.
    """
    from scipy.optimize import linear_sum_assignment
    a, b, f0, f1 = w["a"], w["b"], w["f0"], w["f1"]
    x1 = tr.x1.to_numpy(); x2 = tr.x2.to_numpy(); y1 = tr.y1.to_numpy(); y2 = tr.y2.to_numpy()

    def seg(t, lo, hi):
        m = (emb_tid == t) & (emb_frames >= lo) & (emb_frames <= hi)
        return _unit(E_all[m].mean(0)) if m.sum() else None

    def foot_track(t, lo, hi):
        m = (tid == t) & (frames >= lo) & (frames <= hi)
        if m.sum() < 3:
            return None
        f = frames[m]; cx = (x1[m] + x2[m]) / 2; fy = y2[m]; h = float(np.median(y2[m] - y1[m]))
        o = np.argsort(f); f, cx, fy = f[o], cx[o], fy[o]
        vx, vy = np.polyfit(f, cx, 1)[0], np.polyfit(f, fy, 1)[0]
        return dict(f=f, p0=np.array([cx[0], fy[0]]), p1=np.array([cx[-1], fy[-1]]), v=np.array([vx, vy]), h=max(h, 20.0))

    # crossing region: union of a/b boxes inside the window
    m_win = ((tid == a) | (tid == b)) & (frames >= f0) & (frames <= f1)
    if not m_win.any():
        return {"record": {**w, "decision": "undecidable", "margin": 0.0}, "relabels": [], "participants": [a, b], "conf": 0.3}
    rx1, ry1, rx2, ry2 = x1[m_win].min(), y1[m_win].min(), x2[m_win].max(), y2[m_win].max()
    rh = float(np.median(y2[m_win] - y1[m_win]))
    # tracks born in [f0, f1 + 30] whose first box lies near the region
    births = []
    for t in np.unique(tid[(frames >= f0) & (frames <= f1 + 30)]):
        if t in (a, b):
            continue
        mt = tid == t
        fb = frames[mt].min()
        if fb < f0:
            continue
        i = np.where(mt & (frames == fb))[0][0]
        cx, fy = (x1[i] + x2[i]) / 2, y2[i]
        if rx1 - rh <= cx <= rx2 + rh and ry1 - rh <= fy <= ry2 + rh:
            births.append(int(t))
    pre_ids = [t for t in (a, b) if seg(t, f0 - W, f0 - 1) is not None]
    post_ids = [t for t in [a, b] + births if seg(t, f1 + 1, f1 + W) is not None]
    if len(pre_ids) < 2 or len(post_ids) < 2:
        return {"record": {**w, "decision": "undecidable", "margin": 0.0, "births": births}, "relabels": [], "participants": [a, b] + births, "conf": 0.3}
    pre = {t: seg(t, f0 - W, f0 - 1) for t in pre_ids}; post = {t: seg(t, f1 + 1, f1 + W) for t in post_ids}
    pre_m = {t: foot_track(t, f0 - 25, f0 - 1) for t in pre_ids}; post_m = {t: foot_track(t, f1 + 1, f1 + 25) for t in post_ids}
    A = np.array([[float(pre[i] @ post[j]) for j in post_ids] for i in pre_ids])          # appearance
    M = np.zeros_like(A)
    for r, i in enumerate(pre_ids):
        for c, j in enumerate(post_ids):
            pm, qm = pre_m[i], post_m[j]
            if pm is None or qm is None:
                continue
            gap = qm["f"][0] - pm["f"][-1]
            pred = pm["p1"] + pm["v"] * gap
            M[r, c] = float(np.exp(-np.linalg.norm(pred - qm["p0"]) / (pm["h"] * max(1.0, gap / 12.0))))
    score = A + w_motion * M
    r_idx, c_idx = linear_sum_assignment(-score)
    best = float(score[r_idx, c_idx].sum())
    # second best: force each chosen cell out in turn, take the best alternative assignment
    second = -np.inf
    for r, c in zip(r_idx, c_idx):
        s2 = score.copy(); s2[r, c] = -1e3
        rr, cc = linear_sum_assignment(-s2); second = max(second, float(s2[rr, cc].sum()))
    margin = best - second if np.isfinite(second) else 1.0
    # appearance-only view of the same decision, for the record
    ra, ca = linear_sum_assignment(-A); app_best = float(A[ra, ca].sum())
    identity = all(pre_ids[r] == post_ids[c] for r, c in zip(r_idx, c_idx))
    relabels = []
    if margin <= tau:
        # Undecidable: do not assert continuity. Cut the overlapping tracks at the window end
        # so their post-segments become fresh tracklets (Tier B may re-link them later with
        # more evidence), and flag the window. A wrong identity costs more than a new one.
        tmp_base = int(tid.max()) + 2000
        for k, t in enumerate((a, b)):
            if ((tid == t) & (frames > f1)).any():
                relabels.append((t, tmp_base + k, f1 + 1))
        rec = {**w, "decision": "split", "margin": round(float(margin), 4), "app_best": round(app_best, 4),
               "pre": pre_ids, "post": post_ids, "births": births,
               "assignment": [(int(pre_ids[r]), int(post_ids[c])) for r, c in zip(r_idx, c_idx)]}
        return {"record": rec, "relabels": relabels, "participants": [a, b] + births, "conf": 0.3}
    if not identity:
        # apply: post-segment of post_ids[c] becomes pre_ids[r]
        pairs = [(pre_ids[r], post_ids[c]) for r, c in zip(r_idx, c_idx)]
        # two-phase relabel through temporary ids to avoid collisions
        tmp_base = int(tid.max()) + 1000
        for k, (pi, pj) in enumerate(pairs):
            if pi != pj:
                f_from = f1 + 1 if pj in (a, b) else int(frames[tid == pj].min())
                relabels.append((pj, tmp_base + k, f_from))
        for k, (pi, pj) in enumerate(pairs):
            if pi != pj:
                relabels.append((tmp_base + k, pi, 0))
        # a pre-id whose own post-segment was given away and that received nothing keeps its
        # orphaned continuation under a fresh id (it is a wrong continuation, not the same player)
        given = {pj for pi, pj in pairs if pi != pj}
        received = {pi for pi, pj in pairs if pi != pj}
        for t in given - received:
            if t in (a, b):
                relabels.append((t, tmp_base + 500 + t, f1 + 1))
        decision = "swap"
    else:
        decision = "keep"
    conf = float(min(1.0, abs(margin) / (3 * tau)))
    rec = {**w, "decision": decision, "margin": round(float(margin), 4), "app_best": round(app_best, 4),
           "pre": pre_ids, "post": post_ids, "births": births,
           "assignment": [(int(pre_ids[r]), int(post_ids[c])) for r, c in zip(r_idx, c_idx)]}
    return {"record": rec, "relabels": relabels, "participants": [a, b] + births, "conf": conf}


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

        # ---- 1. embeddings at every_n_frames (cached on raw tracker ids: the expensive part)
        import hashlib
        key = hashlib.sha1(pd.util.hash_pandas_object(tr[["frame", "track_id", "x1", "y1", "x2", "y2"]], index=False).values.tobytes()).hexdigest()[:12] + f"-e{every}"
        cache_uri = ctx.out("track_emb_raw.parquet"); cache_key_uri = ctx.out("track_emb_raw.key")
        emb = None
        if Storage.exists(cache_uri) and Storage.exists(cache_key_uri) and Storage.read_bytes(cache_key_uri).decode() == key:
            c = Storage.read_df(cache_uri)
            emb = pd.DataFrame({"frame": c.frame, "track_id": c.track_id, "emb": [np.asarray(json.loads(e), np.float32) for e in c.emb]})
            log.info("reid %s: %d cached embeddings", self.shot_id, len(emb))
        enc = None if emb is not None else (self.encoder or ReIDEncoder(device=self.cfg.get("runtime", {}).get("device", "auto"),
                                          fp16=bool(self.cfg.get("runtime", {}).get("fp16", False))))
        video = Storage.localize(Storage.join(self.ingest_uri, "video.mp4"), ctx.workdir)
        by_frame = {int(k): g for k, g in tr.groupby("frame")}
        emb_rows = []
        for i, frame in (frames_iter(video, shot.start_frame, shot.end_frame) if emb is None else []):
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
        if emb is None:
            emb = pd.DataFrame(emb_rows, columns=["frame", "track_id", "emb"])
            Storage.write_df(cache_uri, pd.DataFrame({"frame": emb.frame, "track_id": emb.track_id,
                                                      "emb": [json.dumps(np.round(e, 4).tolist()) for e in emb.emb]}))
            Storage.write_bytes(cache_key_uri, key.encode())

        # ---- 2. ambiguous windows
        windows = find_ambiguous_windows(tr, margin_thr)

        # ---- 3. resolve each window as an assignment: pre-segments -> post-segments
        tr["swap_conf"] = 1.0
        tid = tr.track_id.to_numpy().copy(); frames = tr.frame.to_numpy()
        emb_tid = emb.track_id.to_numpy().copy(); emb_frames = emb.frame.to_numpy()
        E_all = np.stack(emb.emb.to_numpy()) if len(emb) else np.zeros((0, 512), np.float32)
        decisions, n_swapped, n_relabel = [], 0, 0
        raw = tr.track_id.to_numpy()
        for w in windows:
            w = current_ids(w, raw, tid, frames)
            res = resolve_window(w, tr, tid, frames, emb_tid, emb_frames, E_all, W, tau, w_motion)
            decisions.append(res["record"])
            for old, new_, f_from in res["relabels"]:
                m = (tid == old) & (frames >= f_from); me = (emb_tid == old) & (emb_frames >= f_from)
                tid[m] = new_; emb_tid[me] = new_; n_relabel += 1
            n_swapped += int(res["record"]["decision"] == "swap")
            for t in res["participants"]:
                m = (tid == t) & (frames > w["f1"]) & (frames <= w["f1"] + W)
                tr.loc[m, "swap_conf"] = np.minimum(tr.loc[m, "swap_conf"].to_numpy(), res["conf"])

        tr["track_id"] = tid
        emb["track_id"] = emb_tid
        # colour signatures (team cue) for ids created by relabelling: inherit from the raw parent
        if Storage.exists(ctx.inp("track_app.parquet")):
            app = Storage.read_df(ctx.inp("track_app.parquet"))
            have = set(app.track_id.tolist()); rows_app = []
            for new_id in np.unique(tid):
                if new_id in have:
                    continue
                parents = pd.Series(raw[tid == new_id]).value_counts()
                parent = int(parents.index[0])
                if parent in have:
                    r = app[app.track_id == parent].iloc[0].to_dict(); r["track_id"] = int(new_id)
                    r["n_frames"] = int((tid == new_id).sum()); rows_app.append(r)
            if rows_app:
                app = pd.concat([app, pd.DataFrame(rows_app)], ignore_index=True)
            Storage.write_df(ctx.out("track_app.parquet"), app)
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
        nsplit = sum(d["decision"] == "split" for d in decisions)
        return {"n_windows": len(windows), "n_swapped": n_swapped, "n_split": nsplit, "n_relabels": n_relabel, "n_undecidable": undec,
                "n_emb": len(emb), "low_conf_frames": int((tr.swap_conf < 0.5).sum())}
