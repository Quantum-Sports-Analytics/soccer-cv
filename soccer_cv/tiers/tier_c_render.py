"""Tier C — fuse Tier A tracks with Tier B identities, then render the overlay video.

fused.parquet (FUSED_COLUMNS): per frame, per visible identity.
overlay.mp4: the acceptance artefact. It deliberately shows what a metric
table would show:
  * marker colour = team cluster; marker alpha/ring = identity confidence
  * `?` label and grey box for abstained identities (reported unknown)
  * a flash + "↩ #id" tag on the first frame an identity is re-linked after a gap
  * ball marker coloured by state (visible / occluded-interpolated / in flight)
  * right-hand roster panel: identity, team, visible / off-frame / uncertain,
    frames since last seen
  * footer: frame, live-track count, abstention count, low-margin count
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..core import Stage, StageContext, Storage, frames_iter
from ..schema import FUSED_COLUMNS, BallState, RosterState

log = logging.getLogger(__name__)

TEAM_COLORS = {0: (60, 76, 231), 1: (255, 200, 60), 2: (90, 220, 90), -1: (160, 160, 160)}   # BGR
BALL_COLORS = {BallState.VISIBLE.value: (255, 255, 255), BallState.IN_FLIGHT.value: (0, 255, 255),
               BallState.OCCLUDED.value: (200, 120, 255), BallState.OUT_OF_FRAME.value: (120, 120, 120),
               BallState.UNKNOWN.value: (120, 120, 120), BallState.HELD.value: (255, 180, 80)}


def _margin_to_err_p(margin: np.ndarray) -> np.ndarray:
    """Uncalibrated placeholder: low assignment margin -> higher link error probability.
    To be replaced by a calibration curve fitted on annotated re-entries."""
    return np.clip(0.5 * np.exp(-6.0 * margin), 0.01, 0.5)


class FuseStage(Stage):
    """Input: run root with tier_a/<shot>/{s4_track,s7_ball} and tier_b/."""
    name = "tier_c_fuse"
    config_key = "render"

    def run(self, ctx: StageContext) -> dict:
        root = ctx.input_uri
        ident = Storage.read_df(Storage.join(root, "tier_b", "identities.parquet"))
        key = ident.set_index(["shot_id", "track_id"])
        fused, balls = [], []
        for u in Storage.list(Storage.join(root, "tier_a")):
            if u.endswith("tracks.parquet"):
                shot_id = Path(u).parts[-3]
                tr = Storage.read_df(u)
                tr["shot_id"] = shot_id
                fused.append(tr)
            elif u.endswith("ball.parquet"):
                balls.append(Storage.read_df(u))
        tr = pd.concat(fused, ignore_index=True) if fused else pd.DataFrame()
        ball = pd.concat(balls, ignore_index=True) if balls else pd.DataFrame()

        out_rows = []
        first_seen: dict[int, int] = {}
        # re-entry: first frame of any tracklet that is NOT the identity's first tracklet
        ident_first_track = ident.sort_values("track_id").groupby("identity_id").head(1).set_index(["shot_id", "track_id"]).index
        for (shot_id, tid), g in tr.groupby(["shot_id", "track_id"]):
            if (shot_id, tid) not in key.index:
                continue
            r = key.loc[(shot_id, tid)]
            g = g.sort_values("frame")
            err = _margin_to_err_p(g.margin.to_numpy(dtype=float))
            is_reentry = (shot_id, tid) not in ident_first_track and not bool(r.abstained)
            for k, (_, row) in enumerate(g.iterrows()):
                out_rows.append((int(row.frame), int(r.identity_id) if not r.abstained else -1,
                                 int(r.team), r.jersey, row.x1, row.y1, row.x2, row.y2, None, None,
                                 float(r.confidence) if not r.abstained else 0.0, float(err[k]),
                                 1 if (is_reentry and k == 0) else 0))
        fused_df = pd.DataFrame(out_rows, columns=FUSED_COLUMNS)
        Storage.write_df(ctx.out("fused.parquet"), fused_df)
        Storage.write_df(ctx.out("ball.parquet"), ball)
        return {"n_rows": len(fused_df), "n_identities": int((fused_df.identity_id > 0).sum() and fused_df[fused_df.identity_id > 0].identity_id.nunique()),
                "abstained_rows": int((fused_df.identity_id == -1).sum()),
                "reentries": int(fused_df.reentry.sum())}


class RenderStage(Stage):
    """Input: run root. Writes overlay.mp4 (H.264 via ffmpeg) next to fused.parquet."""
    name = "tier_c_render"
    config_key = "render"

    def run(self, ctx: StageContext) -> dict:
        root = ctx.input_uri
        p = self.params
        video = Storage.localize(Storage.join(root, "ingest", "video.mp4"), ctx.workdir)
        fused = Storage.read_df(Storage.join(root, "tier_c", "fused.parquet"))
        ball = Storage.read_df(Storage.join(root, "tier_c", "ball.parquet"))
        roster = Storage.read_df(Storage.join(root, "tier_b", "roster.parquet"))
        meta = Storage.read_json(Storage.join(root, "ingest", "meta.json"))
        fps, W, H = meta["fps"], meta["width"], meta["height"]
        PANEL = 300 if p.get("show_roster_panel", True) else 0

        by_frame = {int(k): g for k, g in fused.groupby("frame")}
        ball_by = ball.set_index("frame") if len(ball) else None
        roster_frames = np.sort(roster.frame.unique()) if len(roster) else np.array([])
        raw_path = ctx.workdir / "overlay_raw.mp4"
        vw = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + PANEL, H))
        flash: dict[int, int] = {}      # identity -> frames of re-entry flash remaining
        n_low = 0
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, frame in frames_iter(video):
            canvas = np.zeros((H, W + PANEL, 3), dtype=np.uint8); canvas[:, :W] = frame
            g = by_frame.get(i)
            n_abst = 0
            if g is not None:
                for _, r in g.iterrows():
                    x1, y1, x2, y2 = [int(v) for v in (r.x1, r.y1, r.x2, r.y2)]
                    abst = r.identity_id == -1
                    col = TEAM_COLORS[-1] if abst else TEAM_COLORS.get(int(r.team), TEAM_COLORS[-1])
                    thick = 1 if abst else (2 if r.conf > 0.75 else 1)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), col, thick)
                    label = "?" if abst else f"{int(r.identity_id)}"
                    if r.reentry:
                        flash[int(r.identity_id)] = int(fps)
                    if not abst and flash.get(int(r.identity_id), 0) > 0:
                        cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 2)
                        label = f"<- {label}"
                        flash[int(r.identity_id)] -= 1
                    if p.get("show_confidence", True) and not abst:
                        label += f" {r.conf:.2f}"
                    if r.link_err_p > 0.25:
                        n_low += 1
                        cv2.circle(canvas, (x2, y1), 4, (0, 165, 255), -1)      # low-margin dot
                    (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
                    cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
                    cv2.putText(canvas, label, (x1 + 2, y1 - 4), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
                    n_abst += int(abst)
            if ball_by is not None and i in ball_by.index:
                b = ball_by.loc[i]
                if not np.isnan(b.x):
                    bc = BALL_COLORS.get(str(b.state), (255, 255, 255))
                    cv2.circle(canvas, (int(b.x), int(b.y)), 10, bc, 2)
                    cv2.putText(canvas, str(b.state), (int(b.x) + 12, int(b.y) - 8), font, 0.4, bc, 1, cv2.LINE_AA)
                else:
                    cv2.putText(canvas, f"ball: {b.state}", (12, H - 40), font, 0.5, BALL_COLORS.get(str(b.state), (150, 150, 150)), 1, cv2.LINE_AA)

            # roster panel
            if PANEL:
                cv2.rectangle(canvas, (W, 0), (W + PANEL, H), (28, 28, 28), -1)
                cv2.putText(canvas, "ROSTER  (id / state / since)", (W + 10, 26), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
                if len(roster_frames):
                    rf = roster_frames[np.searchsorted(roster_frames, i, side="right") - 1] if i >= roster_frames[0] else None
                    if rf is not None:
                        rs = roster[roster.frame == rf].sort_values(["team", "identity_id"])
                        y = 50
                        for _, r in rs.iterrows():
                            col = TEAM_COLORS.get(int(r.team), TEAM_COLORS[-1])
                            cv2.circle(canvas, (W + 18, y - 5), 5, col, -1)
                            st = str(r.state)
                            txt_col = (230, 230, 230) if st == RosterState.VISIBLE.value else (150, 150, 150)
                            since = "" if r.frames_since_seen in (None, 0) or pd.isna(r.frames_since_seen) else f"{int(r.frames_since_seen) / fps:.1f}s"
                            cv2.putText(canvas, f"#{int(r.identity_id):<3} {st:<10} {since}", (W + 32, y), font, 0.42, txt_col, 1, cv2.LINE_AA)
                            y += 18
                            if y > H - 70:
                                break
            footer = f"frame {i}  tracks {0 if g is None else len(g)}  unknown {n_abst}  low-margin dots = ambiguous link"
            cv2.putText(canvas, footer, (12, H - 14), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            vw.write(canvas)
        vw.release()

        final = ctx.workdir / "overlay.mp4"
        try:   # re-encode to H.264 so browsers play it
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(final)], check=True)
        except Exception as e:                       # noqa: BLE001
            log.warning("ffmpeg re-encode failed (%s); keeping mp4v", e)
            final = raw_path
        Storage.upload_file(final, ctx.out("overlay.mp4"))
        return {"frames": int(meta.get("n_frames", 0)), "low_margin_marks": n_low, "panel": bool(PANEL)}
