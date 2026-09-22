"""Stage 7 — ball track with an explicit state machine, from per-frame ball detections.

This is the *association* half of the ball problem. The detection half is the
generic detector's `sports ball` class for now; the dedicated signed-motion
detector (TGMA-Net / TrackNetV5 family) plugs in at stage 3 without changing
this file.

Rules:
  * A detection is accepted if it lies within `max_speed_px_per_frame` * gap of
    the last accepted position (or if there is no track yet).
  * Among several candidates the one closest to the constant-velocity
    prediction wins.
  * Gaps shorter than `max_gap_frames` are linearly interpolated and labelled
    OCCLUDED; longer gaps are UNKNOWN. IN_FLIGHT is set when the vertical
    acceleration over a 5-frame window looks parabolic (upward then downward).
  * Positions near the frame edge that vanish are labelled OUT_OF_FRAME.

Output: <output_uri>/ball.parquet (BALL_COLUMNS)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import Stage, StageContext, Storage
from ..schema import BALL_COLUMNS, BallState, ObjClass


class BallStage(Stage):
    name = "s7_ball"
    config_key = "ball"

    def __init__(self, cfg: dict, shot_id: str, ingest_uri: str):
        super().__init__(cfg)
        self.shot_id = shot_id
        self.ingest_uri = ingest_uri

    def run(self, ctx: StageContext) -> dict:
        from .s0_ingest import load_shots
        p = self.params
        shot = next(s for s in load_shots(self.ingest_uri) if s.shot_id == self.shot_id)
        W, H = shot.width, shot.height
        dets = Storage.read_df(ctx.inp("detections.parquet"))
        ball = dets[(dets.cls == ObjClass.BALL.value) & (dets.score >= float(p.get("score_threshold", 0.2)))]
        by_frame = {int(k): g for k, g in ball.groupby("frame")}
        vmax = float(p.get("max_speed_px_per_frame", 90))
        max_gap = int(p.get("max_gap_frames", 20))

        frames = list(range(shot.start_frame, shot.end_frame + 1))
        pos = np.full((len(frames), 2), np.nan); score = np.zeros(len(frames))
        last = None; last_i = None; vel = np.zeros(2)
        for k, f in enumerate(frames):
            g = by_frame.get(f)
            if g is None:
                continue
            c = np.column_stack([(g.x1 + g.x2) / 2, (g.y1 + g.y2) / 2]).astype(float)
            s = g.score.to_numpy()
            if last is None:
                j = int(s.argmax())
            else:
                gap = k - last_i
                pred = last + vel * gap
                dist = np.linalg.norm(c - pred, axis=1)
                ok = dist <= vmax * gap
                if not ok.any():
                    continue
                j = int(np.argmin(np.where(ok, dist - 40 * s, np.inf)))
            if last is not None:
                vel = (c[j] - last) / max(k - last_i, 1)
            pos[k] = c[j]; score[k] = s[j]; last, last_i = c[j], k

        # states + interpolation
        state = np.array([BallState.UNKNOWN.value] * len(frames), dtype=object)
        seen = ~np.isnan(pos[:, 0])
        state[seen] = BallState.VISIBLE.value
        idx = np.where(seen)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a
            if 1 < gap <= max_gap:
                t = np.linspace(0, 1, gap + 1)[1:-1]
                pos[a + 1:b] = pos[a][None] * (1 - t[:, None]) + pos[b][None] * t[:, None]
                state[a + 1:b] = BallState.OCCLUDED.value
            elif gap > max_gap:
                x, y = pos[a]
                edge = min(x, W - x, y, H - y) < 0.06 * min(W, H)
                state[a + 1:b] = BallState.OUT_OF_FRAME.value if edge else BallState.UNKNOWN.value
        # in-flight heuristic: vertical parabola over a 9-frame window
        y = pos[:, 1]
        for k in range(4, len(frames) - 4):
            w = y[k - 4:k + 5]
            if np.isnan(w).any():
                continue
            d2 = np.diff(w, 2)
            if state[k] == BallState.VISIBLE.value and d2.mean() > 0.35 and np.all(d2 > -0.2):
                state[k] = BallState.IN_FLIGHT.value

        df = pd.DataFrame({"frame": frames, "x": pos[:, 0], "y": pos[:, 1], "score": score, "state": state},
                          columns=BALL_COLUMNS)
        Storage.write_df(ctx.out("ball.parquet"), df)
        counts = df.state.value_counts().to_dict()
        return {"visible_frac": round(float(seen.mean()), 3),
                "located_frac": round(float((~np.isnan(pos[:, 0])).mean()), 3),
                "states": {k: int(v) for k, v in counts.items()}}
