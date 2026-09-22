"""Stage 5 — tracklet summaries: the Tier A -> Tier B hand-off. No pixels leave here.

For every tracklet produced by stage 4 we compute one TrackletSummary:
appearance (visibility-weighted colour embedding), geometry (first/last box,
pitch coordinates when calibration is valid), quality (mean margin, occluded
fraction) and, when the OCR stage has run, jersey votes.

Output: <output_uri>/tracklets.parquet — one row per tracklet, JSON-encoded
        embedding columns so it stays a flat table.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..core import Stage, StageContext, Storage


class SummarizeStage(Stage):
    name = "s5_summarize"
    config_key = "reid"

    def __init__(self, cfg: dict, shot_id: str, calib_uri: str | None = None, app_uri: str | None = None):
        super().__init__(cfg)
        self.shot_id = shot_id
        self.calib_uri = calib_uri
        self.app_uri = app_uri          # where track_app.parquet lives (s4_track) when input is s5_reid

    def run(self, ctx: StageContext) -> dict:
        tracks = Storage.read_df(ctx.inp("tracks.parquet"))
        app_src = self.app_uri or ctx.input_uri
        app = Storage.read_df(Storage.join(app_src, "track_app.parquet")).set_index("track_id")
        reid = None
        if Storage.exists(ctx.inp("track_reid.parquet")):
            reid = Storage.read_df(ctx.inp("track_reid.parquet")).set_index("track_id")
        if "swap_conf" not in tracks:
            tracks["swap_conf"] = 1.0
        H = None
        if self.calib_uri and Storage.exists(Storage.join(self.calib_uri, "calib.parquet")):
            H = Storage.read_df(Storage.join(self.calib_uri, "calib.parquet")).set_index("frame")

        rows = []
        for tid, g in tracks.groupby("track_id"):
            g = g.sort_values("frame")
            first, last = g.iloc[0], g.iloc[-1]
            emb = app.loc[tid, "embedding"] if tid in app.index else [0.0]
            emb = np.asarray(emb, dtype=float)
            row = {
                "shot_id": self.shot_id, "track_id": int(tid),
                "start_frame": int(first.frame), "end_frame": int(last.frame), "n_frames": int(len(g)),
                "cls": str(first.cls),
                "app_embedding": json.dumps(np.round(emb, 5).tolist()),
                "app_weight": float(app.loc[tid, "weight"]) if tid in app.index else 0.0,
                "first_box": json.dumps([float(first.x1), float(first.y1), float(first.x2), float(first.y2)]),
                "last_box": json.dumps([float(last.x1), float(last.y1), float(last.x2), float(last.y2)]),
                "first_pitch_xy": None, "last_pitch_xy": None,
                "jersey_votes": json.dumps({}), "team_cluster": None,
                "reid_embedding": (reid.loc[tid, "reid"] if reid is not None and tid in reid.index else None),
                "min_swap_conf": float(g.swap_conf.min()),
                "mean_margin": float(g.margin.mean()), "frac_occluded": float((g.occl > 0.3).mean()),
                "mean_height_px": float((g.y2 - g.y1).mean()),
            }
            if H is not None:
                for key, r in (("first_pitch_xy", first), ("last_pitch_xy", last)):
                    if int(r.frame) in H.index and bool(H.loc[int(r.frame), "valid"]):
                        h = H.loc[int(r.frame), [f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(3, 3)
                        foot = np.array([(r.x1 + r.x2) / 2, r.y2, 1.0])
                        p = h @ foot
                        row[key] = json.dumps([float(p[0] / p[2]), float(p[1] / p[2])])
            rows.append(row)
        df = pd.DataFrame(rows)
        Storage.write_df(ctx.out("tracklets.parquet"), df)
        return {"n_tracklets": len(df),
                "median_len": float(df.n_frames.median()) if len(df) else 0.0,
                "short_frac": round(float((df.n_frames < 25).mean()), 3) if len(df) else 0.0}
