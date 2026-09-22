"""Stage 0 — ingest: probe the video, detect shot cuts, classify shot type.

Output layout under <output_uri>/:
    video.mp4        local copy / GCS copy of the source (Tier A tasks read it)
    shots.json       list[Shot]  — the unit of parallelism for Tier A
    meta.json        fps, size, frame count (decoded, not container-reported)

Shot-type classification is a heuristic placeholder (green fraction + edge
density) until a small classifier is trained; it is enough to separate a
main tactical camera from close-ups and graphics on typical broadcasts.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ..core import Stage, StageContext, Storage, frames_iter, video_meta
from ..schema import Shot, ShotType

log = logging.getLogger(__name__)


def _hsv_hist(frame_small: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    return h / max(h.sum(), 1e-9)


def _green_fraction(frame_small: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    return float(mask.mean() / 255.0)


def classify_shot(green_frac: float, person_scale_hint: float | None = None) -> ShotType:
    """Cheap heuristic: a main tactical camera is mostly pitch."""
    if green_frac > 0.45:
        return ShotType.MAIN
    if green_frac > 0.20:
        return ShotType.CLOSEUP
    return ShotType.OTHER


class IngestStage(Stage):
    name = "s0_ingest"
    config_key = "ingest"

    def run(self, ctx: StageContext) -> dict:
        src = Storage.localize(ctx.input_uri, ctx.workdir)
        meta = video_meta(src)
        thr = float(self.params.get("hist_cut_threshold", 0.35))
        min_len = int(self.params.get("min_shot_frames", 12))

        prev = None
        cuts: list[int] = []
        greens: list[float] = []
        n = 0
        for i, f in frames_iter(src):
            small = cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA)
            h = _hsv_hist(small)
            if prev is not None and 0.5 * np.abs(h - prev).sum() > thr:
                cuts.append(i)
            prev = h
            if i % 5 == 0:
                greens.append(_green_fraction(small))
            n = i + 1
        meta["n_frames"] = n

        # merge cuts closer than min_len (wipes / flashes produce doublets)
        bounds = [0]
        for c in cuts:
            if c - bounds[-1] >= min_len:
                bounds.append(c)
        bounds.append(n)

        shots: list[Shot] = []
        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1] - 1
            g = [greens[j] for j in range(s // 5, min(e // 5 + 1, len(greens)))]
            st = classify_shot(float(np.median(g)) if g else 0.0)
            shots.append(Shot(shot_id=f"shot_{k:04d}", start_frame=s, end_frame=e,
                              shot_type=st, fps=meta["fps"], width=meta["width"],
                              height=meta["height"]))

        Storage.write_json(ctx.out("meta.json"), meta)
        Storage.write_json(ctx.out("shots.json"), [asdict(s) | {"shot_type": s.shot_type.value} for s in shots])
        if Storage.join(ctx.output_uri, "video.mp4") != str(src):
            Storage.upload_file(src, ctx.out("video.mp4"))
        log.info("ingest: %d frames, %d shots, %d cuts", n, len(shots), len(cuts))
        return {"n_frames": n, "n_shots": len(shots), "n_cuts": len(cuts),
                "main_shots": sum(s.shot_type == ShotType.MAIN for s in shots)}


def load_shots(uri: str) -> list[Shot]:
    raw = Storage.read_json(Storage.join(uri, "shots.json"))
    return [Shot(**{**r, "shot_type": ShotType(r["shot_type"])}) for r in raw]
