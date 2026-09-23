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


def _probe(path: str) -> dict:
    """Container-level stream properties (ffprobe): size, nominal and average frame rate."""
    import json as _json, subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,avg_frame_rate", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    st = _json.loads(out)["streams"][0]
    rate = lambda s: (lambda a, b: float(a) / float(b) if float(b) else 0.0)(*s.split("/"))   # noqa: E731
    return {"width": int(st["width"]), "height": int(st["height"]),
            "fps_nominal": rate(st["r_frame_rate"]), "fps_avg": rate(st["avg_frame_rate"])}


def normalize_video(src: str, dst: str, max_height: int = 1080, fps: float = 25.0) -> dict:
    """Bring any input to the working profile the pipeline is tuned for: height <= max_height,
    constant `fps`. Every pixel threshold (line-mask kernels, tolerances, box margins) and every
    per-frame quantity (track ages, ball speed gate, windows) is expressed for that profile.

    Pass-through (no re-encode) when the input already conforms. Measured failure without it:
    a 3024x1716, variable ~59.7 fps screen recording -> degenerate calibration on 5 s of 8.6,
    every player filtered as off-pitch.
    """
    import subprocess
    pr = _probe(src)
    vfr = pr["fps_nominal"] > 0 and abs(pr["fps_nominal"] - pr["fps_avg"]) / pr["fps_nominal"] > 0.002
    need = pr["height"] > max_height or abs(pr["fps_avg"] - fps) > 0.05 or vfr
    info = {"source": {**pr, "vfr": bool(vfr)}, "normalized": bool(need)}
    if not need:
        return info
    vf = f"scale=-2:'min({max_height},ih)':flags=area,fps={fps:g}"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf, "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", dst]
    subprocess.run(cmd, check=True)
    log.info("ingest: normalized %dx%d @ %.2f fps%s -> %s", pr["width"], pr["height"], pr["fps_avg"],
             " (variable)" if vfr else "", vf)
    return info


class IngestStage(Stage):
    name = "s0_ingest"
    config_key = "ingest"

    def run(self, ctx: StageContext) -> dict:
        src = Storage.localize(ctx.input_uri, ctx.workdir)
        norm = {"normalized": False}
        if bool(self.params.get("normalize", True)):
            dst = str(Path(ctx.workdir) / "normalized.mp4")
            norm = normalize_video(src, dst, int(self.params.get("max_height", 1080)), float(self.params.get("target_fps", 25.0)))
            if norm["normalized"]:
                src = dst
        meta = video_meta(src)
        meta.update(norm)
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
        return {"n_frames": n, "n_shots": len(shots), "n_cuts": len(cuts), "normalized": bool(norm["normalized"]),
                "main_shots": sum(s.shot_type == ShotType.MAIN for s in shots)}


def load_shots(uri: str) -> list[Shot]:
    raw = Storage.read_json(Storage.join(uri, "shots.json"))
    return [Shot(**{**r, "shot_type": ShotType(r["shot_type"])}) for r in raw]
