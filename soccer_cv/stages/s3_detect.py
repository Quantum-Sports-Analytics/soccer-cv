"""Stage 3 — detection of persons and ball on one shot.

Output: <output_uri>/detections.parquet  with DETECTION_COLUMNS.
`cls` is one of ObjClass values. Until a football-specific fine-tune exists,
the COCO classes `person` and `sports ball` are mapped to PLAYER / BALL and
goalkeeper / referee are resolved later from colour clustering (stage 5/9).

Adaptive tiling: when enabled, the frame is run once at `resolution` and,
if any detection is smaller than `person_min_height_px` * 2, the far-side
band (upper half of the pitch region) is re-run as a second, zoomed tile with
a margin so nobody is cut at the seam. Detections are merged with NMS.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import pandas as pd

from ..core import Stage, StageContext, Storage, frames_iter
from ..schema import DETECTION_COLUMNS, ObjClass

log = logging.getLogger(__name__)

COCO_PERSON, COCO_BALL = 1, 37   # RF-DETR keeps COCO 91-id indexing


class Detector(Protocol):
    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return array (N, 6): x1, y1, x2, y2, score, coco_class_id."""


class RFDETRDetector:
    def __init__(self, model: str = "rfdetr-medium", resolution: int = 960,
                 threshold: float = 0.35, device: str = "auto"):
        import torch
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
        cls = {"rfdetr-nano": RFDETRNano, "rfdetr-small": RFDETRSmall,
               "rfdetr-medium": RFDETRMedium, "rfdetr-large": RFDETRLarge}[model]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # rfdetr requires resolution divisible by 32 (patch 16 x 2 windows)
        res = int(round(resolution / 32) * 32)
        self.model = cls(resolution=res, device=device)
        self.threshold = threshold
        log.info("RF-DETR %s @ %d on %s", model, res, device)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        det = self.model.predict(rgb, threshold=self.threshold)
        if len(det) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        out = np.concatenate([det.xyxy, det.confidence[:, None], det.class_id[:, None]], axis=1)
        return out.astype(np.float32)


class ReplayDetector:
    """Reads detections from an existing parquet (tests, tracker tuning without GPU)."""

    def __init__(self, parquet_uri: str):
        df = Storage.read_df(parquet_uri)
        self.by_frame = {int(k): g for k, g in df.groupby("frame")}
        self.cls_to_coco = {ObjClass.BALL.value: COCO_BALL}

    def __call__(self, frame_bgr, frame_idx: int | None = None):
        g = self.by_frame.get(frame_idx)
        if g is None:
            return np.zeros((0, 6), dtype=np.float32)
        coco = g["cls"].map(lambda c: self.cls_to_coco.get(c, COCO_PERSON)).to_numpy()
        return np.column_stack([g[["x1", "y1", "x2", "y2", "score"]].to_numpy(), coco]).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.6) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return np.array(keep, dtype=int)


def detect_with_tiling(det: Detector, frame: np.ndarray, min_h: int, tiles: str) -> np.ndarray:
    out = det(frame)
    if tiles != "adaptive" or len(out) == 0:
        return out
    persons = out[out[:, 5] == COCO_PERSON]
    if len(persons) and (persons[:, 3] - persons[:, 1]).min() < 2 * min_h:
        H, W = frame.shape[:2]
        # far-side band: top 55% of the frame, with 6% margin so nobody is split
        y_lo, y_hi = 0, int(0.55 * H)
        band = frame[y_lo:y_hi]
        d2 = det(band)
        if len(d2):
            d2[:, [1, 3]] += y_lo
            out = np.vstack([out, d2])
            keep = nms(out[:, :4], out[:, 4], 0.6)
            out = out[keep]
    return out


def build_detector(cfg: dict, replay_uri: str | None = None) -> Detector:
    if replay_uri:
        return ReplayDetector(replay_uri)
    p = cfg.get("detect", {})
    return RFDETRDetector(model=p.get("model", "rfdetr-medium"),
                          resolution=int(p.get("resolution", 960)),
                          threshold=float(p.get("score_threshold", 0.35)),
                          device=cfg.get("runtime", {}).get("device", "auto"))


class DetectStage(Stage):
    """Input: ingest output (video.mp4 + shots.json). Runs on ONE shot given by `shot_id`."""
    name = "s3_detect"
    config_key = "detect"

    def __init__(self, cfg: dict, shot_id: str, replay_uri: str | None = None, detector=None):
        super().__init__(cfg)
        self.shot_id = shot_id
        self.detector = detector or build_detector(cfg, replay_uri)

    def run(self, ctx: StageContext) -> dict:
        from .s0_ingest import load_shots
        shot = next(s for s in load_shots(ctx.input_uri) if s.shot_id == self.shot_id)
        video = Storage.localize(ctx.inp("video.mp4"), ctx.workdir)
        min_h = int(self.params.get("person_min_height_px", 14))
        tiles = self.params.get("tiles", "adaptive")
        rows = []
        for i, f in frames_iter(video, shot.start_frame, shot.end_frame):
            if isinstance(self.detector, ReplayDetector):
                d = self.detector(f, i)
            else:
                d = detect_with_tiling(self.detector, f, min_h, tiles)
            for x1, y1, x2, y2, s, c in d:
                c = int(c)
                if c == COCO_PERSON:
                    if (y2 - y1) < min_h:
                        continue
                    cls = ObjClass.PLAYER.value
                elif c == COCO_BALL:
                    cls = ObjClass.BALL.value
                else:
                    continue
                rows.append((i, float(x1), float(y1), float(x2), float(y2), float(s), cls))
            if (i - shot.start_frame) % 100 == 0:
                log.info("detect %s frame %d", self.shot_id, i)
        df = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
        Storage.write_df(ctx.out("detections.parquet"), df)
        n_frames = shot.end_frame - shot.start_frame + 1
        return {"n_frames": n_frames, "n_det": len(df),
                "persons_per_frame": round(float((df.cls != "ball").sum()) / max(n_frames, 1), 2),
                "ball_frames": int(df[df.cls == "ball"].frame.nunique())}
