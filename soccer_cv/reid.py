"""Appearance embeddings for re-identification (stage 5 feature extractor).

Default: OSNet-AIN x1.0 trained on MSMT17 (Kaiyang Zhou, MIT). 512-d, L2-normalised.
Chosen because the AIN variant is the domain-generalisation model — it transfers
to football crops without fine-tuning better than the plain OSNet. To be replaced
by a part-based model (KPR) with per-part visibility once we fine-tune on our data.

Weights path: $REID_WEIGHTS or <RF_HOME>/../reid/osnet_ain_x1_0_msmt17.pth or ./weights/reid/...
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_weights() -> Path | None:
    cands = [os.environ.get("REID_WEIGHTS"),
             Path(os.environ.get("RF_HOME", "")).parent / "reid" / "osnet_ain_x1_0_msmt17.pth" if os.environ.get("RF_HOME") else None,
             Path("weights/reid/osnet_ain_x1_0_msmt17.pth"), Path("/weights/reid/osnet_ain_x1_0_msmt17.pth")]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


class ReIDEncoder:
    def __init__(self, weights: str | Path | None = None, device: str = "auto", fp16: bool = False):
        import torch
        from .models.osnet_ain import osnet_ain_x1_0
        weights = Path(weights) if weights else find_weights()
        if weights is None:
            raise FileNotFoundError("OSNet weights not found; set REID_WEIGHTS")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.fp16 = device, fp16 and device == "cuda"
        self.model = osnet_ain_x1_0(num_classes=1000, pretrained=False)
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items() if "classifier" not in k}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not [m for m in missing if "classifier" not in m], missing
        self.model.eval().to(device)
        if self.fp16:
            self.model.half()
        log.info("ReID OSNet-AIN loaded from %s on %s", weights, device)

    @staticmethod
    def preprocess(crops: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(crops), 3, 256, 128), dtype=np.float32)
        for i, c in enumerate(crops):
            if c.size == 0:
                continue
            c = cv2.resize(c, (128, 256), interpolation=cv2.INTER_LINEAR)
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out[i] = ((c - MEAN) / STD).transpose(2, 0, 1)
        return out

    def __call__(self, frame_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """boxes (N,4) xyxy -> (N,512) L2-normalised embeddings."""
        import torch
        if len(boxes) == 0:
            return np.zeros((0, 512), dtype=np.float32)
        H, W = frame_bgr.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = int(max(x1, 0)), int(max(y1, 0)), int(min(x2, W)), int(min(y2, H))
            crops.append(frame_bgr[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((0, 0, 3), np.uint8))
        x = torch.from_numpy(self.preprocess(crops)).to(self.device)
        if self.fp16:
            x = x.half()
        with torch.no_grad():
            f = self.model(x).float()
        f = torch.nn.functional.normalize(f, dim=1)
        return f.cpu().numpy()
