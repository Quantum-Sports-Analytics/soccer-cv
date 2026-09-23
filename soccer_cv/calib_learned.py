"""Learned field calibration (PnLCalib keypoints + lines) used as the *initialisation* of stage 2.

Why: the classical white-line chamfer fit has no global view of the pitch. On a tight,
oblique shot of a penalty area it can lock onto a degenerate solution (one well-aligned
touchline, nothing else) that its own validity test accepts. Measured on the PSG-Arsenal
screen recording: 36 camera positions tested, none beats the degenerate cost. A learned
keypoint/line detector recognises *which* markings are visible (penalty-box corners, goal
posts, arc), which is exactly the missing global information.

PnLCalib (Gutierrez-Perez & Agudo, CVIU 2025; github.com/mguti97/PnLCalib, GPL-2.0, weights
trained on SoccerNet-Calibration) is NOT vendored: set PNLCALIB_DIR to a checkout that also
holds weights/SV_kp and weights/SV_lines (scripts/fetch_pnlcalib.sh). Licence obligation
recorded in docs/architecture_and_difficulties.md §7 (R&D posture, cleanup before shipping).

Conventions: PnLCalib works in pitch-centred metres with y towards the camera side and z
down; ours has y towards the far touchline and z up. The two differ by a 180 deg rotation
about x: (x, y, z)_ours = (x, -y, -z)_pnl. The returned projection is converted, then fitted
by our 6-parameter broadcast camera (roll 0) so every downstream stage is unchanged; the fit
residual is reported and a large one rejects the frame.
"""
from __future__ import annotations

import logging
import os
import sys

import cv2
import numpy as np

log = logging.getLogger(__name__)

_MODELS = None
PNL_TO_OURS = np.diag([1.0, -1.0, -1.0, 1.0])       # homogeneous: ours -> pnl coordinates, applied on the right of P


def available() -> bool:
    d = os.environ.get("PNLCALIB_DIR", "")
    return bool(d) and all(os.path.exists(os.path.join(d, p)) for p in ("weights/SV_kp", "weights/SV_lines", "model/cls_hrnet.py"))


def _load(device: str):
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    import torch
    import yaml
    d = os.environ["PNLCALIB_DIR"]
    if d not in sys.path:
        sys.path.insert(0, d)
    from model.cls_hrnet import get_cls_net                       # noqa: E402  (PnLCalib modules)
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    cfg = yaml.safe_load(open(os.path.join(d, "config/hrnetv2_w48.yaml")))
    cfg_l = yaml.safe_load(open(os.path.join(d, "config/hrnetv2_w48_l.yaml")))
    m = get_cls_net(cfg); m.load_state_dict(torch.load(os.path.join(d, "weights/SV_kp"), map_location=device)); m.to(device).eval()
    ml = get_cls_net_l(cfg_l); ml.load_state_dict(torch.load(os.path.join(d, "weights/SV_lines"), map_location=device)); ml.to(device).eval()
    _MODELS = (m, ml, device)
    log.info("PnLCalib loaded from %s on %s", d, device)
    return _MODELS


def pnl_projection(frame: np.ndarray, device: str = "cpu", kp_threshold: float = 0.3434,
                   line_threshold: float = 0.7867, refine: bool = True) -> np.ndarray | None:
    """3x4 projection in OUR pitch coordinates (metres, centre origin, y far, z up), or None."""
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from PIL import Image
    m, ml, dev = _load(device)
    from utils.utils_calib import FramebyFrameCalib                # noqa: E402
    from utils.utils_heatmap import (complete_keypoints, coords_to_dict,  # noqa: E402
                                     get_keypoints_from_heatmap_batch_maxpool,
                                     get_keypoints_from_heatmap_batch_maxpool_l)
    h0, w0 = frame.shape[:2]
    x = TF.to_tensor(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).float().unsqueeze(0)
    if x.shape[-1] != 960:
        x = T.Resize((540, 960))(x)
    x = x.to(dev)
    _, _, h, w = x.shape
    with torch.no_grad():
        hm, hml = m(x), ml(x)
    kp = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1]), threshold=kp_threshold)
    ln = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool_l(hml[:, :-1]), threshold=line_threshold)
    kp, ln = complete_keypoints(kp[0], ln[0], w=w, h=h, normalize=True)
    cam = FramebyFrameCalib(iwidth=w0, iheight=h0, denormalize=True)
    cam.update(kp, ln)
    try:
        res = cam.heuristic_voting(refine_lines=refine)
    except Exception as e:                                           # noqa: BLE001  (degenerate point sets)
        log.debug("PnLCalib failed: %s", e)
        return None
    if res is None:
        return None
    cp = res["cam_params"]
    K = np.array([[cp["x_focal_length"], 0, cp["principal_point"][0]], [0, cp["y_focal_length"], cp["principal_point"][1]], [0, 0, 1]])
    It = np.eye(4)[:-1]; It[:, -1] = -np.asarray(cp["position_meters"])
    P = K @ (np.asarray(cp["rotation_matrix"]) @ It)
    return P @ PNL_TO_OURS


def params_from_projection(P: np.ndarray, img_w: int, img_h: int, camera_P, model_pts: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Our (cx, cy, cz, pan, tilt, f) that best reproduces P on the in-view pitch model points.
    Returns (params, median reprojection residual in px)."""
    from scipy.optimize import least_squares
    X = np.c_[model_pts, np.zeros(len(model_pts)), np.ones(len(model_pts))]
    q = X @ P.T
    ok = q[:, 2] > 1e-6
    uv = q[ok, :2] / q[ok, 2:]
    inside = (uv[:, 0] > -200) & (uv[:, 0] < img_w + 200) & (uv[:, 1] > -200) & (uv[:, 1] < img_h + 200)
    X, uv = X[ok][inside], uv[inside]
    if len(X) < 30:
        return None
    K_d, _, C_h = cv2.decomposeProjectionMatrix(P.astype(np.float64))[:3]
    C = (C_h[:3] / C_h[3]).ravel()                                       # camera centre
    f0 = float(abs(K_d[0, 0] / K_d[2, 2]))
    if not np.isfinite(f0) or f0 < 300 or f0 > 30000:
        f0 = 3000.0

    def res(p):
        Q = camera_P(p, img_w, img_h); r = X @ Q.T
        return ((r[:, :2] / np.maximum(r[:, 2:], 1e-6)) - uv).ravel()
    best = None
    for pan0 in range(-80, 81, 20):
        for tilt0 in (8.0, 15.0, 25.0):
            p0 = np.array([C[0], C[1], C[2], float(pan0), tilt0, f0])
            r = least_squares(res, p0, x_scale=[1, 1, 1, 1, 1, 100], loss="soft_l1", f_scale=5.0, max_nfev=200)
            if best is None or r.cost < best.cost:
                best = r
    resid = np.median(np.abs(res(best.x)).reshape(-1, 2).max(1))
    return best.x, float(resid)


def ptz_from_projection(P: np.ndarray, pos: np.ndarray, x0: np.ndarray, img_w: int, img_h: int, camera_P,
                        model_pts: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Same fit with the camera position fixed (a broadcast main camera does not translate in a shot)."""
    from scipy.optimize import least_squares
    X = np.c_[model_pts, np.zeros(len(model_pts)), np.ones(len(model_pts))]
    q = X @ P.T; ok = q[:, 2] > 1e-6
    uv = q[ok, :2] / q[ok, 2:]
    inside = (uv[:, 0] > -200) & (uv[:, 0] < img_w + 200) & (uv[:, 1] > -200) & (uv[:, 1] < img_h + 200)
    X, uv = X[ok][inside], uv[inside]
    if len(X) < 30:
        return None

    def res(x):
        Q = camera_P(np.r_[pos, x], img_w, img_h); r = X @ Q.T
        return ((r[:, :2] / np.maximum(r[:, 2:], 1e-6)) - uv).ravel()
    r = least_squares(res, np.asarray(x0, float), x_scale=[1, 1, 100], loss="soft_l1", f_scale=5.0, max_nfev=200)
    return np.r_[pos, r.x], float(np.median(np.abs(res(r.x)).reshape(-1, 2).max(1)))


def model_displacement(H1: np.ndarray, H2: np.ndarray, img_w: int, img_h: int, model_pts: np.ndarray) -> float:
    """Median image distance (px) between the pitch model projected by two homographies, in-view points only."""
    X = np.c_[model_pts, np.ones(len(model_pts))]
    a, b = X @ H1.T, X @ H2.T
    ok = (a[:, 2] > 1e-6) & (b[:, 2] > 1e-6)
    a, b = a[ok, :2] / a[ok, 2:], b[ok, :2] / b[ok, 2:]
    inside = (a[:, 0] >= 0) & (a[:, 0] < img_w) & (a[:, 1] >= 0) & (a[:, 1] < img_h)
    return float(np.median(np.linalg.norm(a[inside] - b[inside], axis=1))) if inside.sum() >= 10 else float("inf")
