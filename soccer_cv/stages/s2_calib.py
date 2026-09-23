"""Stages 1-2 — field calibration from white pitch markings (no learned model).

Stage 1 (line evidence): a white-line mask restricted to the grass region, with
player boxes blanked out, turned into a distance transform.
Stage 2 (fit): a parametric broadcast camera (position, pan, tilt, focal; roll 0)
projects the FIFA pitch model; the cost is the chamfer distance of projected
model-line samples to detected line pixels. Coarse grid over camera parameters
on the first frame of a shot, then per-frame refinement from the previous
solution (calibrated at `every_n_frames`, interpolated in between).

Outputs <output_uri>/calib.parquet: frame, H (9 floats image<-pitch), valid,
err_px (mean chamfer), coverage, cam params. Pitch coordinates: metres, origin
at the pitch centre, x along the long axis, y towards the far touchline.

Why parametric and not a free 8-DOF homography: a camera has 6 meaningful
degrees of freedom here and strong priors (behind a touchline, 10-30 m high);
searching that space is what makes the fit converge without keypoints.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from ..core import Stage, StageContext, Storage, frames_iter
from ..teams import GRASS_HI, GRASS_LO

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- pitch model
L, W = 105.0, 68.0


def pitch_model(L: float = L, W: float = W) -> list[np.ndarray]:
    """Line segments / arcs of the pitch as arrays of (x, y) sample points in metres."""
    hl, hw = L / 2, W / 2
    segs = []

    def seg(a, b, n=None):
        a, b = np.array(a, float), np.array(b, float)
        n = n or max(2, int(np.linalg.norm(b - a) / 1.0))
        segs.append(np.linspace(a, b, n))

    seg((-hl, -hw), (hl, -hw)); seg((-hl, hw), (hl, hw))           # touchlines
    seg((-hl, -hw), (-hl, hw)); seg((hl, -hw), (hl, hw))           # goal lines
    seg((0, -hw), (0, hw))                                          # halfway
    for s in (-1, 1):                                               # penalty + goal areas
        x0 = s * hl
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            xi = x0 - s * depth
            seg((x0, -half), (xi, -half)); seg((x0, half), (xi, half)); seg((xi, -half), (xi, half))
        # penalty arc
        px = x0 - s * 11.0
        th = np.linspace(-1.0, 1.0, 40)
        ang = np.arccos(5.5 / 9.15)
        t = np.linspace(-ang, ang, 40)
        arc = np.stack([px - s * 9.15 * np.cos(t), 9.15 * np.sin(t)], 1)
        segs.append(arc)
    t = np.linspace(0, 2 * np.pi, 120)                              # centre circle
    segs.append(np.stack([9.15 * np.cos(t), 9.15 * np.sin(t)], 1))
    return segs


MODEL = pitch_model()
MODEL_PTS = np.vstack(MODEL)


# ---------------------------------------------------------------- camera
def camera_H(params: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """params = [cx, cy, cz, pan_deg, tilt_deg, f_px]. Returns 3x3 H: pitch (x,y,1) -> image.

    World: x along the pitch, y towards the far touchline, z up. Camera: x right, y down,
    z forward. pan=0 looks along +y; tilt>0 looks down; roll is fixed at 0."""
    cx, cy, cz, pan, tilt, f = params
    t, ph = np.deg2rad(tilt), np.deg2rad(pan)
    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])      # cam <- world, looking +y, level
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Rz = np.array([[np.cos(ph), -np.sin(ph), 0], [np.sin(ph), np.cos(ph), 0], [0, 0, 1]])
    Rm = Rx @ base @ Rz
    tvec = -Rm @ np.array([cx, cy, cz])
    K = np.array([[f, 0, img_w / 2], [0, f, img_h / 2], [0, 0, 1.0]])
    P = K @ np.hstack([Rm, tvec[:, None]])
    return P[:, [0, 1, 3]]


def project(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    ok = p[:, 2] > 1e-6
    uv = np.full((len(pts), 2), np.nan)
    uv[ok] = p[ok, :2] / p[ok, 2:3]
    return uv, ok


# ---------------------------------------------------------------- line evidence
def line_mask(frame: np.ndarray, boxes: np.ndarray | None = None) -> np.ndarray:
    """White thin markings on grass. Returns uint8 mask at frame resolution."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, GRASS_LO, GRASS_HI)
    grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grass)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        grass = ((lab == big) * 255).astype(np.uint8)
    # strictly inside the grass: boards, crowd and the stands' white edges must not enter the mask
    region = cv2.erode(grass, np.ones((9, 9), np.uint8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    white = (tophat > 28) & (hsv[..., 1] < 90) & (hsv[..., 2] > 120) & (region > 0)
    m = (white * 255).astype(np.uint8)
    if boxes is not None:
        for x1, y1, x2, y2 in boxes.astype(int):
            cv2.rectangle(m, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), 0, -1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # drop tiny blobs (texture, socks)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 60
    return (keep[lab] * 255).astype(np.uint8)


def render_model(H: np.ndarray, img_w: int, img_h: int, scale: float = 1.0) -> np.ndarray:
    """Binary image of the projected pitch model (for the reverse chamfer term)."""
    m = np.zeros((int(img_h * scale), int(img_w * scale)), np.uint8)
    for seg in MODEL:
        uv, ok = project(H, seg)
        uv = uv[ok] * scale
        good = np.all(np.isfinite(uv), 1) & (np.abs(uv).max(1) < 8 * max(img_w, img_h))
        pts = np.round(uv[good]).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, 255, 1)
    return m


def _dense_model(step_m: float = 0.5) -> np.ndarray:
    out = []
    for seg in MODEL:
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        t = np.concatenate([[0], np.cumsum(d)])
        n = max(2, int(t[-1] / step_m))
        tt = np.linspace(0, t[-1], n)
        out.append(np.stack([np.interp(tt, t, seg[:, 0]), np.interp(tt, t, seg[:, 1])], 1))
    return np.vstack(out)


MODEL_COARSE = _dense_model(0.5)     # coarse grid search (quarter resolution)
MODEL_DENSE = _dense_model(0.1)      # refinement: <= a few px between samples at broadcast focal lengths


def chamfer_cost(H: np.ndarray, dt: np.ndarray, img_w: int, img_h: int, mask_pts: np.ndarray | None = None,
                 trunc: float = 30.0, scale: float = 1.0, detail: bool = False, model: np.ndarray | None = None):
    """Symmetric truncated chamfer, in full-resolution pixels.

    forward : model points in view -> nearest detected line pixel (distance transform lookup)
    reverse : detected line pixels -> nearest projected model point (KD-tree, no rendering)
    `dt` and `mask_pts` are at `scale` x image resolution. With detail=True also returns
    (coverage, inlier_err_px): the fraction of in-view model points within 4 px of a line,
    and the mean distance of those matched model points — the two quantities used for
    validity, since neither is inflated by white pixels that are not pitch lines."""
    from scipy.spatial import cKDTree
    uv, ok = project(H, MODEL_DENSE if model is None else model)
    inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    if inside.sum() < 60:
        return (1e3, 0.0, 99.0, 0.0) if detail else (1e3, 0.0)
    uvs = uv[inside] * scale
    d = dt[uvs[:, 1].astype(int), uvs[:, 0].astype(int)] / scale
    fwd = float(np.minimum(d, trunc).mean())
    matched = d < 4.0
    cov = float(matched.mean())
    rev, explained = 0.0, 0.0
    if mask_pts is not None and len(mask_pts):
        tree = cKDTree(uvs)
        dr, _ = tree.query(mask_pts, k=1, distance_upper_bound=trunc * scale)
        dr = np.where(np.isfinite(dr), dr, trunc * scale) / scale
        rev = float(dr.mean()); explained = float((dr < 4.0).mean())
    cost = fwd + rev
    if detail:
        return cost, cov, (float(d[d < 8.0].mean()) if (d < 8.0).any() else 99.0), explained
    return cost, cov


def mask_points(m: np.ndarray, scale: float, max_pts: int = 1500) -> np.ndarray:
    small = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale != 1.0 else m
    ys, xs = np.nonzero(small)
    if len(xs) > max_pts:
        idx = np.random.default_rng(0).choice(len(xs), max_pts, replace=False); xs, ys = xs[idx], ys[idx]
    return np.stack([xs, ys], 1)


def _coarse_inputs(m: np.ndarray, sc: float = 0.25, n_pts: int = 600):
    small = cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    return cv2.distanceTransform(255 - small, cv2.DIST_L2, 3), mask_points(m, sc, n_pts), sc


def coarse_search(m: np.ndarray, img_w: int, img_h: int, topk: int = 60, trunc: float = 80.0) -> list[np.ndarray]:
    """Grid over broadcast-camera priors at quarter resolution, with a wide truncation so the
    cost is smooth enough for local descent from a grid point. Returns the top-k distinct seeds."""
    dt, mp, sc = _coarse_inputs(m)
    cands = []
    # Camera behind the near touchline only (y < 0). The pitch is symmetric under a 180 degree
    # rotation, so a far-side camera is the same image with (x, y) -> (-x, -y): searching one
    # side loses nothing and fixes the left/right convention (TV main camera = near side).
    for side in (-1.0,):
        for cx in (-20.0, 0.0, 20.0):
            for dist in (20.0, 40.0, 70.0):
                for cz in (15.0, 25.0, 45.0):
                    for pan in np.arange(-50, 51, 10.0):
                        for tilt in (8, 13, 18, 24, 30):
                            for f in (1500, 2200, 3200, 4600, 6500, 9000):
                                p = np.array([cx, side * (W / 2 + dist), cz, (pan if side < 0 else 180 + pan), tilt, f])
                                c, _ = chamfer_cost(camera_H(p, img_w, img_h), dt, img_w, img_h, mp, scale=sc,
                                                    model=MODEL_COARSE, trunc=trunc)
                                cands.append((c, p))
    cands.sort(key=lambda t: t[0])
    out = []
    for c, p in cands:
        if all(np.abs((p - q) / np.array([10, 10, 8, 8, 4, 1000])).max() >= 1 for q in out):
            out.append(p)
        if len(out) >= topk:
            break
    return out


def _quick_refine(p0: np.ndarray, dt, mp, sc, img_w, img_h, trunc: float, maxfev: int) -> tuple[np.ndarray, float]:
    scale = np.array([8.0, 8.0, 5.0, 6.0, 4.0, 600.0])

    def f(z):
        return chamfer_cost(camera_H(p0 + z * scale, img_w, img_h), dt, img_w, img_h, mp, scale=sc,
                            model=MODEL_COARSE, trunc=trunc)[0]
    res = minimize(f, np.zeros(6), method="Powell", options={"xtol": 1e-2, "ftol": 1e-3, "maxfev": maxfev})
    return p0 + res.x * scale, float(res.fun)


def fit_frame(m: np.ndarray, img_w: int, img_h: int, topk: int = 60, finalists: int = 5):
    """Grid -> quick descent from the `topk` best seeds (smooth coarse cost) -> the `finalists`
    best are refined at half resolution with the sharp cost -> lowest cost wins."""
    dt, mp, sc = _coarse_inputs(m)
    seeds = coarse_search(m, img_w, img_h, topk)
    quick = sorted((_quick_refine(p0, dt, mp, sc, img_w, img_h, 80.0, 250) for p0 in seeds), key=lambda t: t[1])
    quick = [_quick_refine(p, dt, mp, sc, img_w, img_h, 30.0, 300) for p, _ in quick[:3 * finalists]]
    quick.sort(key=lambda t: t[1])
    best = None
    for p, _ in quick[:finalists]:
        r = refine(p, m, img_w, img_h, iters=1, maxfev=800)
        if best is None or r[1] < best[1]:
            best = r
    return best


def refine(p0: np.ndarray, m: np.ndarray, img_w: int, img_h: int, iters: int = 2,
           maxfev: int = 1500) -> tuple[np.ndarray, float, float, float, float]:
    """Local optimisation of the 6 camera parameters.
    Returns (params, cost, model_coverage, inlier_err_px, lines_explained)."""
    sc = 0.5
    small = cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    dt = cv2.distanceTransform(255 - small, cv2.DIST_L2, 3)
    mp = mask_points(m, sc, 1200)
    scale = np.array([8.0, 8.0, 5.0, 6.0, 4.0, 600.0])

    def f(z):
        return chamfer_cost(camera_H(p0 + z * scale, img_w, img_h), dt, img_w, img_h, mp, scale=sc)[0]
    z = np.zeros(6)
    for _ in range(iters):
        z = minimize(f, z, method="Powell", options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": maxfev}).x
    p = p0 + z * scale
    dt_full = cv2.distanceTransform(255 - m, cv2.DIST_L2, 3)
    c, cov, err, expl = chamfer_cost(camera_H(p, img_w, img_h), dt_full, img_w, img_h, mask_points(m, 1.0, 2500), detail=True)
    return p, c, cov, err, expl


def refine_ptz(pos: np.ndarray, ptz0: np.ndarray, m: np.ndarray, img_w: int, img_h: int, maxfev: int = 600):
    """Refine pan / tilt / focal with the camera position fixed. Same return shape as `refine`."""
    sc = 0.5
    small = cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    dt = cv2.distanceTransform(255 - small, cv2.DIST_L2, 3)
    mp = mask_points(m, sc, 1200)
    scale = np.array([4.0, 2.0, 400.0])

    def f(z):
        q = ptz0 + z * scale
        return chamfer_cost(camera_H(np.r_[pos, q], img_w, img_h), dt, img_w, img_h, mp, scale=sc)[0]
    z = minimize(f, np.zeros(3), method="Powell", options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": maxfev}).x
    params = np.r_[pos, ptz0 + z * scale]
    dt_full = cv2.distanceTransform(255 - m, cv2.DIST_L2, 3)
    c, cov, err, expl = chamfer_cost(camera_H(params, img_w, img_h), dt_full, img_w, img_h, mask_points(m, 1.0, 2500), detail=True)
    return params, c, cov, err, expl


def search_ptz(pos: np.ndarray, m: np.ndarray, img_w: int, img_h: int, finalists: int = 4):
    """Grid over pan / tilt / focal for a known camera position, then local refinement."""
    dt, mp, sc = _coarse_inputs(m)
    cands = []
    for pan in np.arange(-60, 61, 3.0):
        for tilt in np.arange(6, 32, 1.5):
            for f in np.geomspace(1400, 9500, 18):
                H = camera_H(np.r_[pos, pan, tilt, f], img_w, img_h)
                cands.append((chamfer_cost(H, dt, img_w, img_h, mp, scale=sc, model=MODEL_COARSE, trunc=80.0)[0],
                              np.array([pan, tilt, f])))
    cands.sort(key=lambda t: t[0])
    best = None
    for _, q in cands[:finalists * 3:3]:
        r = refine_ptz(pos, q, m, img_w, img_h)
        if best is None or r[1] < best[1]:
            best = r
    return best


def pitch_polygon_mask(H: np.ndarray, img_w: int, img_h: int, margin_m: float = 1.0) -> np.ndarray:
    """uint8 mask of the pitch rectangle (with margin) in image space."""
    hl, hw = L / 2 + margin_m, W / 2 + margin_m
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    # sample edges densely so points behind the camera can be dropped
    pts = np.vstack([np.linspace(corners[i], corners[(i + 1) % 4], 60) for i in range(4)])
    uv, ok = project(H, pts)
    uv = uv[ok]
    m = np.zeros((img_h, img_w), np.uint8)
    if len(uv) >= 3:
        cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 255)
    return m


def camera_P(params: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Full 3x4 projection (pitch metres x, y, z-up -> image) for the parametric camera."""
    H = camera_H(params, img_w, img_h)
    cx, cy, cz, pan, tilt, f = params
    t, ph = np.deg2rad(tilt), np.deg2rad(pan)
    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Rz = np.array([[np.cos(ph), -np.sin(ph), 0], [np.sin(ph), np.cos(ph), 0], [0, 0, 1]])
    K = np.array([[f, 0, img_w / 2], [0, f, img_h / 2], [0, 0, 1.0]])
    z_col = K @ (Rx @ base @ Rz)[:, 2]
    return np.column_stack([H[:, 0], H[:, 1], z_col, H[:, 2]])


def feet_pitch_xy(params: np.ndarray, boxes: np.ndarray, img_w: int, img_h: int,
                  person_h: float = 1.8, edge_px: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Pitch position (metres) of each person's feet, and a flag for boxes cut by the bottom edge.

    For a box whose bottom touches the image border the feet are not visible: the box
    bottom is the border, not the feet. We then search down the box's image column for
    the ground point whose head (person_h above it) projects onto the box top — the
    height prior turns a truncated box into a foot estimate. Boxes truncated at the top
    as well (both ends unknown) keep the border point, flagged."""
    H = camera_H(params, img_w, img_h); Hi = np.linalg.inv(H); P = camera_P(params, img_w, img_h)
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    cxs = (boxes[:, 0] + boxes[:, 2]) / 2
    uv = np.c_[cxs, boxes[:, 3], np.ones(len(boxes))] @ Hi.T
    xy = uv[:, :2] / uv[:, 2:3]
    cut = boxes[:, 3] >= img_h - edge_px
    for i in np.where(cut & (boxes[:, 1] > edge_px))[0]:
        h_box = boxes[i, 3] - boxes[i, 1]
        vs = np.linspace(boxes[i, 3], boxes[i, 3] + 4 * max(h_box, 20), 200)
        g = np.c_[np.full_like(vs, cxs[i]), vs, np.ones_like(vs)] @ Hi.T
        g = g[:, :2] / g[:, 2:3]
        head = np.c_[g, np.full(len(g), person_h), np.ones(len(g))] @ P.T
        head_v = head[:, 1] / head[:, 2]
        k = int(np.argmin(np.abs(head_v - boxes[i, 1])))
        xy[i] = g[k]
    return xy, cut


def on_pitch(xy: np.ndarray, margin_m: float, L: float = L, W: float = W) -> np.ndarray:
    return (np.abs(xy[:, 0]) <= L / 2 + margin_m) & (np.abs(xy[:, 1]) <= W / 2 + margin_m)


def load_calib_params(uri: str) -> dict[int, np.ndarray] | None:
    """frame -> camera params [cx, cy, cz, pan, tilt, f] for valid frames."""
    if not Storage.exists(Storage.join(uri, "calib.parquet")):
        return None
    df = Storage.read_df(Storage.join(uri, "calib.parquet"))
    df = df[df.valid]
    cols = ["cx", "cy", "cz", "pan", "tilt", "f"]
    return {int(f): v for f, v in zip(df.frame, df[cols].to_numpy(dtype=float))}


def image_to_pitch(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    Hi = np.linalg.inv(H)
    p = np.c_[uv, np.ones(len(uv))] @ Hi.T
    return p[:, :2] / p[:, 2:3]


# ---------------------------------------------------------------- inter-keyframe camera motion
# Keyframes are fitted on the pitch lines once per `every_n_frames`. In between, a linear
# interpolation of pan / tilt / zoom is wrong as soon as the operator pans non-uniformly
# (measured: players "running" at 7-11 m/s during a whip pan). A fixed-position broadcast
# camera only rotates and zooms, so consecutive images are related by K2 R2 R1^T K1^-1:
# we estimate that from background feature matches (players masked out), chain it forward
# from keyframe a and backward from keyframe b, and blend the two chains so the result is
# exact at both keyframes (no drift) and follows the true motion in between.

def _rot(pan: float, tilt: float) -> np.ndarray:
    t, ph = np.deg2rad(tilt), np.deg2rad(pan)
    base = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Rz = np.array([[np.cos(ph), -np.sin(ph), 0], [np.sin(ph), np.cos(ph), 0], [0, 0, 1]])
    return Rx @ base @ Rz


def _K(f: float, img_w: int, img_h: int) -> np.ndarray:
    return np.array([[f, 0, img_w / 2], [0, f, img_h / 2], [0, 0, 1.0]])


def motion_pairs(video: str, start: int, end: int, dets: dict | None, scale: float = 0.5,
                 n_feat: int = 1500, min_inliers: int = 25) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """t -> (points in frame t, matching points in frame t+1), full-resolution pixels, RANSAC inliers.
    Players are masked out (they move independently of the camera)."""
    orb = cv2.ORB_create(n_feat, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    out, prev = {}, None
    for i, frame in frames_iter(video, start, end):
        g = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        mask = np.full(g.shape, 255, np.uint8)
        for x1, y1, x2, y2 in (dets.get(i, []) if dets else []):
            pad = 0.15 * (y2 - y1)
            cv2.rectangle(mask, (int((x1 - pad) * scale), int((y1 - pad) * scale)), (int((x2 + pad) * scale), int((y2 + pad) * scale)), 0, -1)
        kp, des = orb.detectAndCompute(g, mask)
        if prev is not None and des is not None and prev[1] is not None and len(kp) >= min_inliers and len(prev[0]) >= min_inliers:
            m = [a for a, b in (x for x in bf.knnMatch(prev[1], des, k=2) if len(x) == 2) if a.distance < 0.8 * b.distance]
            if len(m) >= min_inliers:
                pa = np.float32([prev[0][x.queryIdx].pt for x in m]); pb = np.float32([kp[x.trainIdx].pt for x in m])
                # TV graphics (score bug, channel logo, clock) are glued to the screen and match with
                # zero displacement; when a coherent moving set exists they are the outliers, not the
                # camera. Measured on clip1: 53 % of matches static -> pan under-estimated by ~50 %.
                moving = np.linalg.norm(pb - pa, axis=1) > 0.25
                if moving.sum() >= 2 * min_inliers:
                    pa, pb = pa[moving], pb[moving]
                _, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 1.5)
                if inl is not None and inl.sum() >= min_inliers:
                    k = inl.ravel().astype(bool)
                    out[i - 1] = (pa[k] / scale, pb[k] / scale)
        prev = (kp, des)
    return out


def ptz_step(ptz: np.ndarray, pa: np.ndarray, pb: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """(pan, tilt, f) of the next image given this image's (pan, tilt, f) and point matches."""
    from scipy.optimize import least_squares
    rays = _rot(ptz[0], ptz[1]).T @ np.linalg.inv(_K(ptz[2], img_w, img_h)) @ np.c_[pa, np.ones(len(pa))].T

    def res(x):
        q = _K(x[2], img_w, img_h) @ _rot(x[0], x[1]) @ rays
        return ((q[:2] / q[2]).T - pb).ravel()
    r = least_squares(res, ptz, x_scale=[0.1, 0.1, 50.0], loss="soft_l1", f_scale=2.0, max_nfev=50)
    return r.x


def chain_ptz(x0: np.ndarray, a: int, b: int, pairs: dict, img_w: int, img_h: int) -> np.ndarray:
    """Camera (pan, tilt, f) at frame b, chained from frame a through the image motion (either direction)."""
    x = np.array(x0, float)
    if b >= a:
        for t in range(a, b):
            if t in pairs:
                x = ptz_step(x, pairs[t][0], pairs[t][1], img_w, img_h)
    else:
        for t in range(a - 1, b - 1, -1):
            if t in pairs:
                x = ptz_step(x, pairs[t][1], pairs[t][0], img_w, img_h)
    return x


def motion_outliers(frames: list[int], ptz: list[np.ndarray], pairs: dict, img_w: int, img_h: int,
                    tol_deg: float = 1.0) -> dict[int, np.ndarray]:
    """Keyframes contradicted by the camera motion from BOTH neighbours (pan or tilt off by more
    than `tol_deg`). Returns frame -> motion-predicted (pan, tilt, f) (mean of the two chains).
    Measured on clip1: two keyframes with +2 / -3 deg tilt jumps that the image motion does not
    show (a tilt / focal ambiguity when little of the pitch markings is visible)."""
    out = {}
    for j in range(1, len(frames) - 1):
        fwd = chain_ptz(ptz[j - 1], frames[j - 1], frames[j], pairs, img_w, img_h)
        bwd = chain_ptz(ptz[j + 1], frames[j + 1], frames[j], pairs, img_w, img_h)
        bad = lambda pr: max(abs(pr[0] - ptz[j][0]), abs(pr[1] - ptz[j][1])) > tol_deg      # noqa: E731
        if bad(fwd) and bad(bwd) and max(abs(fwd[0] - bwd[0]), abs(fwd[1] - bwd[1])) <= tol_deg:
            pred = (fwd + bwd) / 2
            pred[2] = np.sqrt(fwd[2] * bwd[2])
            out[frames[j]] = pred
    return out


def motion_interpolate(kf: pd.DataFrame, pairs: dict, frames: np.ndarray, img_w: int, img_h: int) -> tuple[pd.DataFrame, dict]:
    """Per-frame camera params: keyframe values at valid keyframes, motion-chained in between
    (forward from a, backward from b, blended linearly in time; log-space for the focal)."""
    kf = kf.sort_values("frame")
    kfr = kf.frame.to_numpy()
    ptz_k = kf[["pan", "tilt", "f"]].to_numpy(dtype=float)
    pos_k = kf[["cx", "cy", "cz"]].to_numpy(dtype=float)
    f0, f1 = int(frames[0]), int(frames[-1])
    ptz = {int(a): ptz_k[j] for j, a in enumerate(kfr)}
    closure = []

    def chain(a, b, x0, fwd=True):
        out, x = {a: x0}, x0
        rng = range(a, b) if fwd else range(a, b, -1)
        for t in rng:
            nxt = t + 1 if fwd else t - 1
            pr = pairs.get(t if fwd else nxt)
            if pr is not None:
                x = ptz_step(x, pr[0], pr[1], img_w, img_h) if fwd else ptz_step(x, pr[1], pr[0], img_w, img_h)
            out[nxt] = x
        return out
    for j in range(len(kfr) - 1):
        a, b = int(kfr[j]), int(kfr[j + 1])
        fw, bw = chain(a, b, ptz_k[j], True), chain(b, a, ptz_k[j + 1], False)
        closure.append(abs(fw[b][0] - ptz_k[j + 1][0]) + abs(fw[b][1] - ptz_k[j + 1][1]))
        for t in range(a + 1, b):
            w = (t - a) / (b - a)
            x = (1 - w) * fw[t] + w * bw[t]
            x[2] = np.exp((1 - w) * np.log(fw[t][2]) + w * np.log(bw[t][2]))
            ptz[t] = x
    if len(kfr):                                   # before the first / after the last valid keyframe: one-sided chains
        ptz.update({k: v for k, v in chain(int(kfr[0]), f0, ptz_k[0], False).items() if k < kfr[0]})
        ptz.update({k: v for k, v in chain(int(kfr[-1]), f1, ptz_k[-1], True).items() if k > kfr[-1]})
    rows = []
    for t in frames:
        x = ptz.get(int(t))
        pos = pos_k[int(np.clip(np.searchsorted(kfr, t), 0, len(kfr) - 1))]
        rows.append([int(t), *pos, *(x if x is not None else (np.nan, np.nan, np.nan))])
    out = pd.DataFrame(rows, columns=["frame", "cx", "cy", "cz", "pan", "tilt", "f"])
    stats = {"motion_pairs": len(pairs), "motion_pair_frac": round(len(pairs) / max(len(frames) - 1, 1), 3),
             "motion_closure_deg_median": round(float(np.median(closure)), 3) if closure else None,
             "motion_closure_deg_p90": round(float(np.percentile(closure, 90)), 3) if closure else None}
    return out, stats


# ---------------------------------------------------------------- stage
class CalibStage(Stage):
    name = "s2_calib"
    config_key = "calib"

    def __init__(self, cfg: dict, shot_id: str, ingest_uri: str):
        super().__init__(cfg)
        self.shot_id, self.ingest_uri = shot_id, ingest_uri

    @staticmethod
    def _ok(r, p) -> bool:
        return r[2] >= float(p.get("min_coverage", 0.30)) and r[3] <= float(p.get("max_err_px", 3.5)) and r[4] >= float(p.get("min_explained", 0.30))

    def run(self, ctx: StageContext) -> dict:
        from .s0_ingest import load_shots
        p = self.params
        every = int(p.get("every_n_frames", 5))
        max_err = float(p.get("max_err_px", 3.5))      # mean distance of matched model points
        min_cov = float(p.get("min_coverage", 0.30))   # fraction of in-view model within 4 px of a line
        shot = next(s for s in load_shots(self.ingest_uri) if s.shot_id == self.shot_id)
        video = Storage.localize(Storage.join(self.ingest_uri, "video.mp4"), ctx.workdir)
        dets = None
        if Storage.exists(ctx.inp("detections.parquet")):
            dets = Storage.read_df(ctx.inp("detections.parquet"))
            dets = {int(k): g[["x1", "y1", "x2", "y2"]].to_numpy() for k, g in dets.groupby("frame")}
        # 1) line masks at keyframes
        from .. import calib_learned as CL
        mode = str(p.get("init", "auto"))                  # auto | learned | classical
        use_learned = mode == "learned" or (mode == "auto" and CL.available())
        if mode == "learned" and not CL.available():
            raise RuntimeError("calib.init=learned but PNLCALIB_DIR is not set up (scripts/fetch_pnlcalib.sh)")
        device = str(self.cfg.get("runtime", {}).get("device", "cpu"))
        keyframes, masks, learned = [], [], []
        for i, frame in frames_iter(video, shot.start_frame, shot.end_frame):
            if (i - shot.start_frame) % every:
                continue
            h_img, w_img = frame.shape[:2]
            keyframes.append(i); masks.append(line_mask(frame, dets.get(i) if dets else None))
            if use_learned:
                Pl = CL.pnl_projection(frame, device=device)
                learned.append(CL.params_from_projection(Pl, w_img, h_img, camera_P, MODEL_DENSE[::5]) + (Pl,) if Pl is not None else None)
        if not keyframes:
            Storage.write_df(ctx.out("calib.parquet"), pd.DataFrame(columns=["frame", "valid"]))
            return {"keyframes": 0}
        # 2) anchor = the keyframe with the most line evidence among a few spread candidates, fitted
        #    from scratch with multi-start; then propagate forward and backward by refinement.
        if use_learned:
            sol, valid_k, lstats = self._learned_solutions(keyframes, masks, learned, w_img, h_img, p)
            return self._finish(ctx, shot, video, dets, keyframes, masks, sol, valid_k, w_img, h_img, p, lstats)
        n_anchor = int(p.get("anchor_candidates", 3))
        idx = np.linspace(0, len(keyframes) - 1, min(n_anchor, len(keyframes))).round().astype(int)
        min_expl = float(p.get("min_explained", 0.30))
        ok_fit = lambda r: r[2] >= min_cov and r[3] <= max_err and r[4] >= min_expl   # noqa: E731
        anchors = []                                                    # (score, k, params, cost, cov, err)
        for k in idx:
            r = fit_frame(masks[k], w_img, h_img, topk=int(p.get("topk", 60)))
            anchors.append((r[1], k, r))
            log.info("calib %s anchor frame %d cost %.1f coverage %.2f explained %.2f err %.1fpx", self.shot_id, keyframes[k], r[1], r[2], r[4], r[3])
        anchors.sort(key=lambda t: t[0])
        _, k0, r0 = anchors[0]
        p0 = r0[0]
        log.info("calib %s: propagating from frame %d", self.shot_id, keyframes[k0])
        sol = {k0: r0}
        refit_gap = int(p.get("refit_every_keyframes", 10))
        for order in (range(k0 + 1, len(keyframes)), range(k0 - 1, -1, -1)):
            prev, last_refit = p0, -10 ** 6
            for k in order:
                r = refine(prev, masks[k], w_img, h_img, iters=1, maxfev=800)
                if not ok_fit(r):                           # lost: other anchors, then (rarely) a full search
                    alts = [r] + [refine(a_[2][0], masks[k], w_img, h_img, iters=1, maxfev=800) for a_ in anchors[1:] if ok_fit(a_[2])]
                    if abs(k - last_refit) >= refit_gap:
                        alts.append(fit_frame(masks[k], w_img, h_img, topk=30, finalists=3)); last_refit = k
                    r = min(alts, key=lambda t: t[1])
                sol[k] = r
                if ok_fit(r):
                    prev = r[0]
        # ---- pass 2: a main broadcast camera does not translate within a shot. Fix its position
        # to the median of the valid keyframes and refit pan / tilt / zoom everywhere: 3 DOF
        # instead of 6 is far better conditioned, rescues keyframes lost in pass 1 and removes
        # position jitter from every downstream metric.
        good_k = [k for k in sol if ok_fit(sol[k])]
        n_pass1 = len(good_k)
        if bool(p.get("fixed_position", True)) and len(good_k) >= 3:
            pos = np.median(np.stack([sol[k][0][:3] for k in good_k]), 0)
            for k in range(len(keyframes)):
                near = min(good_k, key=lambda g: abs(g - k))
                r = refine_ptz(pos, sol[near][0][3:] if not ok_fit(sol[k]) else sol[k][0][3:], masks[k], w_img, h_img)
                if not ok_fit(r):
                    r2 = search_ptz(pos, masks[k], w_img, h_img)
                    if r2 is not None and r2[1] < r[1]:
                        r = r2
                if ok_fit(r) or not ok_fit(sol[k]):
                    sol[k] = r
            log.info("calib %s pass 2 (fixed camera at %s): valid %d -> %d / %d", self.shot_id,
                     np.round(pos, 1).tolist(), n_pass1, sum(ok_fit(sol[k]) for k in sol), len(keyframes))
        valid_k = {k: bool(ok_fit(sol[k])) for k in range(len(keyframes))}
        return self._finish(ctx, shot, video, dets, keyframes, masks, sol, valid_k, w_img, h_img, p, {"init": "classical"})

    def _learned_solutions(self, keyframes, masks, learned, w_img, h_img, p):
        """Learned camera at every keyframe it succeeds on -> camera position fixed to their median ->
        pan / tilt / zoom refit to each learned projection -> optional line polish, kept only if it
        stays within `learned_agree_px` of the learned solution (white text on green boards otherwise
        pulls the chamfer fit). Keyframes the model misses are filled by the classical refinement
        from the nearest learned keyframe, with the classical validity test."""
        from .. import calib_learned as CL
        max_resid = float(p.get("learned_max_resid_px", 6.0))
        agree = float(p.get("learned_agree_px", 6.0))
        max_err, min_cov, min_expl = float(p.get("max_err_px", 3.5)), float(p.get("min_coverage", 0.30)), float(p.get("min_explained", 0.30))
        ok_fit = lambda r: r[2] >= min_cov and r[3] <= max_err and r[4] >= min_expl   # noqa: E731
        good = [k for k, l in enumerate(learned) if l is not None and l[1] <= max_resid]
        stats = {"init": "learned", "learned_ok": len(good), "learned_failed": len(keyframes) - len(good)}
        sol, valid_k = {}, {}

        def score(params, k):
            dt = cv2.distanceTransform(255 - masks[k], cv2.DIST_L2, 3)
            c = chamfer_cost(camera_H(params, w_img, h_img), dt, w_img, h_img, mask_points(masks[k], 1.0, 2500), detail=True)
            return (params, c[0], c[1], c[2], c[3])
        if len(good) >= 2 and bool(p.get("fixed_position", True)):
            pos = np.median(np.stack([learned[k][0][:3] for k in good]), 0)
            stats["camera_position"] = np.round(pos, 1).tolist()
            for k in good:
                r = CL.ptz_from_projection(learned[k][2], pos, learned[k][0][3:], w_img, h_img, camera_P, MODEL_DENSE[::5])
                if r is not None and r[1] <= max_resid:
                    learned[k] = (r[0], r[1], learned[k][2])
        n_polished = 0
        for k in good:
            base = score(learned[k][0], k)
            pol = refine_ptz(learned[k][0][:3], learned[k][0][3:], masks[k], w_img, h_img) if bool(p.get("learned_polish", True)) else None
            if pol is not None and pol[1] < base[1] and CL.model_displacement(camera_H(pol[0], w_img, h_img), camera_H(base[0], w_img, h_img), w_img, h_img, MODEL_DENSE[::5]) <= agree:
                sol[k] = pol; n_polished += 1
            else:
                sol[k] = base
            valid_k[k] = True
        for k in range(len(keyframes)):
            if k in sol:
                continue
            if not good:
                sol[k] = (np.array([0, -60, 20, 0, 15, 3000.0]), np.inf, 0.0, np.inf, 0.0); valid_k[k] = False; continue
            near = min(good, key=lambda g: abs(g - k))
            r = refine_ptz(sol[near][0][:3], sol[near][0][3:], masks[k], w_img, h_img)
            sol[k] = r; valid_k[k] = bool(ok_fit(r))
        stats["learned_polished"] = n_polished
        log.info("calib %s learned init: %s", self.shot_id, stats)
        return sol, valid_k, stats

    def _finish(self, ctx, shot, video, dets, keyframes, masks, sol, valid_k, w_img, h_img, p, init_stats):
        every = int(p.get("every_n_frames", 5))
        # ---- pass 3: camera-motion consistency. Frame-to-frame image motion (background features)
        # is an independent measurement of pan / tilt / zoom; a keyframe it contradicts from both
        # sides is refit from the motion prediction, and dropped if the refit still disagrees.
        pairs, motion_rejected, motion_refit = None, set(), 0
        use_motion = bool(p.get("motion_interp", True))
        if use_motion:
            pairs = motion_pairs(video, shot.start_frame, shot.end_frame, dets)
            good_k = [k for k in range(len(keyframes)) if valid_k[k]]
            if len(good_k) >= 3:
                tol = float(p.get("motion_tol_deg", 1.0))
                outl = motion_outliers([keyframes[k] for k in good_k], [sol[k][0][3:] for k in good_k], pairs, w_img, h_img, tol)
                for fr, pred in outl.items():
                    k = keyframes.index(fr)
                    r = refine_ptz(sol[k][0][:3], pred, masks[k], w_img, h_img)
                    if (init_stats.get("init") == "classical" and self._ok(r, p)) and max(abs(r[0][3] - pred[0]), abs(r[0][4] - pred[1])) <= tol:
                        sol[k] = r; motion_refit += 1
                    else:
                        motion_rejected.add(fr)
                log.info("calib %s pass 3 (motion consistency): %d keyframes contradicted, %d refit, %d dropped",
                         self.shot_id, len(outl), motion_refit, len(motion_rejected))
        rows, n_valid = [], 0
        for k, i in enumerate(keyframes):
            params, cost, cov, err, expl = sol[k]
            valid = bool(valid_k[k]) and i not in motion_rejected
            n_valid += int(valid)
            H = camera_H(params, w_img, h_img)
            rows.append({"frame": i, "valid": valid, "err_px": round(err, 2), "coverage": round(cov, 3), "explained": round(expl, 3), "cost": round(cost, 2),
                         **{f"h{k_}": float(v) for k_, v in enumerate(H.ravel())},
                         **{k_: float(v) for k_, v in zip(("cx", "cy", "cz", "pan", "tilt", "f"), params)}})
        df = pd.DataFrame(rows)
        # interpolate camera params to every frame (linear in parameter space, valid keyframes only)
        allf = np.arange(shot.start_frame, shot.end_frame + 1)
        good = df[df.valid]
        out = pd.DataFrame({"frame": allf})
        motion_stats = {}
        if len(good) >= 2 and pairs is not None:
            mo, motion_stats = motion_interpolate(good, pairs, allf, w_img, h_img)
            motion_stats.update(motion_keyframes_refit=motion_refit, motion_keyframes_dropped=len(motion_rejected))
            for k in ("cx", "cy", "cz", "pan", "tilt", "f"):
                out[k] = mo[k].to_numpy()
            log.info("calib %s motion interpolation: %s", self.shot_id, motion_stats)
        if len(good) >= 1:
            if not motion_stats:
                for k in ("cx", "cy", "cz", "pan", "tilt", "f"):
                    out[k] = np.interp(allf, good.frame, good[k])
            out["valid"] = np.abs(allf[:, None] - good.frame.to_numpy()[None, :]).min(1) <= every
            Hs = np.stack([camera_H(r[["cx", "cy", "cz", "pan", "tilt", "f"]].to_numpy(dtype=float), w_img, h_img).ravel() for _, r in out.iterrows()])
            for k in range(9):
                out[f"h{k}"] = Hs[:, k]
            out["err_px"] = np.interp(allf, df.frame, df.err_px)
            out["coverage"] = np.interp(allf, df.frame, df.coverage)
        else:
            out["valid"] = False
            for k in range(9):
                out[f"h{k}"] = np.nan
            out["err_px"] = np.nan; out["coverage"] = 0.0
        Storage.write_df(ctx.out("calib.parquet"), out)
        Storage.write_df(ctx.out("calib_keyframes.parquet"), df)
        return {"keyframes": len(df), "valid_frac": round(n_valid / max(len(df), 1), 3),
                "median_err_px": float(df.err_px.median()) if len(df) else None,
                "median_coverage": float(df.coverage.median()) if len(df) else None, **motion_stats,
                **{k_: v for k_, v in init_stats.items() if k_ != "camera_position"}}


def load_calib(uri: str) -> dict[int, np.ndarray] | None:
    """frame -> H (image <- pitch) for valid frames, or None if no calibration exists."""
    if not Storage.exists(Storage.join(uri, "calib.parquet")):
        return None
    df = Storage.read_df(Storage.join(uri, "calib.parquet"))
    df = df[df.valid]
    return {int(r.frame): np.array([r[f"h{k}"] for k in range(9)]).reshape(3, 3) for _, r in df.iterrows()}
