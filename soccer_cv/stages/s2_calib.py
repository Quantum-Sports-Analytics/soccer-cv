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


def image_to_pitch(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    Hi = np.linalg.inv(H)
    p = np.c_[uv, np.ones(len(uv))] @ Hi.T
    return p[:, :2] / p[:, 2:3]


# ---------------------------------------------------------------- stage
class CalibStage(Stage):
    name = "s2_calib"
    config_key = "calib"

    def __init__(self, cfg: dict, shot_id: str, ingest_uri: str):
        super().__init__(cfg)
        self.shot_id, self.ingest_uri = shot_id, ingest_uri

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
        keyframes, masks = [], []
        for i, frame in frames_iter(video, shot.start_frame, shot.end_frame):
            if (i - shot.start_frame) % every:
                continue
            h_img, w_img = frame.shape[:2]
            keyframes.append(i); masks.append(line_mask(frame, dets.get(i) if dets else None))
        if not keyframes:
            Storage.write_df(ctx.out("calib.parquet"), pd.DataFrame(columns=["frame", "valid"]))
            return {"keyframes": 0}
        # 2) anchor = the keyframe with the most line evidence among a few spread candidates, fitted
        #    from scratch with multi-start; then propagate forward and backward by refinement.
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
        rows, n_valid = [], 0
        for k, i in enumerate(keyframes):
            params, cost, cov, err, expl = sol[k]
            valid = bool(ok_fit(sol[k]))
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
        if len(good) >= 1:
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
                "median_coverage": float(df.coverage.median()) if len(df) else None}


def load_calib(uri: str) -> dict[int, np.ndarray] | None:
    """frame -> H (image <- pitch) for valid frames, or None if no calibration exists."""
    if not Storage.exists(Storage.join(uri, "calib.parquet")):
        return None
    df = Storage.read_df(Storage.join(uri, "calib.parquet"))
    df = df[df.valid]
    return {int(r.frame): np.array([r[f"h{k}"] for k in range(9)]).reshape(3, 3) for _, r in df.iterrows()}
