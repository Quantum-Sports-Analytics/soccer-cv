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


def chamfer_cost(H: np.ndarray, dt: np.ndarray, img_w: int, img_h: int, mask_pts: np.ndarray | None = None,
                 trunc: float = 30.0, scale: float = 1.0) -> tuple[float, float]:
    """Symmetric truncated chamfer: model->lines (dt of the line mask at projected model points)
    plus lines->model (dt of the rendered model at line-mask pixels). `dt`, `mask_pts` are at
    `scale` times the image resolution."""
    uv, ok = project(H, MODEL_PTS)
    inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    if inside.sum() < 80:
        return 1e3, 0.0
    uvs = uv[inside] * scale
    d = dt[uvs[:, 1].astype(int), uvs[:, 0].astype(int)]
    fwd = np.minimum(d, trunc * scale).mean() / scale
    cov = float((d < 4 * scale).mean())
    rev = 0.0
    if mask_pts is not None and len(mask_pts):
        rm = render_model(H, img_w, img_h, scale)
        dtr = cv2.distanceTransform(255 - rm, cv2.DIST_L2, 3)
        dr = dtr[mask_pts[:, 1], mask_pts[:, 0]]
        rev = np.minimum(dr, trunc * scale).mean() / scale
    return float(fwd + rev), cov


def mask_points(m: np.ndarray, scale: float, max_pts: int = 1500) -> np.ndarray:
    small = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale != 1.0 else m
    ys, xs = np.nonzero(small)
    if len(xs) > max_pts:
        idx = np.random.default_rng(0).choice(len(xs), max_pts, replace=False); xs, ys = xs[idx], ys[idx]
    return np.stack([xs, ys], 1)


def coarse_search(m: np.ndarray, img_w: int, img_h: int, topk: int = 6) -> list[np.ndarray]:
    """Grid over broadcast-camera priors at quarter resolution; returns the top-k distinct candidates."""
    sc = 0.25
    small = cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    dt = cv2.distanceTransform(255 - small, cv2.DIST_L2, 3)
    mp = mask_points(m, sc, 600)
    cands = []
    for side in (-1.0, 1.0):                                   # behind the near (-) or far (+) touchline
        for cx in (-20.0, 0.0, 20.0):
            for dist in (20.0, 40.0, 70.0):
                for cz in (15.0, 25.0, 45.0):
                    for pan in np.arange(-50, 51, 10.0):
                        for tilt in (8, 13, 18, 24, 30):
                            for f in (1500, 2200, 3200, 4600, 6500, 9000):
                                p = np.array([cx, side * (W / 2 + dist), cz, (pan if side < 0 else 180 + pan), tilt, f])
                                c, _ = chamfer_cost(camera_H(p, img_w, img_h), dt, img_w, img_h, mp, scale=sc)
                                cands.append((c, p))
    cands.sort(key=lambda t: t[0])
    out = []
    for c, p in cands:
        # distinct: differ in pan by >= 10 deg or side or position
        if all(abs(p[3] - q[3]) >= 10 or np.sign(p[1]) != np.sign(q[1]) or abs(p[0] - q[0]) >= 20 or abs(p[2] - q[2]) >= 10 for q in out):
            out.append(p)
        if len(out) >= topk:
            break
    return out


def fit_frame(m: np.ndarray, img_w: int, img_h: int, topk: int = 6) -> tuple[np.ndarray, float, float]:
    """Coarse grid -> refine the top-k candidates -> keep the best at full resolution."""
    best = (None, np.inf, 0.0)
    for p0 in coarse_search(m, img_w, img_h, topk):
        p, c, cov = refine(p0, m, img_w, img_h, iters=1)
        if c < best[1]:
            best = (p, c, cov)
    p, c, cov = refine(best[0], m, img_w, img_h, iters=1)
    return p, c, cov


def refine(p0: np.ndarray, m: np.ndarray, img_w: int, img_h: int, iters: int = 2) -> tuple[np.ndarray, float, float]:
    sc = 0.5
    small = cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    dt = cv2.distanceTransform(255 - small, cv2.DIST_L2, 3)
    mp = mask_points(m, sc, 1500)
    scale = np.array([8.0, 8.0, 5.0, 6.0, 4.0, 600.0])

    def f(z):
        return chamfer_cost(camera_H(p0 + z * scale, img_w, img_h), dt, img_w, img_h, mp, scale=sc)[0]
    z = np.zeros(6)
    for _ in range(iters):
        res = minimize(f, z, method="Powell", options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 4000})
        z = res.x
        res = minimize(f, z, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4, "maxfev": 2000, "initial_simplex": z + np.vstack([np.zeros(6), 0.3 * np.eye(6)])})
        z = res.x
    p = p0 + z * scale
    # final numbers at full resolution
    dt_full = cv2.distanceTransform(255 - m, cv2.DIST_L2, 3)
    c, cov = chamfer_cost(camera_H(p, img_w, img_h), dt_full, img_w, img_h, mask_points(m, 1.0, 3000))
    return p, c, cov


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
        max_err = float(p.get("max_err_px", 6.0))
        min_cov = float(p.get("min_coverage", 0.35))
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
        anchors = []
        for k in idx:
            params, err, cov = fit_frame(masks[k], w_img, h_img, topk=int(p.get("topk", 6)))
            anchors.append((err, k, params, cov))
            log.info("calib %s anchor frame %d err %.1f cov %.2f", self.shot_id, keyframes[k], err, cov)
        log.info("calib %s: propagating from frame %d", self.shot_id, keyframes[sorted(anchors)[0][1]])
        anchors.sort(key=lambda t: t[0])
        err0, k0, p0, cov0 = anchors[0]
        sol = {k0: (p0, err0, cov0)}
        refit_gap = int(p.get("refit_every_keyframes", 20))
        for order in (range(k0 + 1, len(keyframes)), range(k0 - 1, -1, -1)):
            prev, last_refit = p0, -10 ** 6
            for k in order:
                params, err, cov = refine(prev, masks[k], w_img, h_img, iters=1)
                if err > max_err * 1.5:                     # lost: cheap fallbacks first (other anchors,
                    alts = [(params, err, cov)]              # all known solutions), full search rarely
                    for a_ in anchors[1:]:
                        alts.append(refine(a_[2], masks[k], w_img, h_img, iters=1))
                    if abs(k - last_refit) >= refit_gap:
                        alts.append(fit_frame(masks[k], w_img, h_img, topk=4)); last_refit = k
                    params, err, cov = min(alts, key=lambda t: t[1])
                sol[k] = (params, err, cov)
                if err <= max_err:
                    prev = params
        rows, n_valid = [], 0
        for k, i in enumerate(keyframes):
            params, err, cov = sol[k]
            valid = bool(err <= max_err and cov >= min_cov)
            n_valid += int(valid)
            H = camera_H(params, w_img, h_img)
            rows.append({"frame": i, "valid": valid, "err_px": round(err, 2), "coverage": round(cov, 3),
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
