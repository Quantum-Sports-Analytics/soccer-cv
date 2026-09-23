"""2D pitch view ("minimap"): every visible identity, the ball and the camera footprint, in metres.

Built from Tier C's fused.parquet + ball.parquet and the per-shot calibration; written as
<run>/tier_c/pitch2d.json and served to the platform, which draws it under the video in
sync with playback.

Coordinates: pitch centre = (0, 0), x along the pitch (-52.5 .. 52.5), y towards the far
touchline (-34 near side, +34 far side). The canvas puts the far side at the top, like the
TV image.

What each element is, and its honest limits:
  * players / referee: foot point (with the height-prior estimate for boxes cut by the image
    border, as in the off-pitch filter), smoothed per identity over 9 frames (0.36 s) — box
    bottoms jitter by a few pixels, i.e. up to ~0.5 m on the far side.
  * ball: the image point projected onto the ground plane. Exact when the ball is on the
    grass; when it is in the air the projection lands *behind* its true ground position
    (no height is estimated yet), so in-flight samples are flagged and drawn differently.
  * camera footprint: the image border projected on the ground, clipped to the pitch
    surroundings — it shows why a player "disappears" (he left the camera view, not the pitch).
Frames without a valid calibration have no positions (reported, not guessed).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..core import Storage
from ..stages.s2_calib import L, W, camera_H, feet_pitch_xy

BALL_STATE_CODE = {"visible": 0, "occluded": 1, "in_flight": 2, "held": 3, "out_of_frame": 4, "unknown": 4}
CLIP_BOX = (-L / 2 - 10, L / 2 + 10, -W / 2 - 8, W / 2 + 8)


def _calib_params(run_uri: str) -> dict[int, np.ndarray]:
    """frame -> camera params, merged over shots (frame indices are global)."""
    out: dict[int, np.ndarray] = {}
    bad: set[int] = set()
    for u in Storage.list(Storage.join(run_uri, "tier_a")):
        if u.endswith("s2_calib/calib.parquet"):
            df = Storage.read_df(u)
            df = df[df.valid]
            cols = ["cx", "cy", "cz", "pan", "tilt", "f"]
            out.update({int(f): v for f, v in zip(df.frame, df[cols].to_numpy(dtype=float))})
        elif u.endswith("s4_track/calib_distrusted.json"):
            bad.update(int(f) for f in Storage.read_json(u))
    return {f: v for f, v in out.items() if f not in bad}        # frames the tracker found implausible: no positions


def _clip_polygon(poly: np.ndarray, box=CLIP_BOX) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon to an axis-aligned box."""
    x0, x1, y0, y1 = box
    edges = [(0, x0, 1), (0, x1, -1), (1, y0, 1), (1, y1, -1)]      # (axis, value, keep-side sign)
    pts = [tuple(p) for p in poly]
    for ax, val, sgn in edges:
        if not pts:
            break
        new = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            cin, pin = sgn * (cur[ax] - val) >= 0, sgn * (prev[ax] - val) >= 0
            if cin != pin:
                t = (val - prev[ax]) / (cur[ax] - prev[ax])
                new.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            if cin:
                new.append(cur)
        pts = new
    return np.array(pts)


def camera_footprint(params: np.ndarray, img_w: int, img_h: int, n: int = 12) -> np.ndarray | None:
    """Image border projected onto the ground plane, clipped to the pitch surroundings."""
    Hi = np.linalg.inv(camera_H(params, img_w, img_h))
    t = np.linspace(0, 1, n, endpoint=False)
    border = np.vstack([np.c_[t * img_w, np.zeros(n)], np.c_[np.full(n, img_w), t * img_h],
                        np.c_[(1 - t) * img_w, np.full(n, img_h)], np.c_[np.zeros(n), (1 - t) * img_h]])
    g = np.c_[border, np.ones(len(border))] @ Hi.T
    ok = g[:, 2] > 1e-9                       # points above the horizon project behind the camera
    if ok.sum() < 3:
        return None
    xy = g[ok, :2] / g[ok, 2:3]
    xy = np.clip(xy, -1000, 1000)             # far rays: keep direction, bounded before clipping
    poly = _clip_polygon(xy)
    return poly if len(poly) >= 3 else None


def pitch_positions(fused: pd.DataFrame, calib: dict[int, np.ndarray], img_w: int, img_h: int) -> np.ndarray:
    """(n, 2) foot position in metres for each fused row; NaN where the frame is not calibrated."""
    xy = np.full((len(fused), 2), np.nan)
    if fused.empty:
        return xy
    for fr, idx in fused.groupby("frame").indices.items():
        p = calib.get(int(fr))
        if p is None:
            continue
        boxes = fused.iloc[idx][["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        xy[idx], _ = feet_pitch_xy(p, boxes, img_w, img_h)
    return xy


def build_pitch2d(run_uri: str, smooth: int = 9, view_step: int = 5) -> dict:
    tc = Storage.join(run_uri, "tier_c")
    fused = Storage.read_df(Storage.join(tc, "fused.parquet"))
    ball = Storage.read_df(Storage.join(tc, "ball.parquet")) if Storage.exists(Storage.join(tc, "ball.parquet")) else pd.DataFrame()
    meta = Storage.read_json(Storage.join(run_uri, "ingest", "meta.json"))
    fps = float(meta.get("fps", 25.0)) or 25.0
    img_w, img_h = int(meta.get("width", 1920)), int(meta.get("height", 1080))
    n_frames = int(meta.get("n_frames") or (int(fused.frame.max()) + 1 if len(fused) else 0))
    calib = _calib_params(run_uri)

    # ---- players: use Tier C positions when present, else compute from boxes + calibration
    if {"px", "py"} <= set(fused.columns) and fused[["px", "py"]].notna().all(axis=1).any():
        xy = fused[["px", "py"]].to_numpy(dtype=float)
    else:
        xy = pitch_positions(fused, calib, img_w, img_h)
    f = fused.assign(x=xy[:, 0], y=xy[:, 1]).dropna(subset=["x", "y"])
    f = f[(f.x.abs() <= L / 2 + 6) & (f.y.abs() <= W / 2 + 6)]          # guard against degenerate projections
    known = f.identity_id >= 0
    if smooth > 1 and known.any():
        k = f[known].sort_values(["identity_id", "frame"]).copy()
        seg = (k.frame.diff().ne(1) | k.identity_id.diff().ne(0)).cumsum()   # contiguous runs per identity
        for c in ("x", "y"):
            k[c] = k.groupby(seg)[c].transform(lambda s: s.rolling(smooth, center=True, min_periods=1).mean())
        f = pd.concat([k, f[~known]])
    players: list[list] = [[] for _ in range(n_frames)]
    for r in f.itertuples(index=False):
        if 0 <= r.frame < n_frames:
            players[int(r.frame)].append([int(r.identity_id), int(r.team), round(float(r.x), 1), round(float(r.y), 1),
                                          round(float(r.conf), 2)])

    # ---- ball: ground-plane projection of the image point
    balls: list = [None] * n_frames
    n_ball = 0
    for r in ball.itertuples(index=False) if len(ball) else []:
        fr = int(r.frame)
        p = calib.get(fr)
        if p is None or not (0 <= fr < n_frames) or not np.isfinite(r.x) or not np.isfinite(r.y):
            continue
        g = np.linalg.inv(camera_H(p, img_w, img_h)) @ np.array([r.x, r.y, 1.0])
        if g[2] <= 1e-9:
            continue
        bx, by = g[0] / g[2], g[1] / g[2]
        if abs(bx) > L / 2 + 15 or abs(by) > W / 2 + 15:
            continue
        balls[fr] = [round(float(bx), 1), round(float(by), 1), BALL_STATE_CODE.get(str(r.state), 4)]
        n_ball += 1

    # ---- camera footprint, every `view_step` frames
    views: list = []
    for fr in range(0, n_frames, view_step):
        p = calib.get(fr)
        poly = camera_footprint(p, img_w, img_h) if p is not None else None
        views.append(None if poly is None else np.round(poly, 1).tolist())

    return {"fps": fps, "n_frames": n_frames, "pitch": [L, W], "view_step": view_step,
            "calibrated_frames": sum(1 for fr in range(n_frames) if fr in calib),
            "ball_frames": n_ball, "players": players, "ball": balls, "view": views}


def write_pitch2d(run_uri: str) -> dict:
    d = build_pitch2d(run_uri)
    Storage.write_bytes(Storage.join(run_uri, "tier_c", "pitch2d.json"),
                        json.dumps(d, separators=(",", ":")).encode())
    return {"pitch2d_frames": d["n_frames"], "pitch2d_calibrated_frames": d["calibrated_frames"],
            "pitch2d_ball_frames": d["ball_frames"]}
