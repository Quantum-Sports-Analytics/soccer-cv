"""Stage 4 — occlusion-aware tracking-by-detection on one shot.

Design (see architecture doc §2.4):
  * Kalman constant-velocity motion model on (cx, cy, w, h).
  * Expanded IoU (Deep-EIoU style): boxes are grown by `eiou_expansion` before
    IoU, which keeps fast, jumpy players associable.
  * Occlusion-corrected positional cost (OA-SORT idea): for each detection we
    estimate how much it is overlapped by *other* detections; a heavily
    overlapped box has an unreliable position, so its positional cost is
    softened (moved toward neutral) and its appearance is trusted more.
  * Two-stage association: high-score detections first, then low-score ones
    (ByteTrack), each solved with the Hungarian algorithm.
  * Assignment margin: for every matched track we record the gap between the
    chosen cost and the best alternative in its row/column. Low margin is the
    dispatch signal for the mask tier and the raw material for per-link error
    probabilities.
  * Appearance: a cheap colour-histogram embedding per detection (torso band),
    EMA-updated per track, weighted by (1 - occlusion). Replaceable by KPR.

Output: <output_uri>/tracks.parquet (TRACK_COLUMNS)
        <output_uri>/track_app.parquet (track_id, embedding, weight, color_hist)
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from ..core import Stage, StageContext, Storage, frames_iter
from ..schema import TRACK_COLUMNS, ObjClass

log = logging.getLogger(__name__)


# ------------------------------------------------------------ geometry
def iou_matrix(a: np.ndarray, b: np.ndarray, expand: float = 0.0) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    def grow(x):
        w = (x[:, 2] - x[:, 0]) * expand / 2; h = (x[:, 3] - x[:, 1]) * expand / 2
        return np.column_stack([x[:, 0] - w, x[:, 1] - h, x[:, 2] + w, x[:, 3] + h])
    a, b = grow(a), grow(b)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0]); yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2]); yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def occlusion_estimate(boxes: np.ndarray) -> np.ndarray:
    """Fraction of each box covered by the union of the *other* boxes (approx: max pairwise)."""
    n = len(boxes)
    if n < 2:
        return np.zeros(n)
    xx1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0]); yy1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    xx2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2]); yy2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    np.fill_diagonal(inter, 0)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + 1e-9
    return np.clip(inter.max(axis=1) / area, 0, 1)


def torso_hist(frame: np.ndarray, box, band=(0.15, 0.55), bins=(8, 4, 4)) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h = y2 - y1
    y1b, y2b = y1 + int(band[0] * h), y1 + int(band[1] * h)
    x1, x2 = max(x1, 0), min(x2, frame.shape[1])
    y1b, y2b = max(y1b, 0), min(y2b, frame.shape[0])
    if x2 - x1 < 2 or y2b - y1b < 2:
        return np.zeros(int(np.prod(bins)), dtype=np.float32)
    crop = cv2.cvtColor(frame[y1b:y2b, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([crop], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256]).flatten()
    return (hist / max(hist.sum(), 1e-9)).astype(np.float32)


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


# ------------------------------------------------------------ track state
class Track:
    _next_id = 1

    def __init__(self, box, score, emb, occl):
        self.id = Track._next_id; Track._next_id += 1
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        dt = 1.0
        self.kf.F = np.eye(8); self.kf.F[:4, 4:] = np.eye(4) * dt
        self.kf.H = np.zeros((4, 8)); self.kf.H[:4, :4] = np.eye(4)
        self.kf.R = np.diag([4, 4, 16, 16]).astype(float)
        self.kf.Q = np.diag([1, 1, 4, 4, 2, 2, 1, 1]).astype(float)
        self.kf.P = np.diag([10, 10, 20, 20, 50, 50, 20, 20]).astype(float)
        self.kf.x[:4, 0] = self._to_z(box)
        self.score = score
        self.emb = emb.copy(); self.emb_w = 1.0 - occl
        self.hits = 1; self.age = 0; self.time_since_update = 0
        self.last_margin = 1.0; self.last_occl = occl
        self.history: list[tuple[int, np.ndarray, float, float, float]] = []

    @staticmethod
    def _to_z(box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1; self.time_since_update += 1
        return self.box

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.kf.x[:4, 0]
        w, h = max(w, 2), max(h, 2)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def update(self, box, score, emb, occl, margin):
        self.kf.update(self._to_z(box))
        self.score = score
        self.hits += 1; self.time_since_update = 0
        self.last_margin = margin; self.last_occl = occl
        w = max(1.0 - occl, 0.05)          # visibility-weighted appearance update
        alpha = 0.1 * w
        self.emb = (1 - alpha) * self.emb + alpha * emb
        self.emb_w += w


# ------------------------------------------------------------ association
def associate(tracks: list[Track], dets: np.ndarray, embs: np.ndarray, occl: np.ndarray,
              p: dict) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if len(tracks) == 0 or len(dets) == 0:
        return [], list(range(len(tracks))), list(range(len(dets)))
    tboxes = np.stack([t.box for t in tracks])
    iou = iou_matrix(tboxes, dets[:, :4], expand=float(p.get("eiou_expansion", 0.3)))
    pos_cost = 1.0 - iou
    if p.get("occlusion_correct", True):
        # OA-SORT idea: an occluded detection's position is unreliable -> soften its
        # positional cost toward neutral (0.5) so appearance/motion decide instead.
        soft = occl[None, :] * 0.6
        pos_cost = pos_cost * (1 - soft) + 0.5 * soft
    app_cost = np.array([[hist_distance(t.emb, e) for e in embs] for t in tracks])
    aw = float(p.get("appearance_weight", 0.35))
    aw_eff = aw * (1 + occl[None, :])          # trust appearance more when occluded
    cost = (1 - aw_eff) * pos_cost + aw_eff * app_cost
    cost[iou < 1e-3] = 1e3                      # gate: no spatial overlap at all
    r, c = linear_sum_assignment(cost)
    matches, ut, ud = [], set(range(len(tracks))), set(range(len(dets)))
    thr = 1.0 - float(p.get("iou_threshold", 0.25))
    for i, j in zip(r, c):
        if cost[i, j] > thr:
            continue
        # assignment margin: best alternative in row or column
        row = np.delete(cost[i], j); col = np.delete(cost[:, j], i)
        alt = min(row.min() if row.size else 1e3, col.min() if col.size else 1e3)
        margin = float(np.clip(alt - cost[i, j], 0, 1))
        matches.append((i, j, margin)); ut.discard(i); ud.discard(j)
    return matches, sorted(ut), sorted(ud)


# ------------------------------------------------------------ the stage
class TrackStage(Stage):
    """Input: detect output dir (detections.parquet) + ingest dir for video. One shot."""
    name = "s4_track"
    config_key = "track"

    def __init__(self, cfg: dict, shot_id: str, ingest_uri: str):
        super().__init__(cfg)
        self.shot_id = shot_id
        self.ingest_uri = ingest_uri

    def run(self, ctx: StageContext) -> dict:
        from .s0_ingest import load_shots
        p = self.params
        shot = next(s for s in load_shots(self.ingest_uri) if s.shot_id == self.shot_id)
        video = Storage.localize(Storage.join(self.ingest_uri, "video.mp4"), ctx.workdir)
        dets = Storage.read_df(ctx.inp("detections.parquet"))
        persons = dets[dets.cls != ObjClass.BALL.value]
        by_frame = {int(k): g for k, g in persons.groupby("frame")}
        band = tuple(self.cfg.get("reid", {}).get("torso_crop", [0.15, 0.55]))

        Track._next_id = 1
        tracks: list[Track] = []
        finished: list[Track] = []
        rows = []
        hi_thr = float(self.cfg.get("detect", {}).get("score_threshold", 0.35)) + 0.15
        max_age, min_hits = int(p.get("max_age", 30)), int(p.get("min_hits", 3))
        n_switch_risk = 0

        for i, frame in frames_iter(video, shot.start_frame, shot.end_frame):
            g = by_frame.get(i)
            d = g[["x1", "y1", "x2", "y2", "score"]].to_numpy(dtype=float) if g is not None else np.zeros((0, 5))
            for t in tracks:
                t.predict()
            occl = occlusion_estimate(d[:, :4]) if len(d) else np.zeros(0)
            embs = np.stack([torso_hist(frame, b, band) for b in d[:, :4]]) if len(d) else np.zeros((0, 128))

            hi = np.where(d[:, 4] >= hi_thr)[0]; lo = np.where(d[:, 4] < hi_thr)[0]
            m1, ut, _ = associate(tracks, d[hi], embs[hi], occl[hi], p)
            matched_d = set()
            for ti, dj, mg in m1:
                j = hi[dj]; tracks[ti].update(d[j, :4], d[j, 4], embs[j], occl[j], mg); matched_d.add(j)
            rem_tracks = [tracks[k] for k in ut]
            m2, ut2, _ = associate(rem_tracks, d[lo], embs[lo], occl[lo], {**p, "appearance_weight": 0.15})
            for ti, dj, mg in m2:
                j = lo[dj]; rem_tracks[ti].update(d[j, :4], d[j, 4], embs[j], occl[j], mg); matched_d.add(j)
            # new tracks from unmatched HIGH-score detections only
            for j in hi:
                if j not in matched_d:
                    tracks.append(Track(d[j, :4], d[j, 4], embs[j], occl[j]))
            # emit + prune
            alive = []
            for t in tracks:
                if t.time_since_update == 0 and t.hits >= min_hits:
                    b = t.box
                    rows.append((i, t.id, *b, t.score, ObjClass.PLAYER.value, t.last_occl, t.last_margin))
                    t.history.append((i, b, t.score, t.last_occl, t.last_margin))
                    if t.last_margin < float(p.get("margin_low", 0.15)):
                        n_switch_risk += 1
                if t.time_since_update > max_age:
                    finished.append(t)
                else:
                    alive.append(t)
            tracks = alive
            if (i - shot.start_frame) % 200 == 0:
                log.info("track %s frame %d: %d live tracks", self.shot_id, i, len(tracks))
        finished.extend(tracks)

        df = pd.DataFrame(rows, columns=TRACK_COLUMNS)
        Storage.write_df(ctx.out("tracks.parquet"), df)
        app = pd.DataFrame([{"track_id": t.id, "embedding": t.emb.tolist(), "weight": float(t.emb_w),
                             "n_frames": len(t.history)} for t in finished if len(t.history)])
        Storage.write_df(ctx.out("track_app.parquet"), app)
        n_frames = shot.end_frame - shot.start_frame + 1
        return {"n_tracks": int(df.track_id.nunique()) if len(df) else 0,
                "tracks_per_frame": round(len(df) / max(n_frames, 1), 2),
                "low_margin_frac": round(n_switch_risk / max(len(df), 1), 3),
                "mean_track_len": round(float(df.groupby("track_id").size().mean()), 1) if len(df) else 0.0}
