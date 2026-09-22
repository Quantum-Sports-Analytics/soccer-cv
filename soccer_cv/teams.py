"""Team-colour model shared by the tracker (online) and Tier B (offline), plus a
pitch mask used to reject people whose feet are off the grass (bench, coaches,
photographers) until field calibration exists.
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

GRASS_LO, GRASS_HI = (35, 40, 40), (85, 255, 255)


def cluster_teams(emb: np.ndarray, weights: np.ndarray, k: int = 2, seed: int = 0):
    """Return (labels, info). labels: 0..k-1 = teams (0 = heavier), k = officials/other.

    k+1 groups are tried first; the lightest is officials only if it is clearly
    lighter than the teams, otherwise k groups and no officials.
    """
    if len(emb) < k:
        return np.zeros(len(emb), dtype=int), {}
    w = np.maximum(weights, 1e-3)
    out = None
    if len(emb) >= k + 2:
        km = KMeans(n_clusters=k + 1, n_init=10, random_state=seed).fit(emb, sample_weight=w)
        mass = np.array([w[km.labels_ == c].sum() for c in range(k + 1)])
        order = np.argsort(-mass)
        if mass[order[-1]] < 0.4 * mass[order[k - 1]]:
            remap = {int(order[i]): i for i in range(k)}
            out = np.array([remap.get(int(l), k) for l in km.labels_])
    if out is None:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb, sample_weight=w)
        mass = np.array([w[km.labels_ == c].sum() for c in range(k)])
        order = np.argsort(-mass)
        remap = {int(order[i]): i for i in range(k)}
        out = np.array([remap[int(l)] for l in km.labels_])
    info = {"cluster_mass": [round(float(w[out == c].sum()), 1) for c in range(k + 1)],
            "n_officials": int((out == k).sum())}
    return out, info


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


class OnlineTeamModel:
    """Two team centroids in colour-histogram space, fitted from live tracks.

    `assign(hist)` -> 0 / 1 when the histogram is clearly closer to one centroid
    (ratio test), else -1 (unknown: officials, occluded crops, not fitted yet).
    """

    def __init__(self, ratio: float = 0.7, min_tracks: int = 8):
        self.centroids: np.ndarray | None = None
        self.ratio, self.min_tracks = ratio, min_tracks

    @property
    def ready(self) -> bool:
        return self.centroids is not None

    def fit(self, embs: list[np.ndarray], weights: list[float]) -> bool:
        if len(embs) < self.min_tracks:
            return False
        E, w = np.stack(embs), np.asarray(weights, dtype=float)
        labels, _ = cluster_teams(E, w, 2)
        cents = []
        for c in (0, 1):
            m = labels == c
            if m.sum() == 0:
                return False
            cents.append((E[m] * w[m, None]).sum(0) / w[m].sum())
        self.centroids = np.stack(cents)
        return True

    def assign(self, hist: np.ndarray) -> int:
        if self.centroids is None:
            return -1
        d = 0.5 * np.abs(self.centroids - hist[None]).sum(1)
        i = int(d.argmin()); j = 1 - i
        return i if d[i] < self.ratio * d[j] else -1


def pitch_mask(frame_bgr: np.ndarray, scale: int = 8) -> np.ndarray:
    """Binary mask (full-res) of the playing surface: green pixels, morphologically
    closed so lines/players are filled, largest connected component kept."""
    H, W = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (W // scale, H // scale), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, GRASS_LO, GRASS_HI)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k, iterations=2)
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        g = np.where(lab == big, 255, 0).astype(np.uint8)
    # convex-ish fill: the pitch region should not have holes (players, logos)
    cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        g = np.zeros_like(g); cv2.fillConvexPoly(g, hull, 255)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)


def feet_on_pitch(mask: np.ndarray, boxes: np.ndarray, tol_px: int = 6) -> np.ndarray:
    """True where the box's foot point (cx, y2) lies on the pitch mask (with tolerance)."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    H, W = mask.shape
    cx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2).astype(int), 0, W - 1)
    fy = np.clip((boxes[:, 3] - tol_px).astype(int), 0, H - 1)
    return mask[fy, cx] > 0
