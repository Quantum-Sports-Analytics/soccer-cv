"""Tier B — match-global identity resolution over tracklet summaries (stages 8-10).

Input : all shots' tracklets.parquet (from stage 5), optional jersey votes.
Output: identities.parquet  (IDENTITY_COLUMNS)  one row per tracklet
        roster.parquet      (ROSTER_COLUMNS)    per-frame roster state
        identity_graph.json edges kept, with their link probability

Algorithm (first implementation; the HMM formulation replaces `_link_belief`
without changing the interface):

 1. Team / role clustering (stage 9 part): k-means on colour embeddings into
    `team_clusters` + 1 outlier group. Clusters are *relative* — no kit
    vocabulary. Referee is the small cluster far from both team centroids;
    goalkeepers are handled as their own cluster when present.
 2. Tracklet graph (stage 8): candidate edge (a -> b) when b starts after a
    ends, gap <= link_max_gap_frames, same team cluster, and appearance
    distance <= link_appearance_threshold. Edge weight = link belief from
    appearance, temporal gap and spatial plausibility (image space until
    calibration is available).
 3. Identity assignment (stage 10): greedy maximum-belief chaining under the
    constraint that identities do not overlap in time. Then the cardinality
    constraint: at most `max_per_team` identities per team may be *visible* on
    any frame; if violated, the weakest concurrent identity is marked
    abstained. Tracklets whose best link belief is below `abstain_below` start
    a new identity with `abstained=True` when they are short — i.e. we report
    `unknown` rather than guess.
"""
from __future__ import annotations

import json
import logging

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..core import Stage, StageContext, Storage
from ..schema import IDENTITY_COLUMNS, ROSTER_COLUMNS, RosterState

log = logging.getLogger(__name__)


def _hist_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def cluster_teams(emb: np.ndarray, weights: np.ndarray, k: int = 2, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Return cluster label per tracklet: 0..k-1 = teams, k = officials/other."""
    if len(emb) < k:
        return np.zeros(len(emb), dtype=int), {}
    w = np.maximum(weights, 1e-3)
    # k team clusters first; officials are then the *outliers* of their cluster
    # (distance to centroid far above the cluster's typical spread), not a third
    # k-means group — a third group over-splits a team when tracklets are few.
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb, sample_weight=w)
    labels = km.labels_.copy()
    dist = 0.5 * np.abs(emb - km.cluster_centers_[labels]).sum(1)
    out = labels.copy()
    for c in range(k):
        m = labels == c
        if m.sum() < 3:
            continue
        med = np.median(dist[m]); mad = np.median(np.abs(dist[m] - med)) + 1e-6
        outl = m & (dist > med + 4.0 * mad) & (dist > 0.25)
        out[outl] = k
    mass = np.array([w[out == c].sum() for c in range(k + 1)])
    order = np.argsort(-mass[:k])            # team 0 = heavier cluster (more tracklet-frames)
    remap = {int(order[i]): i for i in range(k)}
    out = np.array([remap.get(int(l), k) if l < k else k for l in out])
    info = {"cluster_mass": mass.round(1).tolist(), "n_officials": int((out == k).sum())}
    return out, info


def _link_belief(a: pd.Series, b: pd.Series, ea: np.ndarray, eb: np.ndarray, p: dict) -> float:
    gap = b.start_frame - a.end_frame
    if gap < 0 or gap > int(p.get("link_max_gap_frames", 750)):
        return 0.0
    d_app = _hist_dist(ea, eb)
    thr = float(p.get("link_appearance_threshold", 0.45))
    if d_app > thr:
        return 0.0
    s_app = 1.0 - d_app / thr                          # 1 = identical colour signature
    s_gap = float(np.exp(-gap / 250.0))                # 10 s half-life-ish at 25 fps
    # spatial plausibility in image space: displacement vs gap (cheap stand-in for pitch metres)
    la, fb = json.loads(a.last_box), json.loads(b.first_box)
    dist = np.hypot((la[0] + la[2]) / 2 - (fb[0] + fb[2]) / 2, la[3] - fb[3])
    s_sp = float(np.exp(-max(dist - 12.0 * gap, 0) / 300.0))   # 12 px/frame allowance
    # short tracklets carry less evidence
    s_len = min(1.0, (min(a.n_frames, b.n_frames) / 15.0))
    return float(np.clip(0.55 * s_app + 0.25 * s_gap + 0.20 * s_sp, 0, 1) * (0.5 + 0.5 * s_len))


class IdentityTier(Stage):
    """Input: directory containing one sub-dir per shot with tracklets.parquet."""
    name = "tier_b_identity"
    config_key = "identity"

    def run(self, ctx: StageContext) -> dict:
        p = self.params
        files = [u for u in Storage.list(ctx.input_uri) if u.endswith("tracklets.parquet")]
        tl = pd.concat([Storage.read_df(u) for u in files], ignore_index=True) if files else pd.DataFrame()
        if tl.empty:
            Storage.write_df(ctx.out("identities.parquet"), pd.DataFrame(columns=IDENTITY_COLUMNS))
            Storage.write_df(ctx.out("roster.parquet"), pd.DataFrame(columns=ROSTER_COLUMNS))
            return {"n_tracklets": 0}
        tl = tl.sort_values(["start_frame", "shot_id", "track_id"]).reset_index(drop=True)
        emb = np.stack([np.asarray(json.loads(e), dtype=float) for e in tl.app_embedding])
        w = tl.app_weight.to_numpy(dtype=float)

        # ---- stage 9 (team part)
        teams, cinfo = cluster_teams(emb, w, int(p.get("team_clusters", 2)))
        tl["team"] = teams

        # ---- stage 8: graph
        G = nx.DiGraph()
        for i in range(len(tl)):
            G.add_node(i)
        for i in range(len(tl)):
            a = tl.iloc[i]
            for j in range(i + 1, len(tl)):
                b = tl.iloc[j]
                if b.start_frame <= a.end_frame or teams[i] != teams[j]:
                    continue
                if b.start_frame - a.end_frame > int(p.get("link_max_gap_frames", 750)):
                    break
                bel = _link_belief(a, b, emb[i], emb[j], p)
                if bel > 0:
                    G.add_edge(i, j, belief=bel)

        # ---- stage 10: greedy chaining by descending belief, one successor / one predecessor
        edges = sorted(G.edges(data=True), key=lambda e: -e[2]["belief"])
        succ, pred, used_edges = {}, {}, []
        abstain = float(p.get("abstain_below", 0.55))
        for i, j, d in edges:
            if i in succ or j in pred or d["belief"] < abstain:
                continue
            succ[i] = j; pred[j] = i; used_edges.append((i, j, d["belief"]))
        identity = np.full(len(tl), -1); conf = np.ones(len(tl)); next_id = 1
        for i in range(len(tl)):
            if i in pred:
                continue
            k = i; c = 1.0
            while True:
                identity[k] = next_id; conf[k] = c
                if k in succ:
                    c = min(c, G.edges[k, succ[k]]["belief"]); k = succ[k]
                else:
                    break
            next_id += 1
        # abstain: short, unlinked tracklets are reported as unknown rather than a new player
        abst = np.array([(identity[i] not in identity[np.arange(len(tl)) != i]) and tl.n_frames.iloc[i] < 15
                         for i in range(len(tl))])
        tl["identity_id"] = identity; tl["confidence"] = conf; tl["abstained"] = abst

        # ---- cardinality: <= max_per_team visible per frame per team
        max_per = int(p.get("max_per_team", 11))
        n_card_viol = 0
        frames = np.arange(tl.start_frame.min(), tl.end_frame.max() + 1)
        for team in range(int(p.get("team_clusters", 2))):
            sub = tl[(tl.team == team) & (~tl.abstained)]
            active = np.zeros(len(frames), dtype=int)
            for _, r in sub.iterrows():
                active[r.start_frame - frames[0]: r.end_frame - frames[0] + 1] += 1
            if (active > max_per).any():
                n_card_viol += int((active > max_per).sum())
                # demote the lowest-confidence tracklets overlapping the violation
                bad = np.where(active > max_per)[0] + frames[0]
                over = sub[(sub.start_frame <= bad.max()) & (sub.end_frame >= bad.min())].sort_values("confidence")
                for idx in over.index[: max(1, len(over) - max_per)]:
                    tl.loc[idx, "abstained"] = True

        ident = tl[["shot_id", "track_id", "identity_id", "team", "confidence", "abstained"]].copy()
        ident["jersey"] = None
        Storage.write_df(ctx.out("identities.parquet"), ident[IDENTITY_COLUMNS])

        # ---- roster state per frame per identity
        rows = []
        for iid, g in tl[~tl.abstained].groupby("identity_id"):
            g = g.sort_values("start_frame")
            spans = list(zip(g.start_frame, g.end_frame, g.last_box))
            team = int(g.team.iloc[0])
            for f in frames[::5]:                           # 5 Hz roster is plenty for the panel
                st, last_xy, since = RosterState.OFF_FRAME.value, (None, None), None
                for s, e, lb in spans:
                    if s <= f <= e:
                        st, since = RosterState.VISIBLE.value, 0; break
                    if e < f:
                        b = json.loads(lb); last_xy = ((b[0] + b[2]) / 2, b[3]); since = f - e
                if st != RosterState.VISIBLE.value and since is None:
                    continue                                # not yet appeared
                if st != RosterState.VISIBLE.value and since is not None and since > 250:
                    st = RosterState.UNCERTAIN.value
                rows.append((int(f), int(iid), team, None, st, last_xy[0], last_xy[1], since))
        Storage.write_df(ctx.out("roster.parquet"), pd.DataFrame(rows, columns=ROSTER_COLUMNS))
        Storage.write_json(ctx.out("identity_graph.json"),
                           {"edges": [(int(i), int(j), round(b, 3)) for i, j, b in used_edges], "clusters": cinfo})
        n_reentry = len(used_edges)
        return {"n_tracklets": len(tl), "n_identities": int(next_id - 1),
                "n_relinks": n_reentry, "n_abstained": int(tl.abstained.sum()),
                "cardinality_violations_frames": n_card_viol,
                "team_sizes": tl[~tl.abstained].groupby("team").identity_id.nunique().to_dict()}
