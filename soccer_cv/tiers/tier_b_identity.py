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
from ..teams import cluster_teams

from ..core import Stage, StageContext, Storage
from ..schema import IDENTITY_COLUMNS, ROSTER_COLUMNS, RosterState

log = logging.getLogger(__name__)


def _hist_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def _link_belief(a: pd.Series, b: pd.Series, ea: np.ndarray, eb: np.ndarray, p: dict,
                 ra: np.ndarray | None = None, rb: np.ndarray | None = None) -> float:
    gap = b.start_frame - a.end_frame
    if gap < 0 or gap > int(p.get("link_max_gap_frames", 750)):
        return 0.0
    d_app = _hist_dist(ea, eb)
    thr = float(p.get("link_appearance_threshold", 0.45))
    if d_app > thr:
        return 0.0
    s_col = 1.0 - d_app / thr                          # 1 = identical colour signature
    if ra is not None and rb is not None:
        cos = float(ra @ rb)                           # ReID cosine; same person typically > 0.85
        lo, hi = float(p.get("reid_cos_low", 0.75)), float(p.get("reid_cos_high", 0.92))
        s_reid = float(np.clip((cos - lo) / (hi - lo), 0, 1))
        s_app = 0.3 * s_col + 0.7 * s_reid
    else:
        s_app = s_col
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
        reid = [np.asarray(json.loads(r), dtype=float) if isinstance(r, str) else None
                for r in (tl.reid_embedding if "reid_embedding" in tl else [None] * len(tl))]

        # ---- stage 9 (team part)
        k_teams = int(p.get("team_clusters", 2))
        teams, cinfo = cluster_teams(emb, w, k_teams)
        tl["team"] = teams

        # ---- staff: tracks that stay beyond the touch / goal lines and are not officials
        # (coaches, stewards, photographers on the grass). Assistant referees also live beyond
        # the touchline but are in the officials colour group, so they are kept.
        n_staff = 0
        if "beyond_frac" in tl:
            staff = ((tl.beyond_frac.to_numpy(dtype=float) >= float(p.get("staff_min_beyond_frac", 0.7)))
                     & (tl.n_frames.to_numpy() >= 25) & (teams != k_teams))
            n_staff = int(staff.sum())
            if n_staff:
                Storage.write_json(ctx.out("staff.json"), tl.loc[staff, ["shot_id", "track_id", "n_frames", "beyond_frac"]].to_dict(orient="records"))
                keep = ~staff
                tl = tl[keep].reset_index(drop=True); emb = emb[keep]; w = w[keep]
                reid = [r for r, kk in zip(reid, keep) if kk]; teams = teams[keep]

        # ---- stage 8: graph
        split_cands: dict[int, set[int]] = {}
        if "split_candidates" in tl:
            for i, sc in enumerate(tl.split_candidates):
                if isinstance(sc, str):
                    split_cands[i] = set(json.loads(sc))
        key_of = {(r.shot_id, int(r.track_id)): i for i, r in tl.iterrows()}
        G = nx.DiGraph()
        for i in range(len(tl)):
            G.add_node(i)
        n_split_abstain = 0
        for j in range(len(tl)):
            if j in split_cands:
                continue                                  # handled jointly below
            b = tl.iloc[j]
            for i in range(j):
                a = tl.iloc[i]
                if i in split_cands or a.end_frame >= b.start_frame or teams[i] != teams[j]:
                    continue
                if b.start_frame - a.end_frame > int(p.get("link_max_gap_frames", 750)):
                    continue
                bel = _link_belief(a, b, emb[i], emb[j], p, reid[i], reid[j])
                if bel > 0:
                    G.add_edge(i, j, belief=bel)

        # ---- stage 8b: segments born from undecidable crossings, resolved as JOINT assignments.
        # Group = same shot, same candidate set. Each candidate track is followed to the root of
        # its chain-so-far, and the identity's anchor embedding is the crop-weighted mean of every
        # segment already attributed to it — so evidence accumulates across successive crossings.
        # Hungarian assignment of the group's segments to the candidate identities; accepted only
        # if it beats the runner-up assignment by `split_link_margin`. Otherwise every segment of
        # the group stays a new, low-confidence identity (better unknown than wrong).
        from scipy.optimize import linear_sum_assignment
        split_margin = float(p.get("split_link_margin", 0.03))
        n_crops = tl.n_frames.to_numpy(dtype=float)
        split_links: list[tuple[int, int, float]] = []       # (pred, succ, belief)
        chain_pred: dict[int, int] = {}                       # provisional predecessor map for anchors
        groups: dict[tuple, list[int]] = {}
        split_frame = tl.split_frame.to_numpy() if "split_frame" in tl else np.full(len(tl), -1)
        for j, cands in split_cands.items():
            groups.setdefault((tl.shot_id.iloc[j], int(split_frame[j]), tuple(sorted(cands))), []).append(j)

        def root(i):
            while i in chain_pred:
                i = chain_pred[i]
            return i

        def members(r):
            out = [r]; changed = True
            while changed:
                changed = False
                for a_, b_ in chain_pred.items():
                    if b_ in out and a_ not in out:
                        out.append(a_); changed = True
            return out

        for (shot, sf, cands), segs in sorted(groups.items(), key=lambda kv: kv[0][1]):
            segs = sorted(segs, key=lambda j: tl.start_frame.iloc[j])
            seg_tids = {int(tl.track_id.iloc[j]) for j in segs}
            # candidates = pre-window tracks; a track born inside the window is a segment, not an anchor
            cand_idx = [key_of[(shot, t)] for t in cands if (shot, t) in key_of and t not in seg_tids]
            cand_idx = [i for i in cand_idx if tl.end_frame.iloc[i] <= sf and reid[i] is not None]
            # one anchor per distinct identity root among the candidates
            roots = {}
            for i in cand_idx:
                roots.setdefault(root(i), []).append(i)
            anchors = []
            for r, _ in roots.items():
                mem = [m for m in members(r) if reid[m] is not None]
                v = sum(reid[m] * n_crops[m] for m in mem); v = v / (np.linalg.norm(v) + 1e-9)
                anchors.append((r, v))
            segs_ok = [j for j in segs if reid[j] is not None]
            if len(anchors) == 0 or len(segs_ok) == 0:
                n_split_abstain += len(segs); continue
            S = np.array([[float(reid[j] @ v) for _, v in anchors] for j in segs_ok])
            r_idx, c_idx = linear_sum_assignment(-S)
            best = float(S[r_idx, c_idx].sum())
            second = -np.inf
            for r, c in zip(r_idx, c_idx):
                S2 = S.copy(); S2[r, c] = -1e3
                rr, cc = linear_sum_assignment(-S2); second = max(second, float(S2[rr, cc].sum()))
            margin = best - second if np.isfinite(second) else 1.0
            if margin < split_margin:
                n_split_abstain += len(segs); continue
            for r, c in zip(r_idx, c_idx):
                j = segs_ok[r]; root_i = anchors[c][0]
                # link to the LAST segment of that identity's chain that ends before j starts
                chain = [m for m in members(root_i) if tl.end_frame.iloc[m] <= sf]
                if not chain:
                    continue
                i = max(chain, key=lambda m: tl.end_frame.iloc[m])
                bel = float(np.clip(0.6 + margin * 4.0, 0.6, 0.98))
                split_links.append((i, j, bel)); chain_pred[j] = i
        for i, j, bel in split_links:
            G.add_edge(i, j, belief=bel, split=True)

        # ---- stage 10: greedy chaining by descending belief, one successor / one predecessor
        edges = sorted(G.edges(data=True), key=lambda e: -e[2]["belief"])
        succ, pred, used_edges = {}, {}, []
        abstain = float(p.get("abstain_below", 0.55))
        for i, j, bel in split_links:                    # committed first: they carry the crossing evidence
            if i not in succ and j not in pred:
                succ[i] = j; pred[j] = i; used_edges.append((i, j, bel))
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
        for j in split_cands:
            if j not in pred:                      # unresolved split: uncertain identity
                conf[j] = min(conf[j], 0.3)
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
        return {"n_tracklets": len(tl), "n_identities": int(next_id - 1), "split_abstain": n_split_abstain,
                "n_relinks": n_reentry, "n_abstained": int(tl.abstained.sum()), "n_staff_removed": n_staff,
                "cardinality_violations_frames": n_card_viol,
                "team_sizes": tl[~tl.abstained].groupby("team").identity_id.nunique().to_dict()}
