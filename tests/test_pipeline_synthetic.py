"""End-to-end test on a synthetic clip: 6 coloured 'players' + a ball, one leaves
and re-enters the frame. Checks the plumbing and the re-identification path
without any detector or GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from soccer_cv.core import Storage
from soccer_cv.pipeline import run_match
from soccer_cv.schema import DETECTION_COLUMNS

W, H, N, FPS = 640, 360, 150, 25.0


def make_clip(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    # team A: red shirts, team B: white shirts. player 2 exits at frame 50, re-enters at 100
    players = [dict(x=80, y=120, vx=1.2, team=0), dict(x=200, y=200, vx=0.8, team=0), dict(x=320, y=150, vx=6.0, team=0),
               dict(x=120, y=260, vx=1.0, team=1), dict(x=420, y=120, vx=-1.1, team=1), dict(x=520, y=240, vx=-0.7, team=1)]
    rows = []
    for f in range(N):
        img = np.full((H, W, 3), (40, 140, 40), np.uint8)
        for k, p in enumerate(players):
            if k == 2:
                if 50 <= f < 100:
                    continue
                x = p["x"] + p["vx"] * f if f < 50 else 700 - 6.0 * (f - 100) - 60
            else:
                x = p["x"] + p["vx"] * f
            y = p["y"] + 3 * np.sin(f / 9 + k)
            bw, bh = 18, 44
            col = (40, 40, 220) if p["team"] == 0 else (245, 245, 245)
            x1, y1 = int(x - bw / 2), int(y - bh / 2)
            cv2.rectangle(img, (x1, y1), (x1 + bw, y1 + bh), col, -1)
            cv2.rectangle(img, (x1, y1 + int(0.6 * bh)), (x1 + bw, y1 + bh), (30, 30, 30), -1)   # shorts
            if 0 <= x1 < W:
                jitter = rng.normal(0, 0.8, 4)
                rows.append((f, x1 + jitter[0], y1 + jitter[1], x1 + bw + jitter[2], y1 + bh + jitter[3], 0.9, "player"))
        bx, by = 60 + 3.5 * f, 300 - 2.0 * f + 0.02 * f * f
        if 0 < bx < W and 0 < by < H:
            cv2.circle(img, (int(bx), int(by)), 4, (255, 255, 255), -1)
            if f % 7 != 3:                                        # simulate misses
                rows.append((f, bx - 4, by - 4, bx + 4, by + 4, 0.7, "ball"))
        vw.write(img)
    vw.release()
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("scv")
    clip = d / "clip.mp4"
    dets = make_clip(clip)
    Storage.write_df(str(d / "dets.parquet"), dets)
    cfg = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    out = run_match(str(clip), str(d / "run"), str(cfg), replay_uri=str(d / "dets.parquet"))
    return d / "run", out


def test_ingest_single_shot(run_dir):
    run, _ = run_dir
    shots = Storage.read_json(str(run / "ingest" / "shots.json"))
    assert len(shots) == 1 and shots[0]["end_frame"] == N - 1


def test_tracks_are_stable(run_dir):
    run, _ = run_dir
    tr = Storage.read_df(str(run / "tier_a" / "shot_0000" / "s4_track" / "tracks.parquet"))
    # 6 players, one of them fragmented by the exit -> at most 7-8 tracklets
    assert 6 <= tr.track_id.nunique() <= 8, tr.track_id.nunique()
    assert tr.groupby("frame").size().median() >= 5


def test_reentry_is_relinked(run_dir):
    run, _ = run_dir
    ident = Storage.read_df(str(run / "tier_b" / "identities.parquet"))
    live = ident[~ident.abstained]
    # after re-linking, the number of identities equals the number of players
    assert live.identity_id.nunique() == 6, live
    graph = Storage.read_json(str(run / "tier_b" / "identity_graph.json"))
    assert len(graph["edges"]) >= 1


def test_teams_are_two(run_dir):
    run, _ = run_dir
    ident = Storage.read_df(str(run / "tier_b" / "identities.parquet"))
    sizes = ident[~ident.abstained].groupby("team").identity_id.nunique()
    assert sorted(sizes.tolist()) == [3, 3], sizes


def test_ball_states_and_overlay(run_dir):
    run, out = run_dir
    ball = Storage.read_df(str(run / "tier_c" / "ball.parquet"))
    assert (ball.state == "visible").mean() > 0.5
    assert (ball.state == "occluded").sum() > 0            # the simulated misses were interpolated
    assert Path(out["overlay"]).exists() and Path(out["overlay"]).stat().st_size > 10_000
