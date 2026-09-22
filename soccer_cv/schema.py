"""Data contracts between pipeline stages.

Everything that crosses a tier boundary is one of the dataclasses below,
serialised to Parquet (tables) or JSON (small records). Tier A -> Tier B
carries tracklets and cue summaries only, never pixels.

Coordinate conventions
----------------------
* image: pixels, origin top-left, (x, y); boxes are (x1, y1, x2, y2).
* pitch: metres, origin at pitch centre, x along the long axis towards the
  right goal as seen from the main camera, y towards the far touchline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json
import pandas as pd


class ObjClass(str, Enum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    OTHER_PERSON = "other_person"
    BALL = "ball"


class ShotType(str, Enum):
    MAIN = "main"          # main tactical camera
    CLOSEUP = "closeup"
    REPLAY = "replay"
    GRAPHICS = "graphics"
    OTHER = "other"


class BallState(str, Enum):
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    OUT_OF_FRAME = "out_of_frame"
    IN_FLIGHT = "in_flight"
    HELD = "held"
    UNKNOWN = "unknown"


class RosterState(str, Enum):
    VISIBLE = "visible"
    OFF_FRAME = "off_frame"
    UNCERTAIN = "uncertain"


# ----------------------------------------------------------------- stage 0
@dataclass
class Shot:
    shot_id: str
    start_frame: int
    end_frame: int          # inclusive
    shot_type: ShotType
    fps: float
    width: int
    height: int


# ----------------------------------------------------------------- stage 3
DETECTION_COLUMNS = ["frame", "x1", "y1", "x2", "y2", "score", "cls"]


# ----------------------------------------------------------------- stage 4
TRACK_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "score", "cls",
                 "occl", "margin"]
# occl   : estimated occlusion fraction in [0,1] (0 = fully visible)
# margin : Hungarian assignment margin (best - second best cost); low = ambiguous


# ----------------------------------------------------------------- stage 2
CALIB_COLUMNS = ["frame", "valid", "reproj_err_m"] + [f"h{i}{j}" for i in range(3) for j in range(3)]


# ----------------------------------------------------------------- stage 7
BALL_COLUMNS = ["frame", "x", "y", "score", "state"]


# --------------------------------------------------------- Tier A -> Tier B
@dataclass
class TrackletSummary:
    """Everything Tier B needs about one Tier-A tracklet. No pixels."""
    shot_id: str
    track_id: int
    start_frame: int
    end_frame: int
    n_frames: int
    cls: str
    # appearance
    app_embedding: list[float]            # visibility-weighted mean embedding
    app_weight: float                     # total visibility weight behind it
    color_hist: list[float]               # coarse torso colour histogram (team cue)
    # geometry (image space; pitch space when calibration is valid)
    first_box: list[float]
    last_box: list[float]
    first_pitch_xy: Optional[list[float]] = None
    last_pitch_xy: Optional[list[float]] = None
    # identity evidence (sparse)
    jersey_votes: dict[str, float] = field(default_factory=dict)   # "10" -> summed confidence
    team_cluster: Optional[int] = None
    # quality
    mean_margin: float = 0.0
    frac_occluded: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "TrackletSummary":
        return TrackletSummary(**json.loads(s))


# --------------------------------------------------------- Tier B outputs
IDENTITY_COLUMNS = ["shot_id", "track_id", "identity_id", "team", "jersey",
                    "confidence", "abstained"]

ROSTER_COLUMNS = ["frame", "identity_id", "team", "jersey", "state",
                  "last_x", "last_y", "frames_since_seen"]


# --------------------------------------------------------- Tier C outputs
FUSED_COLUMNS = ["frame", "identity_id", "team", "jersey", "x1", "y1", "x2", "y2",
                 "px", "py", "conf", "link_err_p", "reentry"]
# link_err_p : per-link error probability (calibrated from assignment margin)
# reentry    : 1 on the first frame a re-appearing identity is re-linked


def empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
