"""Match-level orchestration.

Run layout (local dir or gs:// prefix):
  <run>/config.yaml
  <run>/ingest/                 stage 0
  <run>/tier_a/<shot_id>/s3_detect/ s4_track/ s7_ball/ s5_summarize/
  <run>/tier_b/                 identities, roster, graph
  <run>/tier_c/                 fused.parquet, ball.parquet, overlay.mp4
  <run>/manifest.json           stage timings + metrics, appended as stages finish
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from .backends.base import ShotTask, get_backend
from .core import Storage, load_config, is_gcs
from .schema import ShotType
from .stages.s0_ingest import IngestStage, load_shots
from .stages.s3_detect import DetectStage, build_detector
from .stages.s4_track import TrackStage
from .stages.s5_reid import ReIDStage
from .stages.s2_calib import CalibStage
from .stages.s5_summarize import SummarizeStage
from .stages.s7_ball import BallStage
from .tiers.tier_b_identity import IdentityTier
from .tiers.tier_c_render import FuseStage, RenderStage

log = logging.getLogger(__name__)


def _append_manifest(run_uri: str, entry: dict) -> None:
    uri = Storage.join(run_uri, "manifest.json")
    m = Storage.read_json(uri) if Storage.exists(uri) else {"stages": []}
    m["stages"].append(entry)
    Storage.write_json(uri, m)


def run_tier_a_shot(run_uri: str, shot_id: str, cfg: dict, detector=None, replay_uri: str | None = None, encoder=None):
    """Tier A for one shot: detect -> track -> reid/swap-fix -> ball -> summarize. Returns the detector for reuse."""
    ingest = Storage.join(run_uri, "ingest")
    base = Storage.join(run_uri, "tier_a", shot_id)
    if detector is None and replay_uri is None:
        detector = build_detector(cfg)
    st = DetectStage(cfg, shot_id, replay_uri=replay_uri, detector=detector if replay_uri is None else None)
    _append_manifest(run_uri, st.execute(ingest, Storage.join(base, "s3_detect")))
    calib_uri = None
    if cfg.get("calib", {}).get("enabled", True):
        calib_uri = Storage.join(base, "s2_calib")
        _append_manifest(run_uri, CalibStage(cfg, shot_id, ingest).execute(Storage.join(base, "s3_detect"), calib_uri))
    _append_manifest(run_uri, TrackStage(cfg, shot_id, ingest, calib_uri=calib_uri).execute(Storage.join(base, "s3_detect"), Storage.join(base, "s4_track")))
    _append_manifest(run_uri, BallStage(cfg, shot_id, ingest).execute(Storage.join(base, "s3_detect"), Storage.join(base, "s7_ball")))
    if cfg.get("reid", {}).get("enabled", True):
        _append_manifest(run_uri, ReIDStage(cfg, shot_id, ingest, encoder=encoder).execute(Storage.join(base, "s4_track"), Storage.join(base, "s5_reid")))
        track_src = Storage.join(base, "s5_reid")
    else:
        track_src = Storage.join(base, "s4_track")
    _append_manifest(run_uri, SummarizeStage(cfg, shot_id, calib_uri=calib_uri, app_uri=Storage.join(base, "s4_track")).execute(track_src, Storage.join(base, "s5_summarize")))
    return st.detector


def run_match(video_uri: str, run_uri: str, config_path: str, backend: str = "local",
              backend_kwargs: dict | None = None, only_main: bool = True, replay_uri: str | None = None) -> dict:
    t0 = time.time()
    cfg_local = Storage.localize(config_path)          # config may itself live on gs://
    cfg = load_config(cfg_local)
    cfg_uri = Storage.join(run_uri, "config.yaml")
    if cfg_uri != config_path:
        Storage.write_bytes(cfg_uri, Path(cfg_local).read_bytes())

    # ---- stage 0
    ingest_uri = Storage.join(run_uri, "ingest")
    _append_manifest(run_uri, IngestStage(cfg).execute(video_uri, ingest_uri))
    shots = load_shots(ingest_uri)
    if only_main:
        shots = [s for s in shots if s.shot_type == ShotType.MAIN] or shots
    log.info("%d shots to process", len(shots))

    # ---- tier A (sharded by shot)
    tasks = [ShotTask(run_uri, s.shot_id, cfg_uri) for s in shots]
    if replay_uri:                                   # tests / tuning without a detector
        for t in tasks:
            run_tier_a_shot(run_uri, t.shot_id, cfg, replay_uri=replay_uri)
        statuses = [{"shot_id": t.shot_id, "ok": True} for t in tasks]
    else:
        statuses = get_backend(backend, **(backend_kwargs or {})).run_shots(tasks)
    failed = [s for s in statuses if not s.get("ok")]
    if failed:
        log.error("tier A failed for %s", [s["shot_id"] for s in failed])
    if statuses and len(failed) == len(statuses):
        # nothing to identify or render: stop here with a message the platform can show
        job = failed[0].get("job", "")
        raise RuntimeError(f"Tier A failed on every shot ({len(failed)}/{len(statuses)}); GPU job {job} "
                           f"state {failed[0].get('state')}. See the job's logs in Cloud Logging.")

    # ---- tier B
    _append_manifest(run_uri, IdentityTier(cfg).execute(Storage.join(run_uri, "tier_a"), Storage.join(run_uri, "tier_b")))
    # ---- tier C
    _append_manifest(run_uri, FuseStage(cfg).execute(run_uri, Storage.join(run_uri, "tier_c")))
    _append_manifest(run_uri, RenderStage(cfg).execute(run_uri, Storage.join(run_uri, "tier_c")))

    summary = {"run_uri": run_uri, "shots": len(shots), "failed_shots": len(failed),
               "wall_seconds": round(time.time() - t0, 1), "overlay": Storage.join(run_uri, "tier_c", "overlay.mp4")}
    Storage.write_json(Storage.join(run_uri, "summary.json"), summary)
    return summary
