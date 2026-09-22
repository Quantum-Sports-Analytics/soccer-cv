"""Shared plumbing: config loading + hashing, URI-agnostic storage, stage base class.

A *URI* is either a local path or ``gs://bucket/prefix``. Stages never care
which; they call ``Storage.open_read`` / ``Storage.write`` and the right thing
happens. That is what lets one binary run on a laptop, in a Cloud Batch task
and in a Cloud Run job.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml

log = logging.getLogger("soccer_cv")


# ------------------------------------------------------------------ config
def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict, stage: str | None = None) -> str:
    """Stable hash of the config (whole, or one stage block) excluding `runtime`."""
    sub = cfg.get(stage, {}) if stage else {k: v for k, v in cfg.items() if k != "runtime"}
    blob = json.dumps(sub, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


# ------------------------------------------------------------------ storage
def is_gcs(uri: str) -> bool:
    return str(uri).startswith("gs://")


def _split_gcs(uri: str) -> tuple[str, str]:
    rest = uri[len("gs://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


class Storage:
    """Minimal read/write over local paths and gs:// URIs."""

    _client = None

    @classmethod
    def _gcs(cls):
        if cls._client is None:
            from google.cloud import storage  # lazy: not needed locally
            cls._client = storage.Client()
        return cls._client

    @staticmethod
    def join(base: str, *parts: str) -> str:
        base = str(base).rstrip("/")
        return base + "/" + "/".join(p.strip("/") for p in parts)

    @classmethod
    def exists(cls, uri: str) -> bool:
        if is_gcs(uri):
            b, k = _split_gcs(uri)
            return cls._gcs().bucket(b).blob(k).exists()
        return Path(uri).exists()

    @classmethod
    def read_bytes(cls, uri: str) -> bytes:
        if is_gcs(uri):
            b, k = _split_gcs(uri)
            return cls._gcs().bucket(b).blob(k).download_as_bytes()
        return Path(uri).read_bytes()

    @classmethod
    def write_bytes(cls, uri: str, data: bytes) -> None:
        if is_gcs(uri):
            b, k = _split_gcs(uri)
            cls._gcs().bucket(b).blob(k).upload_from_string(data)
        else:
            p = Path(uri)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, p)

    @classmethod
    def read_df(cls, uri: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(cls.read_bytes(uri)))

    @classmethod
    def write_df(cls, uri: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        cls.write_bytes(uri, buf.getvalue())

    @classmethod
    def read_json(cls, uri: str) -> Any:
        return json.loads(cls.read_bytes(uri))

    @classmethod
    def write_json(cls, uri: str, obj: Any) -> None:
        cls.write_bytes(uri, json.dumps(obj, indent=1, default=str).encode())

    @classmethod
    def localize(cls, uri: str, workdir: str | Path | None = None) -> Path:
        """Return a local path for `uri`, downloading if it lives on GCS."""
        if not is_gcs(uri):
            return Path(uri)
        workdir = Path(workdir or tempfile.mkdtemp(prefix="scv_"))
        workdir.mkdir(parents=True, exist_ok=True)
        dst = workdir / Path(uri).name
        if not dst.exists():
            b, k = _split_gcs(uri)
            cls._gcs().bucket(b).blob(k).download_to_filename(str(dst))
        return dst

    @classmethod
    def upload_file(cls, local: str | Path, uri: str) -> None:
        if is_gcs(uri):
            b, k = _split_gcs(uri)
            cls._gcs().bucket(b).blob(k).upload_from_filename(str(local))
        else:
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, uri)

    @classmethod
    def list(cls, prefix: str) -> list[str]:
        if is_gcs(prefix):
            b, k = _split_gcs(prefix)
            return [f"gs://{b}/{bl.name}" for bl in cls._gcs().list_blobs(b, prefix=k)]
        p = Path(prefix)
        if not p.exists():
            return []
        return [str(x) for x in sorted(p.rglob("*")) if x.is_file()]


# ------------------------------------------------------------------ stages
@dataclass
class StageContext:
    input_uri: str
    output_uri: str
    cfg: dict
    workdir: Path

    def out(self, *parts: str) -> str:
        return Storage.join(self.output_uri, *parts)

    def inp(self, *parts: str) -> str:
        return Storage.join(self.input_uri, *parts)


class Stage(ABC):
    """One pipeline step. Subclasses implement `run`; `execute` adds caching + timing.

    A stage is *done* when `<output_uri>/_DONE.json` exists with a matching config
    hash. That file is the cache key that makes re-runs and preempted Batch
    tasks safe.
    """

    name: str = "stage"
    config_key: str = ""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.params = cfg.get(self.config_key, {}) if self.config_key else {}

    @abstractmethod
    def run(self, ctx: StageContext) -> dict:
        """Do the work; return a small dict of metrics for the manifest."""

    def execute(self, input_uri: str, output_uri: str, force: bool = False) -> dict:
        marker = Storage.join(output_uri, f"_DONE_{self.name}.json")
        h = config_hash(self.cfg, self.config_key or None)
        if not force and Storage.exists(marker):
            prev = Storage.read_json(marker)
            if prev.get("config_hash") == h:
                log.info("[%s] cached at %s", self.name, output_uri)
                return prev
        t0 = time.time()
        workdir = Path(tempfile.mkdtemp(prefix=f"scv_{self.name}_"))
        ctx = StageContext(input_uri, output_uri, self.cfg, workdir)
        metrics = self.run(ctx) or {}
        manifest = {"stage": self.name, "config_hash": h, "input_uri": input_uri,
                    "output_uri": output_uri, "seconds": round(time.time() - t0, 2),
                    "metrics": metrics}
        Storage.write_json(marker, manifest)
        log.info("[%s] done in %.1fs -> %s", self.name, manifest["seconds"], output_uri)
        return manifest


def frames_iter(video_path: str | Path, start: int = 0, end: int | None = None) -> Iterator[tuple[int, "np.ndarray"]]:
    """Yield (frame_index, BGR frame) for frames start..end inclusive."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    while True:
        ok, f = cap.read()
        if not ok or (end is not None and i > end):
            break
        yield i, f
        i += 1
    cap.release()


def video_meta(video_path: str | Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    meta = {"fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "n_frames_reported": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    return meta
