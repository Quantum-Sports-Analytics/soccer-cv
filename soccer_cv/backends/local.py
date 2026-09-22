"""Local backend: run shot tasks in-process (sequential) or as subprocesses (parallel).

In-process is the default because on a laptop the detector is the bottleneck
and loading it once beats parallel processes fighting for CPU.
"""
from __future__ import annotations

import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from .base import Backend, ShotTask

log = logging.getLogger(__name__)


class LocalBackend(Backend):
    name = "local"

    def __init__(self, parallel: int = 1, in_process: bool = True):
        self.parallel = parallel
        self.in_process = in_process and parallel == 1

    def run_shots(self, tasks: list[ShotTask]) -> list[dict]:
        if self.in_process:
            from ..pipeline import run_tier_a_shot
            from ..core import load_config, Storage
            out = []
            detector = None
            for t in tasks:
                t0 = time.time()
                cfg = load_config(Storage.localize(t.config_uri))
                detector = run_tier_a_shot(t.run_uri, t.shot_id, cfg, detector=detector)
                out.append({"shot_id": t.shot_id, "ok": True, "seconds": round(time.time() - t0, 1)})
            return out

        def _one(t: ShotTask) -> dict:
            t0 = time.time()
            r = subprocess.run(t.argv(), capture_output=True, text=True)
            return {"shot_id": t.shot_id, "ok": r.returncode == 0, "seconds": round(time.time() - t0, 1),
                    "log": (r.stdout + r.stderr)[-2000:]}
        with ThreadPoolExecutor(self.parallel) as ex:
            return list(ex.map(_one, tasks))
