"""Execution backends: where Tier A shot-tasks run.

A backend receives a list of `ShotTask`s — each is one CLI invocation of
`soccer-cv tier-a --run-uri ... --shot-id ...` — and runs them to completion.
The pipeline never imports GCP unless the `batch` backend is chosen.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ShotTask:
    run_uri: str
    shot_id: str
    config_uri: str
    extra_args: list[str] = field(default_factory=list)

    def argv(self) -> list[str]:
        return ["soccer-cv", "tier-a", "--run-uri", self.run_uri, "--shot-id", self.shot_id,
                "--config", self.config_uri, *self.extra_args]


class Backend(ABC):
    name = "base"

    @abstractmethod
    def run_shots(self, tasks: list[ShotTask]) -> list[dict]:
        """Run all tasks; return one status dict per task ({shot_id, ok, seconds, log})."""


def get_backend(name: str, **kw) -> Backend:
    if name == "local":
        from .local import LocalBackend
        return LocalBackend(**kw)
    if name == "batch":
        from .gcp_batch import CloudBatchBackend
        return CloudBatchBackend(**kw)
    raise ValueError(f"unknown backend {name!r}")
