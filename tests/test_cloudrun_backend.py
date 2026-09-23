"""Cloud Run Jobs backend: job spec + lifecycle, with the Cloud Run clients mocked."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

run_v2 = pytest.importorskip("google.cloud.run_v2")

from soccer_cv.backends.base import ShotTask, get_backend
from soccer_cv.backends.gcp_cloudrun import CloudRunJobsBackend

KW = dict(project="quantum-analytics-495309", region="europe-west1",
          image="europe-west1-docker.pkg.dev/quantum-analytics-495309/soccer-cv/tier-a:latest",
          service_account="soccer-cv-runner@quantum-analytics-495309.iam.gserviceaccount.com",
          max_parallel=2, task_timeout_s=1800, poll_s=0)
TASKS = [ShotTask("gs://soccer-cv/runs/r1", f"shot_{i:04d}", "gs://soccer-cv/runs/r1/config.yaml") for i in range(3)]


class FakeOp:
    def __init__(self, value=None, metadata=None):
        self._v, self.metadata = value, metadata

    def result(self, timeout=None):
        return self._v


class FakeJobs:
    def __init__(self):
        self.created, self.deleted, self.ran = None, [], []

    def create_job(self, parent, job, job_id):
        self.created = SimpleNamespace(parent=parent, job=job, job_id=job_id); return FakeOp(job)

    def run_job(self, name):
        self.ran.append(name); return FakeOp(metadata=SimpleNamespace(name=name + "/executions/e1"))

    def delete_job(self, name):
        self.deleted.append(name); return FakeOp()


class FakeExecutions:
    def __init__(self, n, fail_index=None):
        self.n, self.calls, self.fail = n, 0, fail_index

    def get_execution(self, name):
        self.calls += 1
        if self.calls < 2:
            return SimpleNamespace(completion_time=None, running_count=self.n, succeeded_count=0, failed_count=0, cancelled_count=0)
        failed = 1 if self.fail is not None else 0
        return SimpleNamespace(completion_time="t", running_count=0, succeeded_count=self.n - failed, failed_count=failed, cancelled_count=0)

    def cancel_execution(self, name):
        pass


class FakeTasks:
    def __init__(self, n, fail_index=None):
        self.n, self.fail = n, fail_index

    def list_tasks(self, parent):
        st = run_v2.Condition.State
        return [SimpleNamespace(index=i, conditions=[SimpleNamespace(type_="Completed", state=st.CONDITION_FAILED if i == self.fail else st.CONDITION_SUCCEEDED)])
                for i in range(self.n)]


def _backend(fail_index=None):
    b = CloudRunJobsBackend(**KW, code_uri="gs://soccer-cv/code/code-test.tar.gz")
    jobs = FakeJobs()
    b._clients = lambda: (run_v2, jobs, FakeExecutions(len(TASKS), fail_index), FakeTasks(len(TASKS), fail_index))
    return b, jobs


def test_registered():
    assert isinstance(get_backend("cloudrun", **KW), CloudRunJobsBackend)


def test_job_spec():
    b = CloudRunJobsBackend(**KW, code_uri=None)
    job = b.build_job(TASKS, run_v2)
    tmpl = job.template
    assert tmpl.task_count == 3 and tmpl.parallelism == 2
    t = tmpl.template
    assert t.node_selector.accelerator == "nvidia-l4" and t.gpu_zonal_redundancy_disabled
    assert t.timeout.seconds == 1800 if hasattr(t.timeout, "seconds") else str(t.timeout).startswith("0:30")
    assert t.service_account == KW["service_account"]
    c = t.containers[0]
    assert c.image == KW["image"]
    assert dict(c.resources.limits) == {"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "1"}
    env = {e.name: e.value for e in c.env}
    env.pop("PYTHONFAULTHANDLER"); env.pop("PYTHONUNBUFFERED"); assert env.pop("PYTORCH_JIT") == "0"; assert env.pop("PNLCALIB_URI").startswith("gs://")
    assert env == {"RUN_URI": "gs://soccer-cv/runs/r1", "CONFIG_URI": "gs://soccer-cv/runs/r1/config.yaml",
                   "SHOT_LIST": "shot_0000,shot_0001,shot_0002"}
    cmd = c.args[1]
    assert list(c.command) == ["/bin/bash"] and c.args[0] == "-c"
    assert '${SHOTS[$CLOUD_RUN_TASK_INDEX]}' in cmd and "BATCH_TASK_INDEX" not in cmd
    assert cmd.startswith('IFS="," read -ra SHOTS <<< "$SHOT_LIST"; soccer-cv tier-a --run-uri "$RUN_URI"')


def test_job_spec_code_overlay():
    b = CloudRunJobsBackend(**KW, code_uri="gs://soccer-cv/code/code-x.tar.gz")
    c = b.build_job(TASKS, run_v2, "gs://soccer-cv/code/code-x.tar.gz").template.template.containers[0]
    env = {e.name: e.value for e in c.env}
    assert env["CODE_URI"] == "gs://soccer-cv/code/code-x.tar.gz"
    assert "extractall('/code')" in c.args[1] and "PYTHONPATH=/code soccer-cv tier-a" in c.args[1]


def test_run_shots_success_and_cleanup():
    b, jobs = _backend()
    out = b.run_shots(TASKS)
    assert [o["shot_id"] for o in out] == ["shot_0000", "shot_0001", "shot_0002"]
    assert all(o["ok"] and o["state"] == "SUCCEEDED" for o in out)
    assert jobs.created.parent == "projects/quantum-analytics-495309/locations/europe-west1"
    assert jobs.deleted == [f"{jobs.created.parent}/jobs/{jobs.created.job_id}"]      # job removed afterwards


def test_run_shots_partial_failure():
    b, jobs = _backend(fail_index=1)
    out = b.run_shots(TASKS)
    assert [o["ok"] for o in out] == [True, False, True] and out[0]["state"] == "FAILED"
    assert len(jobs.deleted) == 1
