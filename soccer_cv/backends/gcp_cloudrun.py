"""Cloud Run Jobs backend: Tier A shot-tasks on Cloud Run GPUs (one NVIDIA L4 per task).

Same contract as `CloudBatchBackend`: one job per `run_shots` call, one task per camera
shot, the shot id picked from SHOT_LIST by the task index. Differences from Batch:

  * the index variable is $CLOUD_RUN_TASK_INDEX;
  * GPU = node selector `nvidia-l4` + resource limit `nvidia.com/gpu: 1`, with
    `gpu_zonal_redundancy_disabled=True` (required for the non-redundant L4 quota);
  * no Spot tier; task timeout up to 24 h; `max_retries` re-runs a failed task, and the
    `_DONE_<stage>.json` cache makes the re-run resume where it stopped;
  * the job is deleted when `run_shots` returns (success, failure or interrupt), so no
    GPU configuration is left behind.

Code overlay: by default each task downloads the content-hashed code tarball that
`gcp_run.publish_code()` uploads, and runs it instead of the code baked in the image.
Code changes then never require an image rebuild. Pass `code_uri=None` to run the
image's own code.

Uses the REST transport of google-cloud-run (works where gRPC egress is not available).
Required IAM on the submitting principal: roles/run.developer, plus
roles/iam.serviceAccountUser on the runtime service account.
"""
from __future__ import annotations

import logging
import time
import uuid

from .base import Backend, ShotTask

log = logging.getLogger(__name__)


def _retry(fn, *a, tries: int = 8, wait_s: float = 5.0, what: str = "call", **kw):
    """Retry transient network / 5xx errors (a dropped proxy connection must not orphan a GPU job)."""
    for k in range(tries):
        try:
            return fn(*a, **kw)
        except Exception as e:                             # noqa: BLE001
            if k == tries - 1 or getattr(e, "code", None) in (400, 401, 403, 404):
                raise
            log.warning("%s failed (%s), retry %d/%d", what, type(e).__name__, k + 1, tries - 1)
            time.sleep(wait_s * (k + 1))

TIER_A_CMD = ('IFS="," read -ra SHOTS <<< "$SHOT_LIST"; '
              '{prefix}soccer-cv tier-a --run-uri "$RUN_URI" --shot-id "${{SHOTS[$CLOUD_RUN_TASK_INDEX]}}" '
              '--config "$CONFIG_URI"')

CODE_FETCH = ("python - <<'PY'\n"
              "import io, os, tarfile\n"
              "from google.cloud import storage\n"
              "u = os.environ['CODE_URI'][5:]; b, k = u.split('/', 1)\n"
              "data = storage.Client().bucket(b).blob(k).download_as_bytes()\n"
              "tarfile.open(fileobj=io.BytesIO(data)).extractall('/code')\n"
              "print('code overlay', k, len(data))\n"
              "PY\n")


# Learned field calibration (PnLCalib code + weights, ~530 MB), fetched once per task unless the
# image already ships it (PNLCALIB_DIR set). Its two pure/binary-wheel deps are installed alongside.
PNLCALIB_URI = "gs://soccer-cv/models/pnlcalib-v1.tar"
MODELS_FETCH = ("if [ -n \"$PNLCALIB_URI\" ] && [ ! -d \"${PNLCALIB_DIR:-/nonexistent}\" ]; then\n"
                "python - <<'PY'\n"
                "import os, tarfile\n"
                "from google.cloud import storage\n"
                "u = os.environ['PNLCALIB_URI'][5:]; b, k = u.split('/', 1)\n"
                "storage.Client().bucket(b).blob(k).download_to_filename('/tmp/pnl.tar')\n"
                "tarfile.open('/tmp/pnl.tar').extractall('/opt/pnlcalib'); os.remove('/tmp/pnl.tar')\n"
                "print('pnlcalib fetched', k)\n"
                "PY\n"
                "pip install -q lsq-ellipse shapely || true\n"
                "export PNLCALIB_DIR=/opt/pnlcalib\n"
                "fi\n")


class CloudRunJobsBackend(Backend):
    name = "cloudrun"

    def __init__(self, project: str, region: str, image: str, service_account: str | None = None,
                 max_parallel: int = 8, task_timeout_s: int = 3600, poll_s: int = 15,
                 gpu_type: str = "nvidia-l4", cpu: str = "8", memory: str = "32Gi", max_retries: int = 1,
                 code_uri: str | None = "auto", keep_job: bool = False):
        self.project, self.region, self.image, self.sa = project, region, image, service_account
        self.max_parallel, self.task_timeout_s, self.poll_s = max_parallel, task_timeout_s, poll_s
        self.gpu_type, self.cpu, self.memory, self.max_retries = gpu_type, cpu, memory, max_retries
        self.code_uri, self.keep_job = code_uri, keep_job

    # ------------------------------------------------------------ clients
    def _clients(self):
        from google.cloud import run_v2
        return (run_v2, run_v2.JobsClient(transport="rest"), run_v2.ExecutionsClient(transport="rest"),
                run_v2.TasksClient(transport="rest"))

    @property
    def parent(self) -> str:
        return f"projects/{self.project}/locations/{self.region}"

    # ------------------------------------------------------------ spec
    def command(self, with_code: bool) -> str:
        if getattr(self, "command_override", None):
            return self.command_override
        if with_code:
            return "set -e\n" + CODE_FETCH + MODELS_FETCH + TIER_A_CMD.format(prefix="cd /code && PYTHONPATH=/code ")
        return TIER_A_CMD.format(prefix="")

    def build_job(self, tasks: list[ShotTask], run_v2=None, code_uri: str | None = None):
        if run_v2 is None:
            from google.cloud import run_v2
        if not tasks:
            raise ValueError("no tasks")
        run_uri, cfg_uri = tasks[0].run_uri, tasks[0].config_uri
        # PYTHONFAULTHANDLER: a native crash (segfault in CUDA / OpenCV / decoder) prints the Python
        # stack of every thread to stderr, i.e. into Cloud Logging, instead of a bare "Segmentation fault".
        env = {"RUN_URI": run_uri, "CONFIG_URI": cfg_uri, "SHOT_LIST": ",".join(t.shot_id for t in tasks),
               "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1",
               # rfdetr's import-time torch.jit.script segfaults on torch 2.6 (image built from the
               # cu124 index); eager mode is equivalent for these box-geometry helpers.
               "PYTORCH_JIT": "0", "PNLCALIB_URI": PNLCALIB_URI}
        if code_uri:
            env["CODE_URI"] = code_uri
        container = run_v2.Container(
            image=self.image,
            command=["/bin/bash"],
            args=["-c", self.command(bool(code_uri))],
            env=[run_v2.EnvVar(name=k, value=v) for k, v in env.items()],
            resources=run_v2.ResourceRequirements(limits={"cpu": self.cpu, "memory": self.memory, "nvidia.com/gpu": "1"}),
        )
        task = run_v2.TaskTemplate(
            containers=[container],
            max_retries=self.max_retries,
            timeout=f"{int(self.task_timeout_s)}s",
            node_selector=run_v2.NodeSelector(accelerator=self.gpu_type),
            gpu_zonal_redundancy_disabled=True,
        )
        if self.sa:
            task.service_account = self.sa
        return run_v2.Job(
            template=run_v2.ExecutionTemplate(task_count=len(tasks), parallelism=min(self.max_parallel, len(tasks)),
                                              template=task),
            labels={"app": "soccer-cv", "tier": "a"},
        )

    def _resolve_code_uri(self, tasks: list[ShotTask]) -> str | None:
        if self.code_uri != "auto":
            return self.code_uri
        from .gcp_run import publish_code
        run_uri = tasks[0].run_uri
        if not run_uri.startswith("gs://"):
            return None
        bucket = run_uri[5:].split("/", 1)[0]
        return publish_code(f"gs://{bucket}/code")

    # ------------------------------------------------------------ run
    def run_shots(self, tasks: list[ShotTask]) -> list[dict]:
        run_v2, jobs, executions, task_client = self._clients()
        code_uri = self._resolve_code_uri(tasks)
        job_id = f"scv-tiera-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        job_name = f"{self.parent}/jobs/{job_id}"
        t0 = time.time()
        exec_name, state, per_task = None, "UNKNOWN", {}
        try:
            jobs.create_job(parent=self.parent, job=self.build_job(tasks, run_v2, code_uri), job_id=job_id).result(timeout=600)
            log.info("created Cloud Run job %s (%d tasks, code %s)", job_name, len(tasks), code_uri)
            op = jobs.run_job(name=job_name)
            exec_name = op.metadata.name if getattr(op, "metadata", None) is not None and op.metadata.name else None
            if exec_name is None:                              # fall back to the job's latest execution
                exec_name = jobs.get_job(name=job_name).latest_created_execution.name
            while True:
                ex = _retry(executions.get_execution, name=exec_name, what="get_execution")
                done = bool(ex.completion_time)
                log.info("cloudrun %s: running %d succeeded %d failed %d (%.0fs)", job_id, ex.running_count,
                         ex.succeeded_count, ex.failed_count, time.time() - t0)
                if done or (ex.succeeded_count + ex.failed_count + ex.cancelled_count) >= len(tasks):
                    state = "SUCCEEDED" if ex.succeeded_count == len(tasks) else "FAILED"
                    break
                time.sleep(self.poll_s)
            try:
                for t in _retry(lambda: list(task_client.list_tasks(parent=exec_name)), what="list_tasks"):
                    ok = any(c.type_ == "Completed" and c.state.name == "CONDITION_SUCCEEDED" for c in t.conditions)
                    per_task[int(t.index)] = ok
            except Exception as e:                             # noqa: BLE001 — per-task detail is best effort
                log.warning("could not list tasks: %s", e)
        except BaseException:
            state = "CANCELLED"
            if exec_name:
                try:
                    _retry(executions.cancel_execution, name=exec_name, what="cancel_execution", tries=4)
                except Exception:                              # noqa: BLE001
                    pass
            raise
        finally:
            if not self.keep_job:
                try:
                    _retry(jobs.delete_job, name=job_name, what="delete_job")
                    log.info("deleted Cloud Run job %s", job_name)
                except Exception as e:                         # noqa: BLE001
                    log.warning("could not delete job %s: %s", job_name, e)
        secs = round(time.time() - t0, 1)
        return [{"shot_id": t.shot_id, "ok": per_task.get(i, state == "SUCCEEDED"), "job": job_name, "state": state,
                 "seconds": secs} for i, t in enumerate(tasks)]


def list_active(project: str, region: str) -> list[dict]:
    """soccer-cv Cloud Run jobs that currently have running tasks (i.e. hold GPUs)."""
    from google.cloud import run_v2
    jobs, executions = run_v2.JobsClient(transport="rest"), run_v2.ExecutionsClient(transport="rest")
    out = []
    for j in jobs.list_jobs(parent=f"projects/{project}/locations/{region}"):
        if j.labels.get("app") != "soccer-cv":
            continue
        for ex in executions.list_executions(parent=j.name):
            if not ex.completion_time:
                out.append({"name": ex.name, "state": "RUNNING", "running_tasks": int(ex.running_count),
                            "created": str(ex.create_time)})
    return out
