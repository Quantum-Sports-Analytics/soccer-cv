"""Cloud Batch backend: one array job per match, one task per camera shot, on GPU VMs.

Each task runs the Tier-A container image with
    soccer-cv tier-a --run-uri gs://... --shot-id shot_XXXX --config gs://.../config.yaml
The shot id is derived from BATCH_TASK_INDEX inside the container so the job
spec stays a single template. Outputs land directly on GCS via `Storage`, so
a preempted (Spot) task simply re-runs and the `_DONE.json` cache skips
finished stages.

Required IAM on the submitting principal: roles/batch.jobsEditor,
roles/iam.serviceAccountUser (on the job's runtime SA), roles/storage.objectAdmin
on the bucket, roles/logging.viewer to read task logs.
"""
from __future__ import annotations

import logging
import time
import uuid

from .base import Backend, ShotTask

log = logging.getLogger(__name__)

MACHINE_FOR_GPU = {"nvidia-l4": "g2-standard-8", "nvidia-tesla-a100": "a2-highgpu-1g",
                   "nvidia-h100-80gb": "a3-highgpu-1g"}


class CloudBatchBackend(Backend):
    name = "batch"

    def __init__(self, project: str, region: str, image: str, service_account: str | None = None,
                 gpu_type: str = "nvidia-l4", gpu_count: int = 1, spot: bool = True,
                 max_parallel: int = 16, task_timeout_s: int = 3 * 3600, poll_s: int = 30,
                 network: str | None = None, subnetwork: str | None = None):
        self.project, self.region, self.image = project, region, image
        self.sa, self.gpu_type, self.gpu_count, self.spot = service_account, gpu_type, gpu_count, spot
        self.max_parallel, self.task_timeout_s, self.poll_s = max_parallel, task_timeout_s, poll_s
        self.network, self.subnetwork = network, subnetwork

    def _client(self):
        from google.cloud import batch_v1
        return batch_v1, batch_v1.BatchServiceClient()

    def build_job(self, tasks: list[ShotTask]):
        batch_v1, _ = self._client()
        if not tasks:
            raise ValueError("no tasks")
        run_uri, cfg_uri = tasks[0].run_uri, tasks[0].config_uri
        shot_list = ",".join(t.shot_id for t in tasks)

        runnable = batch_v1.Runnable()
        runnable.container = batch_v1.Runnable.Container(
            image_uri=self.image,
            entrypoint="/bin/bash",
            commands=["-c",
                      'IFS="," read -ra SHOTS <<< "$SHOT_LIST"; '
                      'soccer-cv tier-a --run-uri "$RUN_URI" --shot-id "${SHOTS[$BATCH_TASK_INDEX]}" '
                      '--config "$CONFIG_URI"'],
            options="--gpus all" if self.gpu_count else "",
        )
        task = batch_v1.TaskSpec(runnables=[runnable],
                                 max_run_duration=f"{self.task_timeout_s}s", max_retry_count=2)
        task.environment = batch_v1.Environment(variables={"RUN_URI": run_uri, "CONFIG_URI": cfg_uri,
                                                           "SHOT_LIST": shot_list})
        task.compute_resource = batch_v1.ComputeResource(cpu_milli=8000, memory_mib=30000)
        group = batch_v1.TaskGroup(task_spec=task, task_count=len(tasks),
                                   parallelism=min(self.max_parallel, len(tasks)))

        policy = batch_v1.AllocationPolicy.InstancePolicy(
            machine_type=MACHINE_FOR_GPU.get(self.gpu_type, "g2-standard-8"),
            provisioning_model=(batch_v1.AllocationPolicy.ProvisioningModel.SPOT if self.spot
                                else batch_v1.AllocationPolicy.ProvisioningModel.STANDARD))
        if self.gpu_count:
            policy.accelerators = [batch_v1.AllocationPolicy.Accelerator(type_=self.gpu_type, count=self.gpu_count)]
        inst = batch_v1.AllocationPolicy.InstancePolicyOrTemplate(policy=policy, install_gpu_drivers=bool(self.gpu_count))
        alloc = batch_v1.AllocationPolicy(instances=[inst])
        if self.sa:
            alloc.service_account = batch_v1.ServiceAccount(email=self.sa)
        if self.network:
            alloc.network = batch_v1.AllocationPolicy.NetworkPolicy(network_interfaces=[
                batch_v1.AllocationPolicy.NetworkInterface(network=self.network, subnetwork=self.subnetwork or "",
                                                           no_external_ip_address=False)])
        job = batch_v1.Job(task_groups=[group], allocation_policy=alloc,
                           logs_policy=batch_v1.LogsPolicy(destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING),
                           labels={"app": "soccer-cv", "tier": "a"})
        return job

    def run_shots(self, tasks: list[ShotTask]) -> list[dict]:
        batch_v1, client = self._client()
        job = self.build_job(tasks)
        job_id = f"scv-tiera-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        req = batch_v1.CreateJobRequest(parent=f"projects/{self.project}/locations/{self.region}",
                                        job_id=job_id, job=job)
        created = client.create_job(req)
        log.info("submitted Cloud Batch job %s (%d tasks)", created.name, len(tasks))
        t0 = time.time()
        while True:
            j = client.get_job(name=created.name)
            state = j.status.state.name
            counts = {}
            for tg in j.status.task_groups.values():
                counts = dict(tg.counts)
            log.info("job %s: %s %s (%.0fs)", job_id, state, counts, time.time() - t0)
            if state in ("SUCCEEDED", "FAILED", "CANCELLED", "DELETION_IN_PROGRESS"):
                break
            time.sleep(self.poll_s)
        ok = state == "SUCCEEDED"
        return [{"shot_id": t.shot_id, "ok": ok, "job": created.name, "state": state,
                 "seconds": round(time.time() - t0, 1)} for t in tasks]
