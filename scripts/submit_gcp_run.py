#!/usr/bin/env python
"""Submit one whole-video pipeline run as a single Cloud Batch GPU task.

    python scripts/submit_gcp_run.py --video gs://soccer-cv/videos/clip1.mp4 --run gs://soccer-cv/runs/clip1-gpu \
        --config configs/gpu_l4.yaml [--wait]

Uses the Batch REST API with Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS).
The config file is uploaded next to the run so the task reads the exact bytes that were submitted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid

import requests
import google.auth
import google.auth.transport.requests

PROJECT = os.environ.get("GCP_PROJECT", "quantum-analytics-495309")
REGION = os.environ.get("GCP_REGION", "europe-west1")
IMAGE = os.environ.get("TIER_A_IMAGE", f"{REGION}-docker.pkg.dev/{PROJECT}/soccer-cv/tier-a:latest")
RUNTIME_SA = os.environ.get("RUNTIME_SA", f"soccer-cv-runner@{PROJECT}.iam.gserviceaccount.com")
MACHINE = {"nvidia-l4": "g2-standard-8", "nvidia-tesla-a100": "a2-highgpu-1g"}


def token() -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def gcs_put(bucket: str, key: str, data: bytes, tok: str) -> None:
    # virtual-hosted XML API (path-style storage.googleapis.com is not reachable from the sandbox)
    r = requests.put(f"https://{bucket}.storage.googleapis.com/{key}", headers={"Authorization": f"Bearer {tok}"}, data=data, timeout=120)
    r.raise_for_status()


def gcs_get(bucket: str, key: str, tok: str) -> bytes | None:
    r = requests.get(f"https://{bucket}.storage.googleapis.com/{key}", headers={"Authorization": f"Bearer {tok}"}, timeout=120)
    return r.content if r.status_code == 200 else None


def build_job(video: str, run: str, cfg_uri: str, gpu: str, spot: bool, timeout_s: int) -> dict:
    cmd = f'soccer-cv -v run --video-uri "{video}" --run-uri "{run}" --config "{cfg_uri}" --backend local'
    return {
        "taskGroups": [{
            "taskCount": 1,
            "taskSpec": {
                "runnables": [{"container": {"imageUri": IMAGE, "entrypoint": "/bin/bash", "commands": ["-c", cmd],
                                             "options": "--gpus all"}}],
                "computeResource": {"cpuMilli": 8000, "memoryMib": 30000},
                "maxRunDuration": f"{timeout_s}s",
                "maxRetryCount": 1,
            },
        }],
        "allocationPolicy": {
            "instances": [{"installGpuDrivers": True,
                           "policy": {"machineType": MACHINE.get(gpu, "g2-standard-8"),
                                      "provisioningModel": "SPOT" if spot else "STANDARD",
                                      "accelerators": [{"type": gpu, "count": 1}]}}],
            "serviceAccount": {"email": RUNTIME_SA},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"app": "soccer-cv", "kind": "single-run"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/gpu_l4.yaml")
    ap.add_argument("--gpu", default="nvidia-l4"); ap.add_argument("--no-spot", action="store_true")
    ap.add_argument("--timeout", type=int, default=3 * 3600); ap.add_argument("--wait", action="store_true")
    ap.add_argument("--job-id", default=None)
    a = ap.parse_args()
    tok = token()
    m = re.match(r"gs://([^/]+)/(.+)", a.run); bucket, prefix = m.group(1), m.group(2).rstrip("/")
    gcs_put(bucket, f"{prefix}/config.yaml", open(a.config, "rb").read(), tok)
    cfg_uri = f"gs://{bucket}/{prefix}/config.yaml"
    job_id = a.job_id or f"scv-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    url = f"https://batch.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/jobs?job_id={job_id}"
    r = requests.post(url, headers={"Authorization": f"Bearer {tok}"}, json=build_job(a.video, a.run, cfg_uri, a.gpu, not a.no_spot, a.timeout), timeout=60)
    if r.status_code >= 300:
        print("SUBMIT FAILED", r.status_code, r.text[:800]); return 1
    name = r.json()["name"]; print("submitted", name)
    if not a.wait:
        return 0
    t0 = time.time()
    while True:
        time.sleep(30)
        tok = token()
        j = requests.get(f"https://batch.googleapis.com/v1/{name}", headers={"Authorization": f"Bearer {tok}"}, timeout=60).json()
        st = j.get("status", {}); state = st.get("state"); counts = st.get("taskGroups", {}).get("group0", {}).get("counts", {})
        print(f"[{time.time()-t0:5.0f}s] {state} {counts}", flush=True)
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
    man = gcs_get(bucket, f"{prefix}/manifest.json", tok)
    if man:
        for s in json.loads(man)["stages"]:
            print(f"  {s['stage']:16s} {s['seconds']:8.1f}s  {json.dumps(s['metrics'])[:120]}")
    return 0 if state == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())
