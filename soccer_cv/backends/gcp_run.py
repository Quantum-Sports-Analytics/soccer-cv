"""Whole-video runs on Cloud Batch through the REST API (no gRPC client needed).

One Batch job = one video: a single GPU task runs `soccer-cv run --backend local`
inside the Tier-A image, reading and writing gs://. This is the unit the demo
platform submits. For long matches with many shots, `CloudBatchBackend`
(array job, one task per shot) is the scale-out path.

No GPU can be left idle by construction:
  * the VM exists only while the task runs and Batch deletes it on exit;
  * every task has `maxRunDuration` (default 1 h, hard kill);
  * `cancel()` deletes the job and its VM immediately;
  * `list_active()` lets any caller verify nothing is still running.
"""
from __future__ import annotations

import os
import time

PROJECT = os.environ.get("GCP_PROJECT", "quantum-analytics-495309")
REGION = os.environ.get("GCP_REGION", "europe-west1")
IMAGE = os.environ.get("TIER_A_IMAGE", f"{REGION}-docker.pkg.dev/{PROJECT}/soccer-cv/tier-a:latest")
RUNTIME_SA = os.environ.get("RUNTIME_SA", f"soccer-cv-runner@{PROJECT}.iam.gserviceaccount.com")
MACHINE = {"nvidia-l4": "g2-standard-8", "nvidia-tesla-a100": "a2-highgpu-1g"}
API = "https://batch.googleapis.com/v1"
TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED")

_creds = None


def _hdr() -> dict:
    global _creds
    import google.auth
    import google.auth.transport.requests
    if _creds is None:
        _creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not _creds.valid:
        _creds.refresh(google.auth.transport.requests.Request())
    return {"Authorization": f"Bearer {_creds.token}"}


def _req(method: str, url: str, **kw):
    import requests
    last = None
    for attempt in range(4):
        try:
            return requests.request(method, url, headers=_hdr(), timeout=60, **kw)
        except requests.exceptions.RequestException as e:   # transient proxy / network
            last = e; time.sleep(5 * (attempt + 1))
    raise last


def publish_code(bucket_uri: str = "gs://soccer-cv/code") -> str:
    """Tar soccer_cv/ + configs/ (content-hashed) to GCS; returns its gs:// URI.

    The Tier-A image carries dependencies and weights; code ships per job. A code
    change therefore never needs an image rebuild, and every run records exactly
    which code produced it (the hash is in the URI)."""
    import hashlib
    import io
    import tarfile
    from pathlib import Path
    from ..core import Storage
    root = Path(__file__).resolve().parents[2]
    files = sorted(p for d in ("soccer_cv", "configs") for p in (root / d).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    h = hashlib.sha1()
    for f in files:
        h.update(str(f.relative_to(root)).encode()); h.update(f.read_bytes())
    uri = f"{bucket_uri.rstrip('/')}/code-{h.hexdigest()[:12]}.tar.gz"
    if not Storage.exists(uri):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for f in files:
                tar.add(f, arcname=str(f.relative_to(root)))
        Storage.write_bytes(uri, buf.getvalue())
    return uri


_FETCH = ("python - <<'PY'\n"
          "import io, tarfile\n"
          "from google.cloud import storage\n"
          "b, k = '{b}', '{k}'\n"
          "data = storage.Client().bucket(b).blob(k).download_as_bytes()\n"
          "tarfile.open(fileobj=io.BytesIO(data)).extractall('/code')\n"
          "print('code overlay', '{k}', len(data))\n"
          "PY\n")


def job_spec(video_uri: str, run_uri: str, config_uri: str, gpu: str = "nvidia-l4", spot: bool = False,
             timeout_s: int = 3600, labels: dict | None = None, code_uri: str | None = None) -> dict:
    cmd = f'soccer-cv -v run --video-uri "{video_uri}" --run-uri "{run_uri}" --config "{config_uri}" --backend local'
    if code_uri:
        b, k = code_uri[5:].split("/", 1)
        cmd = "set -e\n" + _FETCH.format(b=b, k=k) + f"cd /code && PYTHONPATH=/code python -m soccer_cv.cli -v run --video-uri \"{video_uri}\" --run-uri \"{run_uri}\" --config \"{config_uri}\" --backend local"
    return {
        "taskGroups": [{
            "taskCount": 1,
            "taskSpec": {
                # no --gpus: Batch mounts GPU + drivers itself (CDI); --gpus breaks container start on COS
                "runnables": [{"container": {"imageUri": IMAGE, "entrypoint": "/bin/bash", "commands": ["-c", cmd]}}],
                "computeResource": {"cpuMilli": 8000, "memoryMib": 30000},
                "maxRunDuration": f"{int(timeout_s)}s",
                "maxRetryCount": 2 if spot else 0,          # Spot preemption -> retry; stage cache resumes
                "lifecyclePolicies": [{"action": "FAIL_TASK", "actionCondition": {"exitCodes": [1, 2]}}],
            },
        }],
        "allocationPolicy": {
            "instances": [{"installGpuDrivers": True,
                           "policy": {"machineType": MACHINE.get(gpu, "g2-standard-8"),
                                      "provisioningModel": "SPOT" if spot else "STANDARD",
                                      "accelerators": [{"type": gpu, "count": 1}]}}],
            "location": {"allowedLocations": [f"regions/{REGION}"]},
            "serviceAccount": {"email": RUNTIME_SA},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"app": "soccer-cv", **(labels or {})},
    }


def submit(video_uri: str, run_uri: str, config_uri: str, job_id: str | None = None, **kw) -> str:
    import uuid
    job_id = job_id or f"scv-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    url = f"{API}/projects/{PROJECT}/locations/{REGION}/jobs?job_id={job_id}"
    r = _req("POST", url, json=job_spec(video_uri, run_uri, config_uri, **kw))
    if r.status_code >= 300:
        raise RuntimeError(f"Batch submit failed {r.status_code}: {r.text[:500]}")
    return r.json()["name"]


def status(name: str) -> dict:
    j = _req("GET", f"{API}/{name}").json()
    st = j.get("status", {})
    events = [e.get("description", "") for e in st.get("statusEvents", [])]
    return {"state": st.get("state", "UNKNOWN"), "uid": j.get("uid"),
            "counts": st.get("taskGroups", {}).get("group0", {}).get("counts", {}),
            "last_event": events[-1][:300] if events else "",
            "run_seconds": float(st.get("runDuration", "0s").rstrip("s") or 0)}


def cancel(name: str) -> int:
    """Delete the job (stops and deletes its VM)."""
    return _req("DELETE", f"{API}/{name}").status_code


def list_active() -> list[dict]:
    r = _req("GET", f"{API}/projects/{PROJECT}/locations/{REGION}/jobs?pageSize=100")
    out = []
    for j in r.json().get("jobs", []):
        st = j.get("status", {}).get("state")
        if st not in TERMINAL:
            out.append({"name": j["name"], "state": st, "created": j.get("createTime")})
    return out


def task_logs(uid: str, n: int = 60) -> list[str]:
    body = {"resourceNames": [f"projects/{PROJECT}"], "orderBy": "timestamp desc", "pageSize": n,
            "filter": f'resource.type="batch.googleapis.com/Job" AND labels.job_uid="{uid}" '
                      f'AND NOT textPayload:"installer.go" AND NOT textPayload:"nvidia"'}
    r = _req("POST", "https://logging.googleapis.com/v2/entries:list", json=body)
    ents = r.json().get("entries", []) if r.status_code == 200 else []
    return [(e.get("timestamp", "")[11:19] + " " + (e.get("textPayload") or str(e.get("jsonPayload", ""))))[:400]
            for e in reversed(ents)]
