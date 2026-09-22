"""Demo platform: upload a broadcast clip, run the pipeline on a GPU, review the overlay.

Local (pipeline in a thread on this machine):
    RUNS_URI=./runs uvicorn soccer_cv.demo.app:app --port 8080
Cloud Run (pipeline as a Cloud Batch GPU job, see deploy/cloudrun/README.md):
    RUNS_URI=gs://soccer-cv/runs BACKEND=batch

Everything about a run lives under <RUNS_URI>/<run_id>/ (job.json + pipeline outputs),
so runs survive restarts and the platform keeps a history of every video tested —
the point of the platform is comparing the base model across many broadcasts.

Upload is chunked (8 MB parts) because Cloud Run caps request bodies at 32 MB.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from ..core import Storage

HERE = Path(__file__).parent
REPO = HERE.parents[1]
RUNS_URI = os.environ.get("RUNS_URI", str((REPO / "runs").resolve()))
CONFIG = os.environ.get("SCV_CONFIG", str(REPO / "configs" / ("gpu_l4.yaml" if os.environ.get("BACKEND") == "batch" else "default.yaml")))
BACKEND = os.environ.get("BACKEND", "local")
GPU_TIMEOUT_S = int(os.environ.get("GPU_TIMEOUT_S", "3600"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "2000"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", tempfile.gettempdir())) / "scv_uploads"

app = FastAPI(title="soccer-cv demo")
templates = Jinja2Templates(directory=str(HERE / "templates"))
_lock = threading.Lock()


# ------------------------------------------------------------------ run records
def _run_uri(run_id: str) -> str:
    if "/" in run_id or ".." in run_id:
        raise HTTPException(400, "bad run id")
    return Storage.join(RUNS_URI, run_id)


def _job(run_id: str) -> dict:
    uri = Storage.join(_run_uri(run_id), "job.json")
    if not Storage.exists(uri):
        raise HTTPException(404, "unknown run")
    return Storage.read_json(uri)


def _save_job(job: dict) -> None:
    Storage.write_json(Storage.join(_run_uri(job["id"]), "job.json"), job)
    with _lock:                                   # small index so the history loads with one read
        idx_uri = Storage.join(RUNS_URI, "index.json")
        idx = Storage.read_json(idx_uri) if Storage.exists(idx_uri) else []
        idx = [r for r in idx if r["id"] != job["id"]]
        idx.append({k: job.get(k) for k in ("id", "filename", "created", "status", "size_mb")})
        Storage.write_json(idx_uri, sorted(idx, key=lambda r: r["created"], reverse=True))


def _refresh(job: dict) -> dict:
    """Pull the live state of a Batch job into the record (lazy, on read)."""
    if job.get("status") in ("done", "failed", "cancelled") or not job.get("batch_name"):
        return job
    from ..backends import gcp_run
    try:
        st = gcp_run.status(job["batch_name"])
    except Exception as e:                        # noqa: BLE001
        job["batch_error"] = repr(e)[:200]; return job
    job["batch_state"], job["batch_event"] = st["state"], st["last_event"]
    job["gpu_seconds"] = st["run_seconds"]
    new = {"QUEUED": "queued", "SCHEDULED": "waiting_gpu", "RUNNING": "running",
           "SUCCEEDED": "done", "FAILED": "failed", "CANCELLED": "cancelled"}.get(st["state"], job["status"])
    if new != job["status"]:
        job["status"] = new
        if new in ("done", "failed", "cancelled"):
            job["finished"] = time.time()
            if new == "failed":
                try:
                    job["log_tail"] = gcp_run.task_logs(st["uid"], 40)[-25:]
                except Exception:                 # noqa: BLE001
                    pass
        _save_job(job)
    return job


# ------------------------------------------------------------------ execution
def _run_local(job: dict) -> None:
    from ..pipeline import run_match
    job.update(status="running", started=time.time()); _save_job(job)
    try:
        run_match(job["video_uri"], job["run_uri"], CONFIG, backend="local")
        job.update(status="done", finished=time.time())
    except Exception as e:                        # noqa: BLE001
        job.update(status="failed", error=repr(e)[:500], finished=time.time())
    _save_job(job)


def _start(job: dict) -> None:
    cfg_uri = Storage.join(job["run_uri"], "config.yaml")
    Storage.write_bytes(cfg_uri, Path(CONFIG).read_bytes())
    if BACKEND == "batch":
        from ..backends import gcp_run
        code_uri = gcp_run.publish_code(Storage.join(RUNS_URI.rsplit("/runs", 1)[0], "code"))
        job["code_uri"] = code_uri
        job["batch_name"] = gcp_run.submit(job["video_uri"], job["run_uri"], cfg_uri,
                                           job_id=f"scv-{job['id']}", timeout_s=GPU_TIMEOUT_S,
                                           code_uri=code_uri, labels={"run": job["id"]})
        job["status"] = "queued"; _save_job(job)
    else:
        threading.Thread(target=_run_local, args=(job,), daemon=True).start()


# ------------------------------------------------------------------ upload API
@app.post("/api/uploads")
async def upload_start(request: Request):
    body = await request.json()
    size_mb = float(body.get("size", 0)) / 1e6
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(413, f"video larger than {MAX_UPLOAD_MB} MB")
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    (UPLOAD_DIR / run_id).mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / run_id / "meta.json").write_text(json.dumps({"filename": body.get("filename", "video.mp4"), "size_mb": round(size_mb, 1)}))
    return {"run_id": run_id, "chunk_bytes": 8 * 1024 * 1024}


@app.put("/api/uploads/{run_id}/{index}")
async def upload_chunk(run_id: str, index: int, request: Request):
    d = UPLOAD_DIR / run_id
    if not d.exists() or "/" in run_id:
        raise HTTPException(404)
    (d / f"part{index:05d}").write_bytes(await request.body())
    return {"ok": True}


@app.post("/api/uploads/{run_id}/finish")
def upload_finish(run_id: str):
    d = UPLOAD_DIR / run_id
    if not d.exists():
        raise HTTPException(404)
    meta = json.loads((d / "meta.json").read_text())
    video = d / "input.mp4"
    with open(video, "wb") as out:
        for part in sorted(d.glob("part*")):
            out.write(part.read_bytes()); part.unlink()
    run_uri = _run_uri(run_id)
    video_uri = Storage.join(run_uri, "input.mp4")
    Storage.upload_file(video, video_uri)
    shutil.rmtree(d, ignore_errors=True)
    job = {"id": run_id, "filename": meta["filename"], "size_mb": meta["size_mb"], "created": time.time(),
           "status": "submitting", "run_uri": run_uri, "video_uri": video_uri, "backend": BACKEND,
           "config": Path(CONFIG).name}
    _save_job(job)
    try:
        _start(job)
    except Exception as e:                        # noqa: BLE001
        job.update(status="failed", error=repr(e)[:500]); _save_job(job)
    return {"run_id": run_id, "status": job["status"]}


# ------------------------------------------------------------------ runs API
@app.get("/api/runs")
def runs():
    idx_uri = Storage.join(RUNS_URI, "index.json")
    return Storage.read_json(idx_uri) if Storage.exists(idx_uri) else []


def _stage_metrics(run_uri: str) -> list[dict]:
    uri = Storage.join(run_uri, "manifest.json")
    if not Storage.exists(uri):
        return []
    return [{"stage": s["stage"], "seconds": s["seconds"], "metrics": s.get("metrics", {})}
            for s in Storage.read_json(uri)["stages"]]


def _report(stages: list[dict]) -> dict:
    """The per-video quality card: what to look at before (and while) watching the overlay."""
    m = {}
    for s in stages:
        m.setdefault(s["stage"], {}).update(s["metrics"])
    g = lambda st, k: m.get(st, {}).get(k)              # noqa: E731
    return {
        "frames": g("s0_ingest", "n_frames"), "shots": g("s0_ingest", "n_shots"),
        "persons_per_frame": g("s3_detect", "persons_per_frame"),
        "calibrated_frac": g("s2_calib", "valid_frac"), "calib_err_px": g("s2_calib", "median_err_px"),
        "off_pitch_rejected": g("s4_track", "off_pitch_rejected"),
        "tracklets": g("s4_track", "n_tracks"),
        "ambiguous_windows": g("s5_reid", "n_windows"), "swaps_fixed": g("s5_reid", "n_swapped"),
        "undecidable_windows": g("s5_reid", "n_undecidable"),
        "identities": g("tier_b_identity", "n_identities"), "abstained_tracklets": g("tier_b_identity", "n_abstained"),
        "relinks": g("tier_b_identity", "n_relinks"),
        "ball_visible_frac": g("s7_ball", "visible_frac"), "ball_located_frac": g("s7_ball", "located_frac"),
        "compute_seconds": round(sum(s["seconds"] for s in stages), 1),
    }


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    job = _refresh(_job(run_id))
    stages = _stage_metrics(job["run_uri"])
    return {**job, "stages": stages, "report": _report(stages)}


@app.post("/api/runs/{run_id}/cancel")
def run_cancel(run_id: str):
    job = _job(run_id)
    if job.get("batch_name"):
        from ..backends import gcp_run
        gcp_run.cancel(job["batch_name"])
    job.update(status="cancelled", finished=time.time()); _save_job(job)
    return {"ok": True}


@app.get("/api/runs/{run_id}/windows")
def run_windows(run_id: str):
    """Ambiguous crossings with their decision, in seconds, for click-to-seek review."""
    job = _job(run_id)
    fps = 25.0
    meta_uri = Storage.join(job["run_uri"], "ingest", "meta.json")
    if Storage.exists(meta_uri):
        fps = float(Storage.read_json(meta_uri).get("fps", 25.0)) or 25.0
    out = []
    for u in Storage.list(Storage.join(job["run_uri"], "tier_a")):
        if u.endswith("s5_reid/windows.json"):
            for w in Storage.read_json(u):
                out.append({"t0": round(w["f0"] / fps, 2), "t1": round(w["f1"] / fps, 2), "tracks": [w["a"], w["b"]],
                            "decision": w.get("decision"), "margin": w.get("margin")})
    return sorted(out, key=lambda w: w["t0"])


@app.get("/api/runs/{run_id}/video/{which}")
def run_video(run_id: str, which: str, request: Request):
    """Range-aware video streaming (seeking works for GCS-hosted files too)."""
    job = _job(run_id)
    rel = {"overlay": "tier_c/overlay.mp4", "input": "input.mp4"}.get(which)
    if rel is None:
        raise HTTPException(404)
    uri = Storage.join(job["run_uri"], rel)
    size = Storage.size(uri)
    if not size:
        raise HTTPException(404, "not ready")
    rng = request.headers.get("range")
    start, end = 0, size - 1
    if rng and rng.startswith("bytes="):
        a, _, b = rng[6:].partition("-")
        start = int(a) if a else 0
        end = min(int(b), size - 1) if b else min(start + 8 * 1024 * 1024 - 1, size - 1)
        data = Storage.read_range(uri, start, end)
        return Response(data, status_code=206, media_type="video/mp4",
                        headers={"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes",
                                 "Content-Length": str(len(data))})
    end = min(8 * 1024 * 1024 - 1, size - 1)
    data = Storage.read_range(uri, 0, end)
    return Response(data, status_code=206, media_type="video/mp4",
                    headers={"Content-Range": f"bytes 0-{end}/{size}", "Accept-Ranges": "bytes", "Content-Length": str(len(data))})


@app.get("/api/gpu")
def gpu_status():
    """Every Batch job not yet terminal — the check that no GPU is running unattended."""
    if BACKEND != "batch":
        return {"backend": BACKEND, "active": []}
    from ..backends import gcp_run
    return {"backend": BACKEND, "active": gcp_run.list_active()}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"backend": BACKEND})


@app.get("/healthz")
def healthz():
    return {"ok": True, "backend": BACKEND, "runs_uri": RUNS_URI}
