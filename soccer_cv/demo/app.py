"""Demo platform: upload a broadcast clip, run the pipeline, watch the overlay.

Local:     uvicorn soccer_cv.demo.app:app --reload
Cloud Run: same image; set RUNS_URI=gs://<bucket>/runs and BACKEND=batch with
           BACKEND_KWARGS='{"project":..., "region":..., "image":..., "service_account":...}'.

Jobs run in a background thread (local) or as a Cloud Batch job (GCP); the page
polls /api/jobs/{id} and shows manifest metrics as stages finish — so the user
sees detection density, tracklet counts, re-links and abstentions, not only a video.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from ..core import Storage, is_gcs

HERE = Path(__file__).parent
RUNS_URI = os.environ.get("RUNS_URI", str(Path("runs").resolve()))
CONFIG = os.environ.get("SCV_CONFIG", str(HERE.parents[1] / "configs" / "default.yaml"))
BACKEND = os.environ.get("BACKEND", "local")
BACKEND_KWARGS = json.loads(os.environ.get("BACKEND_KWARGS", "{}"))

app = FastAPI(title="soccer-cv demo")
templates = Jinja2Templates(directory=str(HERE / "templates"))
JOBS: dict[str, dict] = {}


def _run_job(job_id: str, video_uri: str, run_uri: str) -> None:
    from ..pipeline import run_match
    JOBS[job_id].update(status="running", started=time.time())
    try:
        out = run_match(video_uri, run_uri, CONFIG, backend=BACKEND, backend_kwargs=BACKEND_KWARGS)
        JOBS[job_id].update(status="done", summary=out, finished=time.time())
    except Exception as e:                       # noqa: BLE001
        JOBS[job_id].update(status="failed", error=repr(e), finished=time.time())


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "backend": BACKEND})


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_uri = Storage.join(RUNS_URI, job_id)
    video_uri = Storage.join(run_uri, "input.mp4")
    data = await file.read()
    Storage.write_bytes(video_uri, data)
    JOBS[job_id] = {"id": job_id, "status": "queued", "filename": file.filename, "run_uri": run_uri,
                    "size_mb": round(len(data) / 1e6, 1)}
    threading.Thread(target=_run_job, args=(job_id, video_uri, run_uri), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404)
    man_uri = Storage.join(j["run_uri"], "manifest.json")
    stages = Storage.read_json(man_uri)["stages"] if Storage.exists(man_uri) else []
    return {**j, "stages": [{"stage": s["stage"], "seconds": s["seconds"], "metrics": s.get("metrics", {})} for s in stages]}


@app.get("/api/jobs/{job_id}/overlay.mp4")
def overlay(job_id: str):
    j = JOBS.get(job_id)
    if not j or j.get("status") != "done":
        raise HTTPException(404)
    uri = Storage.join(j["run_uri"], "tier_c", "overlay.mp4")
    if is_gcs(uri):
        return StreamingResponse(iter([Storage.read_bytes(uri)]), media_type="video/mp4")
    return FileResponse(uri, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/tracks.json")
def tracks(job_id: str):
    """Fused per-frame tracks for client-side drawing / inspection."""
    j = JOBS.get(job_id)
    if not j or j.get("status") != "done":
        raise HTTPException(404)
    df = Storage.read_df(Storage.join(j["run_uri"], "tier_c", "fused.parquet"))
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/healthz")
def healthz():
    return {"ok": True, "backend": BACKEND}
