# soccer-cv

Broadcast football perception: players, goalkeepers, officials and ball tracked with
persistent identity from a single TV feed, plus an overlay demo.
Design: `docs/architecture_and_difficulties.md` (three tiers; camera shots are the unit
of parallelism; identity is resolved globally with abstention).

## Layout

```
soccer_cv/
  schema.py            data contracts between stages (what crosses tier boundaries)
  core.py              config hashing, local/gs:// storage, Stage base class with caching
  stages/
    s0_ingest.py       decode, shot cuts, shot type          -> ingest/shots.json
    s3_detect.py       RF-DETR persons + ball, adaptive tiling -> tier_a/<shot>/s3_detect
    s4_track.py        Kalman + EIoU + occlusion-corrected cost + assignment margin
    s5_summarize.py    tracklet summaries (no pixels)        -> Tier A/B boundary
    s7_ball.py         ball state machine + interpolation
  tiers/
    tier_b_identity.py team clustering, tracklet graph, identity chaining, cardinality, abstention
    tier_c_render.py   fusion + overlay renderer (confidence, re-entry flags, roster panel)
  backends/            local (in-process / subprocess) and Cloud Batch (array job per match)
  pipeline.py          match orchestration; run layout documented in the module docstring
  cli.py               `soccer-cv run | tier-a | tier-b | tier-c | ingest`
  demo/                FastAPI upload -> run -> overlay viewer
configs/default.yaml   every stage reads its own block; hash of the block is the cache key
deploy/                Dockerfiles (tier-a GPU, tier-bc CPU), cloudbuild.yaml, setup_gcp.sh
tests/                 synthetic end-to-end test (no GPU, no detector)
```

Stages not yet implemented (documented stubs in the design doc): s1/s2 field calibration,
s6 selective mask propagation, s9 jersey OCR, s11 ball 3D. Their outputs are optional
inputs everywhere, so the pipeline runs without them.

## Local

```bash
conda create -n soccer-cv python=3.11 ffmpeg av -c conda-forge && conda activate soccer-cv
pip install -e ".[gpu,demo,dev]"
export RF_HOME=$PWD/weights            # where RF-DETR caches weights
pytest -q                              # synthetic end-to-end, ~10 s
soccer-cv run --video-uri clip.mp4 --run-uri runs/clip --config configs/default.yaml
uvicorn soccer_cv.demo.app:app --reload   # http://127.0.0.1:8000
```

## GCP (project quantum-analytics-495309)

One-time, as project owner:

```bash
PROJECT=quantum-analytics-495309 REGION=europe-west1 bash deploy/setup_gcp.sh
# -> creates bucket, Artifact Registry repo, runtime SA, submitter SA + key (submitter-key.json)
gcloud builds submit --config deploy/cloudbuild.yaml --substitutions=_REGION=europe-west1
```

Register `submitter-key.json` in the Claude Science workspace (Customize -> Credentials -> GCP),
then delete the local copy. The submitter can only: create Batch jobs, act as the runtime SA,
read/write the bucket, read logs.

Run a match on GPU (Cloud Batch, one task per camera shot, L4 Spot):

```bash
soccer-cv run --video-uri gs://$BUCKET/in/match.mp4 --run-uri gs://$BUCKET/runs/match \
  --config configs/default.yaml --backend batch --backend-kwargs \
  '{"project":"quantum-analytics-495309","region":"europe-west1",
    "image":"europe-west1-docker.pkg.dev/quantum-analytics-495309/soccer-cv/tier-a:latest",
    "service_account":"soccer-cv-runner@quantum-analytics-495309.iam.gserviceaccount.com",
    "gpu_type":"nvidia-l4","spot":true}'
```

Demo on Cloud Run (CPU image; it submits Batch jobs for Tier A):

```bash
gcloud run deploy soccer-cv-demo --region europe-west1 \
  --image europe-west1-docker.pkg.dev/quantum-analytics-495309/soccer-cv/tier-bc:latest \
  --service-account soccer-cv-runner@quantum-analytics-495309.iam.gserviceaccount.com \
  --memory 4Gi --cpu 2 --timeout 3600 --no-allow-unauthenticated \
  --set-env-vars RUNS_URI=gs://$BUCKET/runs,BACKEND=batch,BACKEND_KWARGS='{"project":"quantum-analytics-495309","region":"europe-west1","image":"europe-west1-docker.pkg.dev/quantum-analytics-495309/soccer-cv/tier-a:latest","service_account":"soccer-cv-runner@quantum-analytics-495309.iam.gserviceaccount.com"}'
```

## Run layout (local dir or gs:// prefix)

```
<run>/config.yaml
<run>/ingest/{video.mp4, shots.json, meta.json}
<run>/tier_a/<shot_id>/{s3_detect, s4_track, s7_ball, s5_summarize}/
<run>/tier_b/{identities.parquet, roster.parquet, identity_graph.json}
<run>/tier_c/{fused.parquet, ball.parquet, overlay.mp4}
<run>/manifest.json      per-stage timings + metrics
```

Every stage writes `_DONE_<stage>.json` with its config hash; re-runs and preempted
Spot tasks skip finished stages.
