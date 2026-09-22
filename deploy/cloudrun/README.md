# Demo platform on Cloud Run

The platform is a small CPU web service. Each uploaded video becomes one Cloud Batch
job on one L4 GPU; the service itself never holds a GPU.

## 1. Build the images (after any change to `deploy/docker/*` or dependencies)

```bash
gcloud builds submit --config deploy/cloudbuild.yaml --substitutions=_REGION=europe-west1
```

Code-only changes do **not** need a rebuild of the GPU image: every job downloads the
current code as a content-hashed tarball (`gs://soccer-cv/code/code-<hash>.tar.gz`)
published by the platform at submission time. Rebuild the `tier-bc` image (the platform)
to pick up platform code changes.

## 2. Deploy (authenticated — recommended)

GPU work goes to **Cloud Run Jobs** (`BACKEND=cloudrun`, where the L4 quota is). The service
itself drives each run on its CPU (ingest, identity, render), so it must keep CPU while no
request is in flight: `--no-cpu-throttling`, and `--min-instances 1` during a demo session
(set it back to 0 afterwards; an idle 2-vCPU instance is billed while it exists).

```bash
gcloud run deploy soccer-cv-demo \
  --image europe-west1-docker.pkg.dev/quantum-analytics-495309/soccer-cv/tier-bc:latest \
  --region europe-west1 \
  --service-account soccer-cv-submitter@quantum-analytics-495309.iam.gserviceaccount.com \
  --memory 4Gi --cpu 2 --timeout 3600 \
  --max-instances 1 --min-instances 1 --session-affinity --no-cpu-throttling \
  --set-env-vars BACKEND=cloudrun \
  --no-allow-unauthenticated
```

`BACKEND=batch` (the image default) submits one whole-video Batch job instead.

`--max-instances 1 --session-affinity`: upload chunks are assembled on the instance's
disk, so all chunks of one upload must reach the same instance. Fine for a demo; the
production path is direct browser-to-GCS signed uploads.

Open it through an authenticated local tunnel:

```bash
gcloud run services proxy soccer-cv-demo --region europe-west1 --port 8080
# then http://localhost:8080
```

**Do not deploy with `--allow-unauthenticated`**: anyone with the URL could start GPU
jobs. To share with someone, grant them `roles/run.invoker` on the service, or put the
service behind Identity-Aware Proxy.

## 3. Cost and idle-GPU guarantees

- `cloudrun`: one L4 per camera shot; the Cloud Run job is deleted as soon as Tier A
  returns (success, failure or interrupt). `batch`: one L4 VM per video, deleted on exit.
- Every job has a hard `maxRunDuration` (`GPU_TIMEOUT_S`, default 3600 s).
- The header shows the number of active GPU jobs; the **annuler** button deletes the
  job and its VM immediately.
- Check from a terminal at any time:
  `gcloud run jobs list --region europe-west1` (soccer-cv jobs exist only while running) and
  `gcloud batch jobs list --location europe-west1 --filter="status.state!=SUCCEEDED AND status.state!=FAILED"`

## 4. Run locally instead

```bash
pip install -e ".[demo,gcp]"
RUNS_URI=./runs BACKEND=local SCV_CONFIG=configs/local_cpu.yaml uvicorn soccer_cv.demo.app:app --port 8080
```
(CPU only: expect ~30 s of compute per second of video.)
