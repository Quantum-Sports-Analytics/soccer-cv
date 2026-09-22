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

```bash
gcloud run deploy soccer-cv-demo \
  --image europe-west1-docker.pkg.dev/quantum-analytics-495309/soccer-cv/tier-bc:latest \
  --region europe-west1 \
  --service-account soccer-cv-submitter@quantum-analytics-495309.iam.gserviceaccount.com \
  --memory 4Gi --cpu 2 --timeout 3600 \
  --max-instances 1 --session-affinity \
  --no-allow-unauthenticated
```

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

- One L4 (g2-standard-8) per video, on-demand (not Spot, to avoid capacity waits).
  The VM exists only while the job runs; Batch deletes it when the task exits.
- Every job has a hard `maxRunDuration` (`GPU_TIMEOUT_S`, default 3600 s).
- The header shows the number of active GPU jobs; the **annuler** button deletes the
  job and its VM immediately.
- Check from a terminal at any time:
  `gcloud batch jobs list --location europe-west1 --filter="status.state!=SUCCEEDED AND status.state!=FAILED"`

## 4. Run locally instead

```bash
pip install -e ".[demo,gcp]"
RUNS_URI=./runs BACKEND=local SCV_CONFIG=configs/local_cpu.yaml uvicorn soccer_cv.demo.app:app --port 8080
```
(CPU only: expect ~30 s of compute per second of video.)
