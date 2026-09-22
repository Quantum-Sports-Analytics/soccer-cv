"""One-task Cloud Run GPU diagnostic: driver, CUDA, imports, decode, then tier-a under faulthandler.
Usage: python scripts/diag_cloudrun_gpu.py <run_uri>   (run_uri must already hold ingest/ + config.yaml)"""
import sys, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
from soccer_cv.backends.base import ShotTask
from soccer_cv.backends.gcp_cloudrun import CloudRunJobsBackend, CODE_FETCH

STEPS = r'''
echo "== nvidia-smi"; nvidia-smi || echo "nvidia-smi FAILED"
echo "== ldconfig cuda"; ldconfig -p | grep -E "libcuda|libnvidia-ml" | head -5
cd /code && export PYTHONPATH=/code
echo "== python/torch"; python -X faulthandler -c "import torch, sys; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x = torch.randn(512, 512, device='cuda'); print('matmul ok', float((x @ x).sum()))" || echo "TORCH STEP FAILED rc=$?"
echo "== cv2"; python -X faulthandler -c "import cv2; print('cv2', cv2.__version__)" || echo "CV2 STEP FAILED rc=$?"
echo "== decode"; python -X faulthandler -c "
from soccer_cv.core import Storage, frames_iter
import tempfile, os
p = os.path.join(tempfile.mkdtemp(), 'v.mp4'); open(p, 'wb').write(Storage.read_bytes(os.environ['RUN_URI'] + '/ingest/video.mp4'))
n = sum(1 for _ in frames_iter(p, 0, 50)); print('decoded', n)" || echo "DECODE STEP FAILED rc=$?"
echo "== rfdetr"; python -X faulthandler -c "
import os, numpy as np; from soccer_cv.core import load_config; from soccer_cv.stages.s3_detect import build_detector
cfg = load_config(os.environ['CONFIG_URI']); d = build_detector(cfg); print('detector loaded', type(d).__name__)
print('detections on blank frame', len(d(np.zeros((1080, 1920, 3), np.uint8))))" || echo "RFDETR STEP FAILED rc=$?"
echo "== tier-a"; python -X faulthandler -m soccer_cv.cli tier-a --run-uri "$RUN_URI" --shot-id "${SHOTS[$CLOUD_RUN_TASK_INDEX]}" --config "$CONFIG_URI" || echo "TIER-A FAILED rc=$?"
echo "== end"
'''

run = sys.argv[1]
b = CloudRunJobsBackend(project="quantum-analytics-495309", region="europe-west1",
                        image="europe-west1-docker.pkg.dev/quantum-analytics-495309/soccer-cv/tier-a:latest",
                        service_account="soccer-cv-runner@quantum-analytics-495309.iam.gserviceaccount.com",
                        max_parallel=1, task_timeout_s=1200, poll_s=20, max_retries=0)
b.command_override = CODE_FETCH + 'IFS="," read -ra SHOTS <<< "$SHOT_LIST"\n' + STEPS
print(b.run_shots([ShotTask(run, "shot_0000", run + "/config.yaml")]))
