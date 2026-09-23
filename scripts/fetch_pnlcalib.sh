#!/usr/bin/env bash
# Local setup of the learned field calibration used by stage 2 (calib.init: auto|learned).
# Usage: bash scripts/fetch_pnlcalib.sh [target_dir]   then  export PNLCALIB_DIR=<target_dir>
set -euo pipefail
DIR="${1:-third_party/pnlcalib}"
git clone https://github.com/mguti97/PnLCalib.git "$DIR" && git -C "$DIR" checkout 8c87391d6f4ea40c5e4d65e61529916c7a49ce62
mkdir -p "$DIR/weights"
for W in SV_kp SV_lines; do curl -sSfL -o "$DIR/weights/$W" "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/$W"; done
pip install lsq-ellipse shapely
echo "export PNLCALIB_DIR=$(cd "$DIR" && pwd)"
