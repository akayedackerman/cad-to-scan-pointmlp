#!/usr/bin/env bash
# Download the trained PointMLP checkpoint (300 epochs on ModelNet40) from the GitHub release of this repository.
#
#   bash scripts/download_checkpoint.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=checkpoints/pointMLP_baseline_best.pth
if [[ -f "$OUT" ]]; then
  echo "$OUT already present"
  exit 0
fi
REPO=$(git config --get remote.origin.url | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')
URL="https://github.com/${REPO}/releases/download/v1.0/pointMLP_baseline_best.pth"
mkdir -p checkpoints
echo "downloading $URL"
curl -fL --retry 3 --progress-bar -o "$OUT" "$URL"
echo "saved $OUT ($(du -h "$OUT" | cut -f1))"
