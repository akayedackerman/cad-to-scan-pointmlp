#!/usr/bin/env bash
# Data preparation, visualisation, testing and plots. Logs are written to outputs/logs/.
#
#   bash run_all.sh
#   SKIP_CLIP=1 bash run_all.sh   # without CLIP
#
# Training: bash train.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
export PYTHONUNBUFFERED=1
LOG=outputs/logs
mkdir -p "$LOG"

run() {  # run <log name> <command>
  local name=$1; shift
  echo "================ $name ================"
  { echo "\$ $*"; "$@" 2>&1 | grep --line-buffered -v -i "warning"; } | tee "$LOG/$name.log"
}

# Point-cloud data preparation
run prepare_data       $PY scripts/prepare_data.py --workers 8
run make_hpr_testset   $PY scripts/make_hpr_testset.py
run make_removal_sets  $PY scripts/make_removal_sets.py
SONN=data/raw/ScanObjectNN/h5_files/main_split/test_objectdataset.h5
if [[ -f "$SONN" ]]; then
  run prepare_scanobjectnn $PY scripts/prepare_scanobjectnn.py
else
  echo "ScanObjectNN not found ($SONN): ScanObjectNN and test-time adaptation are skipped"
fi
# Data visualisation
run visualize          $PY scripts/visualize.py
# Model testing
run test_supervised    $PY scripts/test_supervised.py
run test_zeroshot      $PY scripts/test_zeroshot.py
if [[ "${SKIP_CLIP:-0}" != "1" ]]; then
  run test_zeroshot_clip $PY scripts/test_zeroshot_clip.py
  run test_zeroshot_clip_removal $PY scripts/test_zeroshot_clip.py --sets removal
fi
# Test-time adaptation (ScanObjectNN, random point removal)
[[ -f "$SONN" ]] && run test_tta $PY scripts/test_tta.py
run make_plots         $PY scripts/make_plots.py
echo "all done -> outputs/"
