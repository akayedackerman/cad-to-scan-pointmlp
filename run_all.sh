#!/usr/bin/env bash
# Data preparation, visualisation, testing and plots.
#
#   bash run_all.sh
#   SKIP_CLIP=1 bash run_all.sh                              # without CLIP
#   DATA_CONFIG=my_data.yaml TEST_CONFIG=my_test.yaml bash run_all.sh
#
# Settings: configs/data.yaml (data preparation) and configs/test.yaml (testing).
# Every script prints its configuration and writes a timestamped log to outputs/logs/<script>.log.
# Training: bash train.sh (configs/train.yaml)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
DATA=${DATA_CONFIG:-configs/data.yaml}
TEST=${TEST_CONFIG:-configs/test.yaml}
export PYTHONUNBUFFERED=1
mkdir -p outputs/logs

run() {  # run <title> <command>
  echo "================ $1 ================"
  shift
  "$@" 2>&1 | grep --line-buffered -v -i "warning"
}

# Point-cloud data preparation
run "data: ModelNet point clouds"    $PY scripts/prepare_data.py      --config "$DATA"
run "data: HPR test sets"            $PY scripts/make_hpr_testset.py  --config "$DATA"
run "data: random point removal"     $PY scripts/make_removal_sets.py --config "$DATA"
SONN=data/raw/ScanObjectNN/h5_files/main_split/test_objectdataset.h5
if [[ -f "$SONN" ]]; then
  run "data: ScanObjectNN"           $PY scripts/prepare_scanobjectnn.py --config "$DATA"
else
  echo "ScanObjectNN not found ($SONN): ScanObjectNN and test-time adaptation are skipped"
fi
# Data visualisation
run "visualisation"                  $PY scripts/visualize.py
# Model testing
run "test: supervised"               $PY scripts/test_supervised.py --config "$TEST"
run "test: zero-shot PointMLP"       $PY scripts/test_zeroshot.py   --config "$TEST"
if [[ "${SKIP_CLIP:-0}" != "1" ]]; then
  run "test: zero-shot CLIP"         $PY scripts/test_zeroshot_clip.py --config "$TEST"
  run "test: zero-shot CLIP, removal" $PY scripts/test_zeroshot_clip.py --config "$TEST" --sets removal
fi
# Test-time adaptation (ModelNet40 -> ScanObjectNN)
[[ -f "$SONN" ]] && run "test: test-time adaptation" $PY scripts/test_tta.py --config "$TEST"
run "plots"                          $PY scripts/make_plots.py
echo "all done -> outputs/   (logs: outputs/logs/)"
