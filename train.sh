#!/usr/bin/env bash
# Train PointMLP on ModelNet40 in the background.
#
#   bash train.sh                                   # configs/pointmlp_modelnet40.yaml
#   bash train.sh configs/<config>.yaml
#   bash train.sh configs/pointmlp_modelnet40.yaml --epochs 50 --run_name short
#   RESUME=1 bash train.sh                          # resume from last.pth
#   GPU=1 bash train.sh
#
# Log:   tail -f outputs/training/<run_name>/train.log
# Stop:  kill $(cat outputs/training/<run_name>/train.pid)
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=${1:-configs/pointmlp_modelnet40.yaml}
shift || true
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}

# run name from --run_name or from the config
RUN=$(grep -E '^run_name:' "$CONFIG" | awk '{print $2}')
for ((i = 1; i <= $#; i++)); do
  if [[ "${!i}" == "--run_name" ]]; then j=$((i + 1)); RUN=${!j}; fi
done
OUT=outputs/training/$RUN
mkdir -p "$OUT"

EXTRA=()
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(--resume)

# create the training data if needed
[[ -f data/processed/modelnet40_train_2048.h5 ]] || $PY scripts/prepare_data.py
[[ -f data/processed/modelnet40_test_full_1024.h5 ]] || $PY scripts/make_hpr_testset.py

PYTHONUNBUFFERED=1 nohup $PY scripts/train_pointmlp.py --config "$CONFIG" "$@" "${EXTRA[@]}" \
  >> "$OUT/train.log" 2>&1 &
echo $! > "$OUT/train.pid"
echo "training started (pid $(cat "$OUT/train.pid")), GPU $CUDA_VISIBLE_DEVICES"
echo "  log    : tail -f $OUT/train.log"
echo "  curves : $OUT/curves.png   (updated every epoch)"
echo "  table  : $OUT/log.csv"
echo "  stop   : kill \$(cat $OUT/train.pid)"
