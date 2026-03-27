#!/bin/bash
# Run RE-SAC experiments only.
# Usage: bash run_resac.sh [GPU] [ENSEMBLE_SIZE]
#   GPU: GPU id (default 0, -1 for CPU)
#   ENSEMBLE_SIZE: override ensemble_size in config (optional)

set -e

GPU=${1:-0}
ENSEMBLE_SIZE=${2:-}
SEEDS="0 1 2 3 4"
EXPERIMENT_ROOT="$(cd "$(dirname "$0")" && pwd)"
ENVS=("hopper" "halfcheetah" "walker2d" "ant")

LOG_DIR="${EXPERIMENT_ROOT}/logs"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "  RE-SAC Experiments"
echo "  GPU: ${GPU}  |  Seeds: ${SEEDS}"
if [ -n "$ENSEMBLE_SIZE" ]; then
  echo "  Ensemble Size Override: ${ENSEMBLE_SIZE}"
fi
echo "============================================="

for env in "${ENVS[@]}"; do
  for seed in $SEEDS; do
    suffix=""
    if [ -n "$ENSEMBLE_SIZE" ]; then
      suffix="_e${ENSEMBLE_SIZE}"
    fi
    log_file="${LOG_DIR}/resac_${env}${suffix}_seed${seed}.log"
    echo "  [RE-SAC] env=${env} seed=${seed} -> ${log_file}"

    cd "$EXPERIMENT_ROOT"
    python resac.py \
      --config "configs/resac/${env}.yaml" \
      --gpu "$GPU" \
      --seed "$seed" \
      > "$log_file" 2>&1 &
  done
done

echo ""
echo "All RE-SAC experiments launched. Wait with: wait"
wait
echo "All RE-SAC experiments finished!"
