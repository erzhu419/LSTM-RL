#!/bin/bash
# Run SAC + TD3 + DSAC baselines only.
# Usage: bash run_baselines.sh [GPU]

set -e

GPU=${1:-0}
SEEDS="0 1 2 3 4"
DSAC_ROOT="$(cd "$(dirname "$0")/../../../dsac" && pwd)"
EXPERIMENT_ROOT="$(cd "$(dirname "$0")" && pwd)"
ENVS=("hopper" "halfcheetah" "walker2d" "ant")

LOG_DIR="${EXPERIMENT_ROOT}/logs"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "  Baseline Experiments (SAC, TD3, DSAC)"
echo "  GPU: ${GPU}  |  Seeds: ${SEEDS}"
echo "  DSAC root: ${DSAC_ROOT}"
echo "============================================="

# --- SAC ---
echo ""
echo ">>> SAC..."
for env in "${ENVS[@]}"; do
  for seed in $SEEDS; do
    log_file="${LOG_DIR}/sac_${env}_seed${seed}.log"
    echo "  [SAC] ${env} seed=${seed}"
    cd "$DSAC_ROOT"
    python sac.py \
      --config "configs/sac-normal/${env}.yaml" \
      --gpu "$GPU" \
      --seed "$seed" \
      > "$log_file" 2>&1 &
  done
done

# --- TD3 ---
echo ""
echo ">>> TD3..."
for env in "${ENVS[@]}"; do
  for seed in $SEEDS; do
    log_file="${LOG_DIR}/td3_${env}_seed${seed}.log"
    echo "  [TD3] ${env} seed=${seed}"
    cd "$DSAC_ROOT"
    python td3.py \
      --config "configs/td3-normal/${env}.yaml" \
      --gpu "$GPU" \
      --seed "$seed" \
      > "$log_file" 2>&1 &
  done
done

# --- DSAC ---
echo ""
echo ">>> DSAC..."
for env in "${ENVS[@]}"; do
  for seed in $SEEDS; do
    log_file="${LOG_DIR}/dsac_${env}_seed${seed}.log"
    echo "  [DSAC] ${env} seed=${seed}"
    cd "$DSAC_ROOT"
    python dsac.py \
      --config "configs/dsac-normal-iqn-neutral/${env}.yaml" \
      --gpu "$GPU" \
      --seed "$seed" \
      > "$log_file" 2>&1 &
  done
done

echo ""
echo "All baselines launched. Wait with: wait"
wait
echo "All baseline experiments finished!"
