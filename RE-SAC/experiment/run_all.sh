#!/bin/bash
# Run all experiments: RE-SAC, SAC, TD3, DSAC across 4 MuJoCo envs × 3 seeds.
# Uses the LSTM-RL conda environment.
# Usage: bash run_all.sh [GPU_ID]
#
# Algo-aware concurrency:
#   RE-SAC (ensemble_size=10) → 2 concurrent (GPU-heavy)
#   SAC/TD3/DSAC (2 Q-nets)  → 8 concurrent (GPU-light)
# PID tracking via kill -0 for reliable process monitoring.

GPU=${1:-0}
SEEDS="0 1 2"
DSAC_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)/dsac"
EXPERIMENT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="conda run --no-capture-output -n LSTM-RL python"

ENVS=("hopper" "halfcheetah" "walker2d" "ant")
LOG_DIR="${EXPERIMENT_ROOT}/logs"
mkdir -p "$LOG_DIR"

declare -a ACTIVE_PIDS=()

# Remove finished PIDs from the tracking array
reap_pids() {
    local alive=()
    for pid in "${ACTIVE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        fi
    done
    ACTIVE_PIDS=("${alive[@]}")
}

# Block until active jobs drop below the given limit
wait_for_slot() {
    local max=$1
    while true; do
        reap_pids
        if [ "${#ACTIVE_PIDS[@]}" -lt "$max" ]; then
            break
        fi
        sleep 3
    done
}

# Wait for ALL active jobs to finish
wait_all() {
    while true; do
        reap_pids
        if [ "${#ACTIVE_PIDS[@]}" -eq 0 ]; then
            break
        fi
        sleep 3
    done
}

# Launch one experiment in background, track its PID
launch() {
    local algo=$1 script=$2 config=$3 seed=$4 max_jobs=$5
    COUNT=$((COUNT + 1))
    local log_file="${LOG_DIR}/${algo}_seed${seed}.log"
    wait_for_slot "$max_jobs"
    echo "[${COUNT}/${TOTAL}] ${algo} seed=${seed} (active: ${#ACTIVE_PIDS[@]}/${max_jobs})"
    $PYTHON "$script" \
        --config "$config" \
        --gpu "$GPU" \
        --seed "$seed" \
        > "$log_file" 2>&1 &
    ACTIVE_PIDS+=($!)
}

TOTAL=48  # 4 algos × 4 envs × 3 seeds
COUNT=0

echo "============================================="
echo "  MuJoCo Benchmark Experiments (LSTM-RL env)"
echo "  GPU: ${GPU}  |  Seeds: ${SEEDS}"
echo "  RE-SAC: max 2 concurrent (GPU-heavy)"
echo "  SAC/TD3/DSAC: max 8 concurrent (GPU-light)"
echo "============================================="

# # --- RE-SAC (GPU-heavy: ensemble_size=10, max 2 concurrent) ---
# echo ""
# echo ">>> RE-SAC experiments (max 2 concurrent)..."
# for env in "${ENVS[@]}"; do
#   for seed in $SEEDS; do
#     launch "resac_${env}" \
#       "${EXPERIMENT_ROOT}/resac.py" \
#       "${EXPERIMENT_ROOT}/configs/resac/${env}.yaml" \
#       "$seed" 2
#   done
# done

# # Wait for all RE-SAC to finish before starting lighter baselines
# echo ">>> Waiting for RE-SAC to complete..."
# wait_all

# --- SAC baseline (GPU-light, max 8 concurrent) ---
echo ""
echo ">>> SAC baseline (max 8 concurrent)..."
for env in "${ENVS[@]}"; do
  for seed in $SEEDS; do
    launch "sac_${env}" \
      "${EXPERIMENT_ROOT}/run_sac.py" \
      "${DSAC_ROOT}/configs/sac-normal/${env}.yaml" \
      "$seed" 8
  done
done

# --- TD3 baseline ---
echo ""
echo ">>> TD3 baseline (max 8 concurrent)..."
for env in "${ENVS[@]}"; do
  for seed in $SEEDS; do
    launch "td3_${env}" \
      "${EXPERIMENT_ROOT}/run_td3.py" \
      "${DSAC_ROOT}/configs/td3-normal/${env}.yaml" \
      "$seed" 8
  done
done

# --- DSAC baseline ---
echo ""
echo ">>> DSAC baseline (max 8 concurrent)..."
for env in "${ENVS[@]}"; do
  for seed in $SEEDS; do
    launch "dsac_${env}" \
      "${EXPERIMENT_ROOT}/run_dsac.py" \
      "${DSAC_ROOT}/configs/dsac-normal-iqn-neutral/${env}.yaml" \
      "$seed" 8
  done
done

echo ""
echo "All ${TOTAL} experiments queued. Waiting for completion..."
wait_all
echo "============================================="
echo "  All experiments finished!"
echo "============================================="
