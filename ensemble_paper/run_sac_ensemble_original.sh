#!/bin/bash
# 运行脚本：适用于8e1c7c6版本的sac_ensemble_original.py
# 注意：该版本的ensemble_size硬编码为10，不支持命令行参数

MAX_PARALLEL_JOBS=10

# 8e1c7c6版本支持的参数
WEIGHT_REG_VALUES=("0.01" "0.03" "0.05" "0.07" "0.09")
MAX_ALPHA_VALUES=("0.3" "0.6" "0.9" "1.2" "1.5" "1.8" "2.1")
CRITIC_ACTOR_RATIOS=("2" "4" "6" "8" "10")
REPLAY_BUFFER_SIZES=("100000" "1000000")
TRAIN_SIGMAS=("1.0" "1.5" "2.0")
EVAL_SIGMAS=("1.0" "1.5" "2.0")

LOG_DIR="logs_ensemble_original"
mkdir -p "$LOG_DIR"

cleanup() {
    echo "Killing all running original ensemble experiments..."
    pkill -f "python sac_ensemble_original.py"
    exit 1
}

trap cleanup SIGINT

running_jobs=0

echo "=========================================="
echo "Running SAC Ensemble Original (8e1c7c6)"
echo "Note: ensemble_size is hardcoded to 10"
echo "=========================================="
echo ""

for train_sigma in "${TRAIN_SIGMAS[@]}"; do
  train_sigma_label=${train_sigma//./p}
  for weight_reg in "${WEIGHT_REG_VALUES[@]}"; do
    for alpha in "${MAX_ALPHA_VALUES[@]}"; do
      for ratio in "${CRITIC_ACTOR_RATIOS[@]}"; do
        for buffer_size in "${REPLAY_BUFFER_SIZES[@]}"; do

          log_path="$LOG_DIR/wr${weight_reg}_a${alpha}_r${ratio}_buf${buffer_size}_sigma${train_sigma_label}.log"
          echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: weight_reg=${weight_reg}, max_alpha=${alpha}, ratio=${ratio}, buffer=${buffer_size}, train_sigma=${train_sigma}, eval_sigmas=${EVAL_SIGMAS[*]}"

          python sac_ensemble_original.py \
            --weight_reg=$weight_reg \
            --maximum_alpha=$alpha \
            --critic_actor_ratio=$ratio \
            --replay_buffer_size=$buffer_size \
            --route_sigma=$train_sigma \
            --eval_sigmas ${EVAL_SIGMAS[@]} \
            --max_episodes=500 \
            --train \
            > "$log_path" 2>&1 &

          ((running_jobs++))
          if [[ $running_jobs -ge $MAX_PARALLEL_JOBS ]]; then
            wait -n
            ((running_jobs--))
          fi

        done
      done
    done
  done
done

wait
echo ""
echo "=========================================="
echo "All original ensemble experiments finished!"
echo "Logs saved to: $LOG_DIR/"
echo "Models saved to: ./model/sac_ensemble_original/"
echo "=========================================="
