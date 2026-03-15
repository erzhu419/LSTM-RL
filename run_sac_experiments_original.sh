#!/bin/bash

# 降低并行数以减少内存压力 (原来16个太多)
MAX_PARALLEL_JOBS=8

WEIGHT_REG_VALUES=("0.01" "0.03" "0.05" "0.07" "0.09")
MAX_ALPHA_VALUES=("0.3" "0.6" "0.9" "1.2" "1.5" "1.8" "2.1")
CRITIC_ACTOR_RATIOS=("2" "4" "6" "8" "10")
REPLAY_BUFFER_SIZES=("10000" "50000" "100000")
ENSEMBLE_SIZES=("5" "10")

LOG_DIR="logs_original"
mkdir -p "$LOG_DIR"

cleanup() {
    echo "Killing all running original ensemble experiments..."
    pkill -f "python sac_ensemble_original.py"
    exit 1
}

trap cleanup SIGINT

running_jobs=0

for weight_reg in "${WEIGHT_REG_VALUES[@]}"; do
  for alpha in "${MAX_ALPHA_VALUES[@]}"; do
    for ratio in "${CRITIC_ACTOR_RATIOS[@]}"; do
      for buffer_size in "${REPLAY_BUFFER_SIZES[@]}"; do
        for ens_size in "${ENSEMBLE_SIZES[@]}"; do

          log_path="$LOG_DIR/wr${weight_reg}_a${alpha}_r${ratio}_buf${buffer_size}_ens${ens_size}.log"
          echo "Running original ensemble: weight_reg=${weight_reg}, max_alpha=${alpha}, ratio=${ratio}, buffer=${buffer_size}, ensemble=${ens_size}"

          python sac_ensemble_original.py \
            --weight_reg=$weight_reg \
            --maximum_alpha=$alpha \
            --critic_actor_ratio=$ratio \
            --replay_buffer_size=$buffer_size \
            --ensemble_size=$ens_size \
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
echo "All original ensemble experiments finished!"
