#!/bin/bash

MAX_PARALLEL_JOBS=8

WEIGHT_REG_VALUES=("0.01" "0.03" "0.05" "0.07" "0.09")
MAX_ALPHA_VALUES=("0.3" "0.6" "0.9" "1.2" "1.5" "1.8" "2.1")
CRITIC_ACTOR_RATIOS=("2" "4" "6" "8" "10")
REPLAY_BUFFER_SIZES=("10000" "50000" "100000")

# 定义清理函数
cleanup() {
    echo "Killing all running experiments..."
    pkill -f "python sac_v2_bus_ensemble.py"
    exit 1
}

# 监听 Ctrl + C
trap cleanup SIGINT

running_jobs=0

for weight_reg in "${WEIGHT_REG_VALUES[@]}"; do
  for alpha in "${MAX_ALPHA_VALUES[@]}"; do
    for ratio in "${CRITIC_ACTOR_RATIOS[@]}"; do
      for buffer_size in "${REPLAY_BUFFER_SIZES[@]}"; do

        log_path="logs/wr${weight_reg}_a${alpha}_r${ratio}_buf${buffer_size}.log"
        echo "Running: weight_reg=${weight_reg}, max_alpha=${alpha}, ratio=${ratio}, buffer=${buffer_size}"

        python sac_v2_bus_ensemble.py \
          --weight_reg=$weight_reg \
          --maximum_alpha=$alpha \
          --critic_actor_ratio=$ratio \
          --replay_buffer_size=$buffer_size \
          > "$log_path" 2>&1 &

        ((running_jobs++))
        if [[ $running_jobs -ge $MAX_PARALLEL_JOBS ]]; then
          wait -n  # wait for at least one to finish
          ((running_jobs--))
        fi

      done
    done
  done
done

wait
echo "All experiments finished!"
