#!/bin/bash

MAX_PARALLEL_JOBS=16
WEIGHT_REG_VALUES="0 1e-5 1e-4 1e-3 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1"

# 定义清理函数
cleanup() {
    echo "Killing all running experiments..."
    pkill -f "python sac_bus.py"
    exit 1
}

# 监听 Ctrl + C
trap cleanup SIGINT

running_jobs=0

for weight_reg in $WEIGHT_REG_VALUES; do
    echo "Running SAC with weight_reg=$weight_reg, use_reward_scaling=True, auto_entropy=True, maximum_alpha=0.3"
    
    python sac_bus.py --weight_reg=$weight_reg --use_reward_scaling=True --auto_entropy=True --maximum_alpha=0.3 > logs/weight_reg_${weight_reg}.log 2>&1 &

    ((running_jobs++))

    if [[ $running_jobs -ge $MAX_PARALLEL_JOBS ]]; then
        wait -n  
        ((running_jobs--))
    fi

done

wait
echo "All experiments finished!"
