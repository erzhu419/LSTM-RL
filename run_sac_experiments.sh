#!/bin/bash

MAX_PARALLEL_JOBS=4
WEIGHT_REG_VALUES=$(python -c "print(' '.join([str(x) for x in [0.0001 * (10 ** i) for i in range(5)] + [j for j in range(1, 11)]]))")

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
    echo "Running SAC with weight_reg=$weight_reg"
    
    python sac_bus.py --weight_reg=$weight_reg > logs/weight_reg_${weight_reg}.log 2>&1 &

    ((running_jobs++))

    if [[ $running_jobs -ge $MAX_PARALLEL_JOBS ]]; then
        wait -n  
        ((running_jobs--))
    fi
done

wait
echo "All experiments finished!"

