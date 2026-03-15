#!/bin/bash

# Run the ensemble training with the best parameters identified from historical analysis
# Best Params:
#   weight_reg: 0.01
#   maximum_alpha: 0.6
#   critic_actor_ratio: 2
#   replay_buffer_size: 1000000
#   hidden_dim: 64
#   ensemble_size: 10 (default, confirmed)
# Environment: ensemble_paper/env_original


# Default ensemble size is 10, can be overridden by first argument
ENSEMBLE_SIZE=${1:-10}

python ensemble_paper/sac_ensemble_original_logging.py \
    --weight_reg 0.01 \
    --maximum_alpha 0.6 \
    --critic_actor_ratio 2 \
    --replay_buffer_size 1000000 \
    --hidden_dim 64 \
    --max_episodes 500 \
    --save_root ensemble_paper/ensemble_${ENSEMBLE_SIZE} \
    --run_name . \
    --ensemble_size ${ENSEMBLE_SIZE}
