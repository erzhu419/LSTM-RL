#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Experiment: Test Sign Before RegNorm (Manual Python Code Modification)
# Settings: 
#   Standard best params
#   Ensemble Size: 10
#   Output: RE-SAC/test_sign_before_regnorm

python RE-SAC/sac_ensemble_original_logging.py \
    --weight_reg 0.01 \
    --maximum_alpha 0.6 \
    --critic_actor_ratio 2 \
    --replay_buffer_size 1000000 \
    --hidden_dim 64 \
    --max_episodes 500 \
    --save_root RE-SAC/test_sign_before_regnorm \
    --run_name . \
    --ensemble_size 10
