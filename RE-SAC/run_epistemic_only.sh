#!/bin/bash

# Experiment B: Epistemic Only
# Settings: 
#   weight_reg: 0 (Disabled Parameter Reg)
#   beta: -2 (Enabled Policy LCB)
#   beta_bc: 0 (Disabled Imitation)
#   beta_ood: 0.01 (Enabled Critic Consensus)

export CUDA_VISIBLE_DEVICES=0

python RE-SAC/sac_ensemble_original_logging.py \
    --weight_reg 0 \
    --beta -2 \
    --beta_bc 0 \
    --beta_ood 0.01 \
    --maximum_alpha 0.6 \
    --critic_actor_ratio 2 \
    --replay_buffer_size 1000000 \
    --hidden_dim 64 \
    --max_episodes 500 \
    --save_root RE-SAC/Epistemic_Only \
    --run_name .
