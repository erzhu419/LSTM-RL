#!/bin/bash

# Experiment A: Aleatoric Only
# Settings: 
#   weight_reg: 0.01 (Enabled)
#   beta: 0 (Disabled Policy LCB)
#   beta_bc: 0 (Disabled Imitation)
#   beta_ood: 0 (Disabled Critic Consensus)

export CUDA_VISIBLE_DEVICES=0

python ensemble_paper/sac_ensemble_original_logging.py \
    --weight_reg 0.01 \
    --beta 0 \
    --beta_bc 0 \
    --beta_ood 0 \
    --maximum_alpha 0.6 \
    --critic_actor_ratio 2 \
    --replay_buffer_size 1000000 \
    --hidden_dim 64 \
    --max_episodes 500 \
    --save_root ensemble_paper/Aleatoric_Only \
    --run_name .
