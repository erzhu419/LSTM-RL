#!/bin/bash

# Retrain Ensemble 10 on the CLEAN environment (Sticky Reward & KeyError Fixed)
# Output Dir: ensemble_10_clean

ENSEMBLE_SIZE=10
OUTPUT_DIR="ensemble_10_clean"

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/model
mkdir -p $OUTPUT_DIR/pic

echo "Starting Retraining of Ensemble (Size 10) on Clean Environment..."
echo "Logs will be saved to $OUTPUT_DIR"

python sac_ensemble_original_logging.py \
    --weight_reg 0.01 \
    --maximum_alpha 0.6 \
    --critic_actor_ratio 2 \
    --replay_buffer_size 1000000 \
    --hidden_dim 64 \
    --max_episodes 500 \
    --save_root $OUTPUT_DIR \
    --run_name clean_run \
    --ensemble_size $ENSEMBLE_SIZE

echo "Retraining Complete."
