#!/bin/bash

# Configuration
ENSEMBLE_DIR="ensemble_10/model"
SAC_DIR="../model/sac_v2_bus"
INTERVAL=10
HIDDEN_DIM_ENS=64
HIDDEN_DIM_SAC=32
ENSEMBLE_SIZE=10

echo "Starting Q-Accuracy Analysis (Fixed Normalization)..."

# 1. Analyze Ensemble
echo "Analyzing Ensemble Models..."
python q_accuracy_analysis/analyze_q_accuracy.py \
    --model_type ensemble \
    --checkpoint_dir $ENSEMBLE_DIR \
    --output_file q_comparasion/ensemble_q_accuracy_norm.npy \
    --hidden_dim $HIDDEN_DIM_ENS \
    --ensemble_size $ENSEMBLE_SIZE \
    --step_interval $INTERVAL \
    --num_eval_episodes 1 \
    --samples_per_episode 5

# 2. Analyze SAC
echo "Analyzing SAC Models..."
python q_accuracy_analysis/analyze_q_accuracy.py \
    --model_type sac \
    --checkpoint_dir $SAC_DIR \
    --output_file q_comparasion/sac_q_accuracy_norm.npy \
    --hidden_dim $HIDDEN_DIM_SAC \
    --step_interval $INTERVAL \
    --num_eval_episodes 1 \
    --samples_per_episode 5

# 3. Plot Results
echo "Plotting Results..."
python q_accuracy_analysis/plot_q_accuracy_corrected.py \
    --sac_file q_comparasion/sac_q_accuracy_norm.npy \
    --ensemble_file q_comparasion/ensemble_q_accuracy_norm.npy \
    --log_dir ensemble_10/logs \
    --output_file q_comparasion/q_accuracy_comparison_corrected.png

echo "Analysis Complete. Check q_comparasion/q_accuracy_comparison_corrected.png"
