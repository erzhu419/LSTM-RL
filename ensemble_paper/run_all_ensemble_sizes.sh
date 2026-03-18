#!/bin/bash

# Run experiments for Ensemble Sizes [2, 5, 20, 40] in parallel

echo "Starting Ensemble Size Experiments..."

# Size 2
echo "Launching Ensemble Size = 2"
bash ensemble_paper/run_best_ensemble_logging.sh 2 &

# Size 5
echo "Launching Ensemble Size = 5"
bash ensemble_paper/run_best_ensemble_logging.sh 5 &

# Size 20
echo "Launching Ensemble Size = 20"
bash ensemble_paper/run_best_ensemble_logging.sh 20 &

# Size 40
echo "Launching Ensemble Size = 40"
bash ensemble_paper/run_best_ensemble_logging.sh 40 &

# Experiment: Aleatoric Only
echo "Launching Aleatoric Only Experiment"
bash ensemble_paper/run_aleatoric_only.sh &

# Experiment: Epistemic Only
echo "Launching Epistemic Only Experiment"
bash ensemble_paper/run_epistemic_only.sh &

# Wait for all background jobs to finish
echo "All experiments launched! Waiting for completion..."
wait

echo "All Ensemble Size experiments finished."
