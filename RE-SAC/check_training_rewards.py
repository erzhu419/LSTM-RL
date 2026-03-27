import numpy as np
import os

log_path = 'ensemble_10/logs/rewards.npy'
if os.path.exists(log_path):
    rewards = np.load(log_path)
    print(f"Loaded {len(rewards)} episodes.")
    print(f"Mean Reward: {np.mean(rewards)}")
    print(f"Std Reward: {np.std(rewards)}")
    print(f"Projected Normalized Score of -50: (-50 - {np.mean(rewards)}) / {np.std(rewards)}")
    print(f"Max Reward: {np.max(rewards)}")
    print(f"Min Reward: {np.min(rewards)}")
    print("First 10 rewards:", rewards[:10])
    print("Last 10 rewards:", rewards[-10:])
else:
    print(f"File not found: {log_path}")
