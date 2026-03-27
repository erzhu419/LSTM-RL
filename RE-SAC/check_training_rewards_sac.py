import numpy as np
import os

log_path = 'logs/sac_v2_bus/rewards.npy'
if os.path.exists(log_path):
    rewards = np.load(log_path)
    print(f"Loaded {len(rewards)} episodes.")
    print(f"Mean Reward: {np.mean(rewards)}")
    print(f"Max Reward: {np.max(rewards)}")
    print(f"Min Reward: {np.min(rewards)}")
    print("First 10 rewards:", rewards[:10])
else:
    print(f"File not found: {log_path}")
