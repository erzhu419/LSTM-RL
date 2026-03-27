import numpy as np
import os

rewards_path = 'RE-SAC/model/original_plus_sign_logging_run/rewards.npy'

if os.path.exists(rewards_path):
    try:
        rewards = np.load(rewards_path)
        print(f"Total episodes recorded: {len(rewards)}")
        print("Last 10 rewards:")
        for i in range(0, len(rewards)):
            print(f"Episode {i}: {rewards[i]}")
    except Exception as e:
        print(f"Error loading rewards: {e}")
else:
    print(f"File not found: {rewards_path}")
