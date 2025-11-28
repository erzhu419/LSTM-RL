"""
Evaluate May 2024 Ensemble Models
Load and evaluate the ensemble models from May 1-2, 2024
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from env.sim import env_bus

# Model configurations from May 2024
configs = [
    {
        'name': 'buffer_100k_ratio_2_alpha_0.3_wreg_0.01',
        'path': 'model/sac_v2_bus_ensemble/replay_buffer_size_100000/critic_actor_ratio_2/maximum_alpha_0.3/weight_reg_0.01'
    },
    {
        'name': 'buffer_100k_ratio_3_alpha_0.3_wreg_0.01',
        'path': 'model/sac_v2_bus_ensemble/replay_buffer_size_100000/critic_actor_ratio_3/maximum_alpha_0.3/weight_reg_0.01'
    },
    {
        'name': 'buffer_100k_ratio_4_alpha_0.3_wreg_0.01',
        'path': 'model/sac_v2_bus_ensemble/replay_buffer_size_100000/critic_actor_ratio_4/maximum_alpha_0.3/weight_reg_0.01'
    },
]

# Get available episodes for each config
for config in configs:
    full_path = os.path.join('/home/erzhu419/mine_code/LSTM-RL', config['path'])
    if os.path.exists(full_path):
        files = os.listdir(full_path)
        # Extract episode numbers
        episodes = []
        for f in files:
            if 'weight_reg' in f:
                try:
                    ep = int(f.split()[-1])
                    episodes.append(ep)
                except:
                    pass
        config['episodes'] = sorted(episodes)
        print(f"{config['name']}: {len(episodes)} checkpoints (episodes {min(episodes) if episodes else 'N/A'} - {max(episodes) if episodes else 'N/A'})")
    else:
        print(f"{config['name']}: Path not found")
        config['episodes'] = []

print("\nThese are the ensemble models from May 1-2, 2024")
print("To evaluate them, you need to:")
print("1. Load the original sac_ensemble_original.py trainer")
print("2. Load these model checkpoints")
print("3. Run evaluation on the environment")
print("\nNote: The reward curve data from that time may have been overwritten.")
print("You may need to re-run evaluation to see the performance.")
