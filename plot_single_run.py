import numpy as np
import matplotlib.pyplot as plt
import os

file_path = "/home/erzhu419/mine_code/sumo-rl/LSTM-RL/logs/sac_v2_bus_original_sigma1p5_embed-full_wreg0p0_gpt_version/rewards.npy"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

try:
    rewards = np.load(file_path)
    print(f"Loaded rewards with shape: {rewards.shape}")
except Exception as e:
    print(f"Error loading numpy file: {e}")
    exit(1)

plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title("Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
output_path = "single_reward_plot.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
