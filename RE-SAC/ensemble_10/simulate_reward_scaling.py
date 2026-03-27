import numpy as np
import matplotlib.pyplot as plt
import os

# Output directory
output_dir = '/home/erzhu419/mine_code/LSTM-RL/RE-SAC/ensemble_10/pic'
os.makedirs(output_dir, exist_ok=True)

# 1. Simulate Headways
# Assume headways follow a distribution. 
# In a controlled env, they cluster around 360, but with variance.
# Let's simulate a batch of data (e.g., batch_size=2048 as in the script)
batch_size = 2048
np.random.seed(42)

# Mixture of Gaussians to simulate some "good" control and some "bad" scenarios
# 70% close to target (360 +/- 60), 30% chaotic (uniform 100 to 800)
good_headways = np.random.normal(loc=360, scale=60, size=int(0.7 * batch_size))
bad_headways = np.random.uniform(low=100, high=800, size=batch_size - len(good_headways))
headways = np.concatenate([good_headways, bad_headways])

# 2. Calculate Raw Rewards
# Logic: -abs(headway - 360)
# Simplified version of the env logic (ignoring forward/backward weighting for this demo, same distribution)
target = 360.0
raw_rewards = -np.abs(headways - target)

# Apply the large deviation penalty found in env (-20 if deviation > 180)
# deviation > 180 means headway < 180 or headway > 540
penalty_mask = np.abs(headways - target) > 180
raw_rewards[penalty_mask] -= 20.0

# 3. Apply Reward Scaling (Batch Normalization)
# Logic from code: reward = reward_scale * (reward - mean) / (std + 1e-6)
reward_scale = 10.0
mean = raw_rewards.mean()
std = raw_rewards.std()

scaled_rewards = reward_scale * (raw_rewards - mean) / (std + 1e-6)

# 4. Plotting
plt.figure(figsize=(12, 6))

# Plot Raw Rewards
plt.subplot(1, 2, 1)
plt.hist(raw_rewards, bins=50, color='salmon', alpha=0.7, edgecolor='black')
plt.title(f'Raw Rewards Distribution\nMean: {mean:.2f}, Std: {std:.2f}')
plt.xlabel('Raw Reward value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Plot Scaled Rewards
plt.subplot(1, 2, 2)
plt.hist(scaled_rewards, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
plt.title(f'Scaled Rewards Distribution (Scale={reward_scale})\nMean: {scaled_rewards.mean():.2f}, Std: {scaled_rewards.std():.2f}')
plt.xlabel('Scaled Reward value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.axvline(0, color='red', linestyle='--', linewidth=1, label='Zero')
plt.legend()

plt.tight_layout()
save_path = os.path.join(output_dir, 'reward_scaling_simulation.png')
plt.savefig(save_path, dpi=300)
print(f"Plot saved to: {save_path}")

# Print sample statistics
print(f"Raw Reward Range: [{raw_rewards.min():.2f}, {raw_rewards.max():.2f}]")
print(f"Scaled Reward Range: [{scaled_rewards.min():.2f}, {scaled_rewards.max():.2f}]")
print(f"Percentage of Scaled Rewards > 0: {(scaled_rewards > 0).mean() * 100:.2f}%")
