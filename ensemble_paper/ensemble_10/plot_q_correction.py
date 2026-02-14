import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
base_dir = '/home/erzhu419/mine_code/LSTM-RL/ensemble_paper/ensemble_10'
log_dir = os.path.join(base_dir, 'logs')
pic_dir = os.path.join(base_dir, 'pic')
os.makedirs(pic_dir, exist_ok=True)

# Load data
try:
    q_values = np.load(os.path.join(log_dir, 'q_values_episode.npy'))
    reg_norms1 = np.load(os.path.join(log_dir, 'reg_norms1_episode.npy'))
    reg_norms2 = np.load(os.path.join(log_dir, 'reg_norms2_episode.npy'))
except FileNotFoundError as e:
    print(f"Error loading logs: {e}")
    exit(1)

# Parameters
gamma = 0.99

# Compute correction
# reg_norms logs store 'args.weight_reg * reg_norm', which is the additive term per step.
# We average the two sets of critics for robustness.
avg_reg_term = (reg_norms1 + reg_norms2) / 2.0

# Q_bias = C / (1 - gamma)
q_bias = avg_reg_term / (1 - gamma)

# Q_corrected = Q_original - Q_bias
q_corrected = q_values - q_bias

# Plotting
plt.figure(figsize=(10, 6))

episodes = np.arange(len(q_values))

plt.plot(episodes, q_values, label='Ensemble Q (Original)', color='blue', alpha=0.7)
plt.plot(episodes, q_corrected, label='Ensemble Q (Corrected)', color='green', linewidth=2)
# plt.plot(episodes, -q_bias, label='Negative Bias Term', color='red', linestyle='--', alpha=0.5)

plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

plt.title('Q-Value Analysis: Removing the Regularization Bias')
plt.xlabel('Episode')
plt.ylabel('Q-Value')
plt.legend()
plt.grid(True, alpha=0.3)

output_path = os.path.join(pic_dir, 'q_value_correction.png')
plt.savefig(output_path, dpi=300)
print(f"Plot saved to: {output_path}")

# Print stats
print(f"Original Q Mean (Last 100): {q_values[-100:].mean():.2f}")
print(f"Corrected Q Mean (Last 100): {q_corrected[-100:].mean():.2f}")
print(f"Mean Bias Adjustment: {q_bias.mean():.2f}")
