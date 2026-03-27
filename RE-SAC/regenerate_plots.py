
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
ENSEMBLE_SIZE = 10
LOG_DIR = f'RE-SAC/ensemble_{ENSEMBLE_SIZE}/logs'
PIC_DIR = f'RE-SAC/ensemble_{ENSEMBLE_SIZE}/pic'

os.makedirs(PIC_DIR, exist_ok=True)

def safe_load(filename):
    path = os.path.join(LOG_DIR, filename)
    if os.path.exists(path):
        return np.load(path)
    print(f"Warning: {filename} not found.")
    return None

# Load Data
rewards = safe_load('rewards.npy')
q_values = safe_load('q_values.npy') # Note: file found was q_values_episode.npy in earlier list? Let's check.
# Step 1302 showed: q_values_episode.npy, rewards.npy, etc.
# Wait, check strict names.
# rewards.npy
# q_values_episode.npy
# reg_norms1_episode.npy
# reg_norms2_episode.npy
# log_probs_episode.npy
# alpha_values_episode.npy
# ood_losses_episode.npy
# q_stds_episode.npy

# Correcting load names
rewards = safe_load('rewards.npy')
q_values_episode = safe_load('q_values_episode.npy')
reg_norms1 = safe_load('reg_norms1_episode.npy')
reg_norms2 = safe_load('reg_norms2_episode.npy')
log_probs = safe_load('log_probs_episode.npy')
alpha_values = safe_load('alpha_values_episode.npy')
ood_losses = safe_load('ood_losses_episode.npy')
q_stds = safe_load('q_stds_episode.npy')

def plot_metric(data, name, ylabel):
    if data is None: return
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(f'{name} (Ensemble {ENSEMBLE_SIZE})')
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(PIC_DIR, f'{name}.png'))
    plt.close()
    print(f"Saved {name}.png")

# Plotting
plot_metric(rewards, 'rewards', 'Reward')
plot_metric(q_values_episode, 'q_values', 'Q Value')
plot_metric(reg_norms1, 'reg_norms1', 'Reg Norm 1')
plot_metric(reg_norms2, 'reg_norms2', 'Reg Norm 2')
plot_metric(log_probs, 'log_probs', 'Log Prob')
plot_metric(alpha_values, 'alpha_values', 'Alpha')
plot_metric(ood_losses, 'ood_losses', 'OOD Loss (Std)')
plot_metric(q_stds, 'q_stds', 'Q Std')

print(f"Done! Plots saved to {PIC_DIR}")
