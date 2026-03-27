import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
baseline_dir = "model/sac_v2_bus"
ensemble_dir = "RE-SAC/ensemble_10/logs"

# Output file
output_path = "RE-SAC/ensemble_vs_baseline_final_uq.png"

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)

# 1. Load Rewards
try:
    baseline_rewards = np.load(os.path.join(baseline_dir, "rewards.npy"))
    ensemble_rewards = np.load(os.path.join(ensemble_dir, "rewards.npy"))
    print(f"Baseline rewards shape: {baseline_rewards.shape}")
    print(f"Ensemble rewards shape: {ensemble_rewards.shape}")
except Exception as e:
    print(f"Error loading rewards: {e}")
    exit(1)

# 2. Load Q-Values and Stds
try:
    # Baseline Q might be (N,) or (N, 2)
    baseline_q = np.load(os.path.join(baseline_dir, "q_values.npy"))
    
    # Ensemble Q is (N,) representing the mean
    ensemble_q_mean = np.load(os.path.join(ensemble_dir, "q_values_episode.npy"))
    # Ensemble Std is (N,) representing the std across ensemble
    ensemble_q_std = np.load(os.path.join(ensemble_dir, "q_stds_episode.npy"))
    
    print(f"Baseline Q shape: {baseline_q.shape}")
    print(f"Ensemble Q Mean shape: {ensemble_q_mean.shape}")
    print(f"Ensemble Q Std shape: {ensemble_q_std.shape}")
except Exception as e:
    print(f"Error loading Q-values: {e}")
    exit(1)

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Colors
COLOR_BASELINE = '#1f77b4' # Blue
COLOR_ENSEMBLE = '#d62728' # Red (Using Red for high contrast/visibility as requested "vivid")
# Actually, let's use Red for Ensemble to make it pop more than Orange against Blue?
# User cited "Five algo comparison". DDPG was Red. MADDPG was Orange.
# I'll stick to Orange (#ff7f0e) as it's standard complimentary to Blue.
COLOR_ENSEMBLE = '#ff7f0e'

# ... (rest of code logic adapted with variables)
episodes_b = np.arange(len(baseline_rewards))
episodes_e = np.arange(len(ensemble_rewards))

# Raw
ax1.plot(episodes_b, baseline_rewards, color=COLOR_BASELINE, alpha=0.15, label='Baseline (Raw)')
ax1.plot(episodes_e, ensemble_rewards, color=COLOR_ENSEMBLE, alpha=0.15, label='Ensemble (Raw)')

# Smoothed
baseline_smooth = smooth_curve(baseline_rewards)
ensemble_smooth = smooth_curve(ensemble_rewards)
ax1.plot(episodes_b, baseline_smooth, color=COLOR_BASELINE, linewidth=2.5, label='Baseline (SAC)')
ax1.plot(episodes_e, ensemble_smooth, color=COLOR_ENSEMBLE, linewidth=2.5, label='Ensemble (Proposed)')

ax1.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
ax1.set_title('Training Reward Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Subplot 2: Q-Values ---
# Baseline Q: Just plot the line(s)
if baseline_q.ndim == 1:
    ax2.plot(episodes_b, baseline_q, color=COLOR_BASELINE, linewidth=2.5, label='Baseline Q (Mean)')
elif baseline_q.ndim == 2:
    mean_bq = np.mean(baseline_q, axis=1)
    min_bq = np.min(baseline_q, axis=1)
    max_bq = np.max(baseline_q, axis=1)
    ax2.plot(episodes_b, mean_bq, color=COLOR_BASELINE, linewidth=2.5, label='Baseline Q (Mean)')
    ax2.fill_between(episodes_b, min_bq, max_bq, color=COLOR_BASELINE, alpha=0.2, label='Baseline Q Range')
else:
    print("Unexpected Baseline Q shape, plotting line 0")
    ax2.plot(episodes_b, baseline_q[:, 0], color=COLOR_BASELINE, label='Baseline Q-0')

# Ensemble Q: Plot Mean +/- 2*Std (approx 95% interval)
ax2.plot(episodes_e, ensemble_q_mean, color=COLOR_ENSEMBLE, linewidth=2.5, label='Ensemble Q (Mean)')
ax2.fill_between(episodes_e, 
                 ensemble_q_mean - 2 * ensemble_q_std, 
                 ensemble_q_mean + 2 * ensemble_q_std, 
                 color=COLOR_ENSEMBLE, alpha=0.25, label='Ensemble Q (Mean ± 2σ)')

ax2.set_ylabel('Q-Value', fontsize=12, fontweight='bold')
ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
ax2.set_title('Q-Value Dynamics & Uncertainty', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', frameon=True, fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
