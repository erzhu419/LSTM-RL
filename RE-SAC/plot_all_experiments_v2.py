
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
# Path -> Label, Color, Linestyle, Type (npy or npz)
BASE_DIR = "/home/erzhu419/mine_code/LSTM-RL"
experiments = {
    os.path.join(BASE_DIR, "model/sac_v2_bus"): ("Baseline (SAC)", "black", "-", "npy"),
    os.path.join(BASE_DIR, "RE-SAC/ensemble_2/logs"): ("Ensemble 2", "#1f77b4", "-", "npy"),
    os.path.join(BASE_DIR, "RE-SAC/ensemble_5/logs"): ("Ensemble 5", "#ff7f0e", "-", "npy"),
    os.path.join(BASE_DIR, "RE-SAC/ensemble_10/logs"): ("Ensemble 10", "#2ca02c", "-", "npy"),
    os.path.join(BASE_DIR, "RE-SAC/ensemble_20/logs"): ("Ensemble 20", "#d62728", "-", "npy"),
    os.path.join(BASE_DIR, "RE-SAC/ensemble_40/logs"): ("Ensemble 40", "#9467bd", "-", "npy"),
    os.path.join(BASE_DIR, "RE-SAC/Aleatoric_Only/logs"): ("Aleatoric Only", "#8c564b", "--", "npy"),
    os.path.join(BASE_DIR, "RE-SAC/Epistemic_Only/logs"): ("Epistemic Only", "#e377c2", "--", "npy"),
    # New DSAC versions
    os.path.join(BASE_DIR, "RE-SAC/full_rewards_debug.npy"): ("DSAC-v1", "#00ced1", "-.", "npy_direct"), # DarkTurquoise
    os.path.join(BASE_DIR, "RE-SAC/logs_bac/logs/bac_v1_lambda0p5_q0p7"): ("BAC", "#9400D3", ":", "npy"), # DarkViolet
}

OUTPUT_FILE = os.path.join(BASE_DIR, "RE-SAC/all_experiments_comparison_v2.png")

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)

plt.figure(figsize=(18, 10))

# Iterate and Plot
for path, (label, color, linestyle, file_type) in experiments.items():
    try:
        if file_type == "npy":
            reward_path = os.path.join(path, "rewards.npy")
            if not os.path.exists(reward_path):
                print(f"Warning: {reward_path} not found.")
                continue
            rewards = np.load(reward_path)
        elif file_type == "npy_direct":
            if not os.path.exists(path):
                print(f"Warning: {path} not found.")
                continue
            rewards = np.load(path)
        elif file_type == "npz":
            if not os.path.exists(path):
                print(f"Warning: {path} not found.")
                continue
            data = np.load(path)
            rewards = data['rewards']
        else:
            continue

        # Handle different lengths by using x-axis
        x = np.arange(len(rewards))
        
        # Plot smoothed
        smoothed = smooth_curve(rewards, factor=0.95)
        
        plt.plot(x, smoothed, label=label, color=color, linestyle=linestyle, linewidth=2.5 if "DSAC" in label else 2.0)
        
        print(f"Loaded {label}: {len(rewards)} episodes, Last 100 Mean: {np.mean(rewards[-100:])}")
        
    except Exception as e:
        print(f"Error loading {path} ({label}): {e}")

plt.title("Comparative Analysis: Ensemble vs SAC vs DSAC Versions", fontsize=20, fontweight='bold')
plt.xlabel("Episode", fontsize=16)
plt.ylabel("Cumulative Reward (Smoothed)", fontsize=16)
plt.legend(loc='lower right', fontsize=14, ncol=2)
plt.tick_params(labelsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Comparison plot saved to {OUTPUT_FILE}")
