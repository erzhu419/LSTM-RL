
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
# Path -> Label, Color
experiments = {
    "model/sac_v2_bus": ("Baseline (SAC)", "black", "-"),
    "ensemble_paper/ensemble_2/logs": ("Ensemble 2", "#1f77b4", "-"),   # Blue
    "ensemble_paper/ensemble_5/logs": ("Ensemble 5", "#ff7f0e", "-"),   # Orange
    "ensemble_paper/ensemble_10/logs": ("Ensemble 10", "#2ca02c", "-"),  # Green
    "ensemble_paper/ensemble_20/logs": ("Ensemble 20", "#d62728", "-"),  # Red
    "ensemble_paper/ensemble_40/logs": ("Ensemble 40", "#9467bd", "-"),  # Purple
    "ensemble_paper/Aleatoric_Only/logs": ("Aleatoric Only", "#8c564b", "--"), # Brown, Dashed
    "ensemble_paper/Epistemic_Only/logs": ("Epistemic Only", "#e377c2", "--"), # Pink, Dashed
}

OUTPUT_FILE = "ensemble_paper/all_experiments_comparison.png"

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)

plt.figure(figsize=(14, 8))

# Iterate and Plot
for log_dir, (label, color, linestyle) in experiments.items():
    reward_path = os.path.join(log_dir, "rewards.npy")
    
    if not os.path.exists(reward_path):
        print(f"Warning: Missing {reward_path}, skipping.")
        continue
        
    try:
        rewards = np.load(reward_path)
        # Handle different lengths by using x-axis
        x = np.arange(len(rewards))
        
        # Plot smoothed
        smoothed = smooth_curve(rewards, factor=0.95) # Stronger smoothing for clarity with many lines
        
        plt.plot(x, smoothed, label=label, color=color, linestyle=linestyle, linewidth=2)
        
        # Optional: Plot raw transparently (maybe too messy for 8 lines, skipping for now)
        # plt.plot(x, rewards, color=color, alpha=0.1)
        
        print(f"Loaded {label}: {len(rewards)} episodes")
        
    except Exception as e:
        print(f"Error loading {label}: {e}")

plt.title("Comparative Analysis: Ensemble Sizes & Uncertainty Ablations", fontsize=16, fontweight='bold')
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Cumulative Reward (Smoothed)", fontsize=12)
plt.legend(loc='lower right', fontsize=10, ncol=2) # 2 columns for legend
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Comparison plot saved to {OUTPUT_FILE}")
