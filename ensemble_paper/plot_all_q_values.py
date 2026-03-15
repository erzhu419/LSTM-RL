
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
# Path -> Label, Color, Linestyle, MainFile, StdFile(Optional)
experiments = {
    "model/sac_v2_bus": ("Baseline (SAC)", "black", "-", "q_values.npy", None),
    # "ensemble_paper/ensemble_2/logs": ("Ensemble 2", "#1f77b4", "-", "q_values_episode.npy", "q_stds_episode.npy"),
    "ensemble_paper/ensemble_5/logs": ("Ensemble 5", "#ff7f0e", "-", "q_values_episode.npy", "q_stds_episode.npy"),
    "ensemble_paper/ensemble_10/logs": ("Ensemble 10", "#2ca02c", "-", "q_values_episode.npy", "q_stds_episode.npy"),
    "ensemble_paper/ensemble_20/logs": ("Ensemble 20", "#d62728", "-", "q_values_episode.npy", "q_stds_episode.npy"),
    "ensemble_paper/ensemble_40/logs": ("Ensemble 40", "#9467bd", "-", "q_values_episode.npy", "q_stds_episode.npy"),
    "ensemble_paper/Aleatoric_Only/logs": ("Aleatoric Only", "#8c564b", "--", "q_values_episode.npy", "q_stds_episode.npy"),
    # "ensemble_paper/Epistemic_Only/logs": ("Epistemic Only", "#e377c2", "--", "q_values_episode.npy", "q_stds_episode.npy"),
}

OUTPUT_FILE = "ensemble_paper/all_q_values_comparison.png"

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
for log_dir, (label, color, linestyle, filename, std_filename) in experiments.items():
    file_path = os.path.join(log_dir, filename)
    
    if not os.path.exists(file_path):
        print(f"Warning: Missing {file_path}, skipping.")
        continue
        
    try:
        q_data = np.load(file_path)
        q_std = None
        
        # Handle Baseline which might be (N, 2)
        if q_data.ndim == 2:
            # Baseline doesn't have ensemble std, but we can compute std between the 2 critics
            # Or just plot range
            q_std = np.std(q_data, axis=1) # Std between the 2 critics
            q_data = np.mean(q_data, axis=1)
        elif std_filename:
            std_path = os.path.join(log_dir, std_filename)
            if os.path.exists(std_path):
                q_std = np.load(std_path)
            
        # Match length (x-axis)
        x = np.arange(len(q_data))
        
        # Plot smoothed Mean
        smoothed_mean = smooth_curve(q_data, factor=0.95)
        plt.plot(x, smoothed_mean, label=label, color=color, linestyle=linestyle, linewidth=2)
        
        # Plot Uncertainty Band (Smoothed)
        if q_std is not None and len(q_std) == len(x):
            smoothed_std = smooth_curve(q_std, factor=0.95)
            upper = smoothed_mean + 2 * smoothed_std
            lower = smoothed_mean - 2 * smoothed_std
            plt.fill_between(x, lower, upper, color=color, alpha=0.15)
            
        print(f"Loaded {label}: {len(q_data)} episodes")
        
    except Exception as e:
        print(f"Error loading {label}: {e}")

plt.title("Comparative Analysis: Q-Value Estimation with Uncertainty (Mean ± 2σ)", fontsize=16, fontweight='bold')
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Q-Value", fontsize=12)
plt.legend(loc='lower right', fontsize=10, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
print(f"Q-Value comparison plot saved to {OUTPUT_FILE}")
