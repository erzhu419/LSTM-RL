#!/usr/bin/env python3
"""
Plot reward and Q-value curves for 4 methods: SAC, SAC-Ensemble, DSAC, DSAC-v2
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter1d

# Define the paths to the experiment logs (using the 161232 run with more complete data)
EXPERIMENTS = {
    'SAC': 'experiments/logs/sac_v2_bus_sigma1p5_embed-full_wreg0p03_paper1_sac_wreg0p03_20251022_161232',
    'SAC-Ensemble': 'experiments/logs/sac_v2_bus_ensemble_sigma1p5_embed-full_wreg0p03_paper1_ensemble_wreg0p03_20251022_161232',
    'DSAC': 'experiments/logs/dsac_bus_sigma1p5_embed-full_paper1_dsac_20251022_161232',
    'DSAC-v2': 'experiments/logs/dsac_bus_v2_sigma1p5_embed-full_risk-CVaR_tau-iqn_paper1_dsac_v2_20251022_161232',
}

# Colors for each method
COLORS = {
    'SAC': '#1f77b4',
    'SAC-Ensemble': '#ff7f0e',
    'DSAC': '#2ca02c',
    'DSAC-v2': '#d62728',
}

def smooth_curve(data, window_size=10):
    """Apply moving average smoothing to the data."""
    if len(data) < window_size:
        return data
    return uniform_filter1d(data, size=window_size, mode='nearest')

def load_data(method_name, log_dir):
    """Load rewards and Q-values for a given method."""
    rewards_path = os.path.join(log_dir, 'rewards.npy')

    # Try different Q-value file names
    q_values_path = os.path.join(log_dir, 'q_values.npy')
    if not os.path.exists(q_values_path):
        q_values_path = os.path.join(log_dir, 'q_values_episode.npy')

    data = {}

    # Load rewards
    if os.path.exists(rewards_path):
        rewards = np.load(rewards_path)
        data['rewards'] = rewards
        print(f"{method_name}: Loaded {len(rewards)} reward episodes")
    else:
        print(f"{method_name}: Warning - rewards.npy not found")
        data['rewards'] = None

    # Load Q-values
    if os.path.exists(q_values_path):
        q_values = np.load(q_values_path)
        data['q_values'] = q_values

        # For ensemble methods, q_values has shape (num_q_networks, episodes)
        # For other methods, q_values has shape (episodes,)
        if len(q_values.shape) == 2:
            print(f"{method_name}: Loaded ensemble Q-values with shape {q_values.shape} (networks x episodes)")
            # Scale by 50 to match the range of other methods (as done in original code)
            data['q_values_ensemble'] = q_values / 50.0
            # Also compute mean for comparison
            data['q_values'] = np.mean(q_values, axis=0) / 50.0
        else:
            print(f"{method_name}: Loaded {len(q_values)} Q-value episodes")
            data['q_values_ensemble'] = None
    else:
        print(f"{method_name}: Warning - Q-values not found")
        data['q_values'] = None
        data['q_values_ensemble'] = None

    return data

def plot_comparison():
    """Plot reward and Q-value curves for all methods."""
    # Load data for all methods
    all_data = {}
    for method_name, log_dir in EXPERIMENTS.items():
        if os.path.exists(log_dir):
            all_data[method_name] = load_data(method_name, log_dir)
        else:
            print(f"Warning: {log_dir} does not exist")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot 1: Rewards
    ax1 = axes[0]
    for method_name, data in all_data.items():
        if data['rewards'] is not None:
            rewards = data['rewards']
            episodes = np.arange(len(rewards))

            # Plot raw data with transparency
            ax1.plot(episodes, rewards, alpha=0.2, color=COLORS[method_name])

            # Plot smoothed curve
            smoothed = smooth_curve(rewards, window_size=20)
            ax1.plot(episodes, smoothed, label=method_name,
                    color=COLORS[method_name], linewidth=2)

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Rewards Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Q-values
    ax2 = axes[1]
    for method_name, data in all_data.items():
        if data['q_values_ensemble'] is not None:
            # For ensemble methods, plot all Q-networks
            q_ensemble = data['q_values_ensemble']
            episodes = np.arange(q_ensemble.shape[1])

            # Plot each Q-network with light lines
            for i in range(q_ensemble.shape[0]):
                ax2.plot(episodes, q_ensemble[i], alpha=0.15,
                        color=COLORS[method_name], linewidth=0.8)

            # Plot the mean of all Q-networks as the main line
            q_mean = np.mean(q_ensemble, axis=0)
            smoothed = smooth_curve(q_mean, window_size=20)
            ax2.plot(episodes, smoothed, label=method_name,
                    color=COLORS[method_name], linewidth=2)

        elif data['q_values'] is not None:
            # For non-ensemble methods, plot as before
            q_values = data['q_values']
            episodes = np.arange(len(q_values))

            # Plot raw data with transparency
            ax2.plot(episodes, q_values, alpha=0.2, color=COLORS[method_name])

            # Plot smoothed curve
            smoothed = smooth_curve(q_values, window_size=20)
            ax2.plot(episodes, smoothed, label=method_name,
                    color=COLORS[method_name], linewidth=2)

    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Q-value', fontsize=12)
    ax2.set_title('Q-values Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = 'four_methods_reward_qvalue_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    plt.show()

def print_statistics():
    """Print summary statistics for all methods."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for method_name, log_dir in EXPERIMENTS.items():
        if not os.path.exists(log_dir):
            continue

        data = load_data(method_name, log_dir)

        print(f"\n{method_name}:")
        print("-" * 40)

        if data['rewards'] is not None:
            rewards = data['rewards']
            print(f"  Rewards:")
            print(f"    Total episodes: {len(rewards)}")
            print(f"    Mean: {np.mean(rewards):.2f}")
            print(f"    Std: {np.std(rewards):.2f}")
            print(f"    Min: {np.min(rewards):.2f}")
            print(f"    Max: {np.max(rewards):.2f}")

            # Last 50 episodes statistics
            if len(rewards) >= 50:
                last_50 = rewards[-50:]
                print(f"    Last 50 episodes mean: {np.mean(last_50):.2f}")

        if data['q_values'] is not None:
            q_values = data['q_values']
            print(f"  Q-values:")
            print(f"    Total episodes: {len(q_values)}")
            print(f"    Mean: {np.mean(q_values):.2f}")
            print(f"    Std: {np.std(q_values):.2f}")
            print(f"    Min: {np.min(q_values):.2f}")
            print(f"    Max: {np.max(q_values):.2f}")

            # Last 50 episodes statistics
            if len(q_values) >= 50:
                last_50 = q_values[-50:]
                print(f"    Last 50 episodes mean: {np.mean(last_50):.2f}")

if __name__ == '__main__':
    print("Loading data for 4 methods comparison...")
    print_statistics()
    print("\nGenerating plots...")
    plot_comparison()
