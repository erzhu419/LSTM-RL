"""
Plot learning curves from rlkit experiment logs.

Reads progress.csv files from rlkit's data/ directory, plots evaluation
return vs. epochs for all algorithms, showing mean ± std across seeds.

Usage:
    python plot_results.py --data_dir /path/to/rlkit/data --output results.png

The script auto-discovers experiment directories by naming convention:
    {algorithm}_{env}_{version}/seed_{N}/progress.csv
"""
import argparse
import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# rlkit logs evaluation return under these possible column names
EVAL_RETURN_COLS = [
    'evaluation/Average Returns',
    'eval/Average Returns',
    'AverageReturn',
    'evaluation/Returns Mean',
]

ALGO_DISPLAY = {
    'sac': 'SAC',
    'td3': 'TD3',
    'dsac': 'DSAC',
    'resac': 'RE-SAC (Ours)',
}

ALGO_COLORS = {
    'SAC': '#1f77b4',
    'TD3': '#ff7f0e',
    'DSAC': '#2ca02c',
    'RE-SAC (Ours)': '#d62728',
}

ENV_DISPLAY = {
    'hopper': 'Hopper-v2',
    'halfcheetah': 'HalfCheetah-v2',
    'walker2d': 'Walker2d-v2',
    'ant': 'Ant-v3',
    'humanoid': 'Humanoid-v2',
    'bipedalwalkerhardcore': 'BipedalWalkerHardcore-v3',
}


def find_progress_files(data_dir):
    """Discover all progress.csv files and parse metadata."""
    results = defaultdict(lambda: defaultdict(list))
    # rlkit saves to: data/{log_prefix}_{timestamp}/progress.csv
    for csv_path in glob.glob(os.path.join(data_dir, '**/progress.csv'), recursive=True):
        dir_name = os.path.basename(os.path.dirname(csv_path))
        # Parse algorithm and env from directory name
        # Expected patterns: resac_hopper_ensemble-10_2024_..., sac_hopper_normal_2024_...
        parts = dir_name.lower().split('_')
        if len(parts) < 2:
            continue

        algo = parts[0]
        env = parts[1]
        if algo not in ALGO_DISPLAY:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        # Find the evaluation return column
        return_col = None
        for col in EVAL_RETURN_COLS:
            if col in df.columns:
                return_col = col
                break
        if return_col is None:
            print(f"Warning: no eval return column found in {csv_path}")
            print(f"  Available columns: {list(df.columns[:10])}")
            continue

        results[env][algo].append(df[return_col].values)

    return results


def smooth(data, window=5):
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_learning_curves(results, output_path, smooth_window=5):
    """Plot learning curves: one subplot per environment."""
    envs = sorted(results.keys())
    if not envs:
        print("No results found!")
        return

    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        algo_data = results[env]
        for algo_key in ['sac', 'td3', 'dsac', 'resac']:
            if algo_key not in algo_data:
                continue
            runs = algo_data[algo_key]
            display_name = ALGO_DISPLAY[algo_key]
            color = ALGO_COLORS[display_name]

            # Truncate to shortest run
            min_len = min(len(r) for r in runs)
            aligned = np.array([r[:min_len] for r in runs])

            mean = np.mean(aligned, axis=0)
            std = np.std(aligned, axis=0)

            if smooth_window > 1:
                mean_s = smooth(mean, smooth_window)
                std_s = smooth(std, smooth_window)
                x = np.arange(len(mean_s))
            else:
                mean_s, std_s = mean, std
                x = np.arange(len(mean_s))

            ax.plot(x, mean_s, label=display_name, color=color, linewidth=2)
            ax.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.15, color=color)

        env_title = ENV_DISPLAY.get(env, env)
        ax.set_title(env_title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Average Return', fontsize=12)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_table(results):
    """Print final performance table: mean ± std of last 10 epochs."""
    print("\n" + "="*80)
    print("Final Performance (last 10 epochs, mean ± std across seeds)")
    print("="*80)

    envs = sorted(results.keys())
    algos = ['sac', 'td3', 'dsac', 'resac']

    header = f"{'Environment':<25}" + "".join(f"{ALGO_DISPLAY.get(a, a):>18}" for a in algos)
    print(header)
    print("-" * len(header))

    for env in envs:
        row = f"{ENV_DISPLAY.get(env, env):<25}"
        for algo in algos:
            if algo in results[env]:
                runs = results[env][algo]
                min_len = min(len(r) for r in runs)
                aligned = np.array([r[:min_len] for r in runs])
                # Last 10 epochs average per seed, then mean/std across seeds
                last_k = min(10, aligned.shape[1])
                seed_means = aligned[:, -last_k:].mean(axis=1)
                mean = seed_means.mean()
                std = seed_means.std()
                row += f"{mean:>10.1f}±{std:<6.1f}"
            else:
                row += f"{'N/A':>18}"
        print(row)

    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot MuJoCo benchmark results')
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/mine_code/dsac/data'),
                        help='Root directory containing rlkit experiment data')
    parser.add_argument('--output', type=str, default='mujoco_benchmark_results.png',
                        help='Output image path')
    parser.add_argument('--smooth', type=int, default=5, help='Smoothing window')
    args = parser.parse_args()

    results = find_progress_files(args.data_dir)
    plot_learning_curves(results, args.output, smooth_window=args.smooth)
    print_summary_table(results)
