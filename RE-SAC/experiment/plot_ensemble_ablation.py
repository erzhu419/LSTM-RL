"""
Plot RE-SAC ensemble size ablation study.

Compares RE-SAC with different ensemble sizes (N=2, 5, 10, 20) on MuJoCo,
similar to DSAC's risk_param ablation.

Usage:
    python plot_ensemble_ablation.py --data_dir /path/to/data --output ablation.png
"""
import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

EVAL_RETURN_COLS = [
    'evaluation/Average Returns',
    'eval/Average Returns',
    'AverageReturn',
    'evaluation/Returns Mean',
]

ENV_DISPLAY = {
    'hopper': 'Hopper-v2',
    'halfcheetah': 'HalfCheetah-v2',
    'walker2d': 'Walker2d-v2',
    'ant': 'Ant-v3',
}

ENSEMBLE_COLORS = {
    2: '#e377c2',
    5: '#ff7f0e',
    10: '#d62728',
    20: '#9467bd',
}


def find_ablation_files(data_dir):
    """Find RE-SAC runs with different ensemble sizes."""
    results = defaultdict(lambda: defaultdict(list))

    for csv_path in glob.glob(os.path.join(data_dir, '**/progress.csv'), recursive=True):
        dir_name = os.path.basename(os.path.dirname(csv_path))
        parts = dir_name.lower().split('_')

        if len(parts) < 3 or parts[0] != 'resac':
            continue

        env = parts[1]

        # Parse ensemble size from version string e.g. "ensemble-10"
        ensemble_size = 10  # default
        for part in parts:
            if part.startswith('ensemble-'):
                try:
                    ensemble_size = int(part.split('-')[1])
                except (ValueError, IndexError):
                    pass

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        return_col = None
        for col in EVAL_RETURN_COLS:
            if col in df.columns:
                return_col = col
                break
        if return_col is None:
            continue

        results[env][ensemble_size].append(df[return_col].values)

    return results


def smooth(data, window=5):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_ablation(results, output_path, smooth_window=5):
    envs = sorted(results.keys())
    if not envs:
        print("No ablation results found!")
        return

    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5))
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        ensemble_data = results[env]
        for n_ens in sorted(ensemble_data.keys()):
            runs = ensemble_data[n_ens]
            color = ENSEMBLE_COLORS.get(n_ens, '#333333')

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

            ax.plot(x, mean_s, label=f'N={n_ens}', color=color, linewidth=2)
            ax.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.15, color=color)

        env_title = ENV_DISPLAY.get(env, env)
        ax.set_title(f'RE-SAC Ablation: {env_title}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Average Return', fontsize=12)
        ax.legend(title='Ensemble Size', fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot ensemble size ablation')
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/mine_code/dsac/data'))
    parser.add_argument('--output', type=str, default='ensemble_ablation.png')
    parser.add_argument('--smooth', type=int, default=5)
    args = parser.parse_args()

    results = find_ablation_files(args.data_dir)
    plot_ablation(results, args.output, smooth_window=args.smooth)
