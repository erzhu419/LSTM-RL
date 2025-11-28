import os

# Ensure headless-friendly defaults for matplotlib/OMP
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    csv_path = Path('ensemble_original_500ep_stats.csv')
    if not csv_path.exists():
        raise SystemExit(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)
    df_r2 = df[df['critic_actor_ratio'] == 2].copy()
    if df_r2.empty:
        raise SystemExit('No rows with critic_actor_ratio == 2 in the CSV')

    # Build display labels and sort by performance
    df_r2['label'] = df_r2.apply(
        lambda row: f"α={row['max_alpha']} | buf={int(row['buffer_size']) // 1000}k",
        axis=1,
    )
    df_r2 = df_r2.sort_values('mean_last_50', ascending=False)

    # Plot mean reward with std dev error bars
    plt.figure(figsize=(9, 5.2))
    y_positions = range(len(df_r2))
    plt.barh(
        y_positions,
        df_r2['mean_last_50'],
        xerr=df_r2['std_last_50'],
        alpha=0.85,
        color='#4C78A8',
        ecolor='gray',
        capsize=3,
    )
    plt.yticks(list(y_positions), df_r2['label'], fontsize=9)
    plt.xlabel('Mean Reward (Last 50 Episodes)')
    plt.title('Ensemble Original (r=2) Comparison')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    output = Path('ensemble_original_r2_comparison.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
