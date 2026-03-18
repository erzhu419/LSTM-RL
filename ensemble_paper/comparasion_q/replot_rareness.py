"""
replot_rareness.py — Regenerate the comparison_mahalanobis_rareness figure
from the saved bin-statistics cache (bin_stats_all.pkl).

No need to reprocess the 32GB of raw data.
Just tweak the style section below and run:

    python replot_rareness.py
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Style tweaks — edit freely ────────────────────────────────────────────────
TITLE_FONTSIZE   = 16    # subplot title font size
LABEL_FONTSIZE   = 14    # axis label font size
TICK_FONTSIZE    = 12    # tick label font size
LEGEND_FONTSIZE  = 12    # legend font size
LINE_WIDTH       = 2.5   # main curve linewidth
MARKER_SIZE      = 6     # marker size
FIG_SIZE         = (16, 14)   # (width, height) in inches
DPI              = 300
# ─────────────────────────────────────────────────────────────────────────────

STATS_PATH = 'analysis_results/bin_stats_all.pkl'
OUT_PATH   = 'analysis_results/comparison_mahalanobis_rareness_restyled.png'


def main():
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"Cache not found: {STATS_PATH}\n"
            "Run analyze_mahalanobis_rareness.py first to generate the cache."
        )

    with open(STATS_PATH, 'rb') as f:
        data = pickle.load(f)

    meta   = data['meta']
    stats  = data['stats']

    bin_centers     = meta['bin_centers']
    bin_width       = meta['bin_width']
    global_r_clip   = meta['global_r_clip']
    algos           = meta['algos']
    labels          = meta['labels']
    colors          = meta['colors']
    markers         = meta['markers']
    bar_width_ratio = meta['bar_width_ratio']

    offsets = np.linspace(
        -bar_width_ratio * (len(algos) - 1) / 2,
         bar_width_ratio * (len(algos) - 1) / 2,
        len(algos)
    )

    mpl.rcParams.update({'font.size': TICK_FONTSIZE})

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=FIG_SIZE, sharex=True,
        gridspec_kw={'height_ratios': [1, 1]}
    )
    ax2 = ax1.twinx()

    # Ground truth
    if 'ground_truth' in stats:
        gt = stats['ground_truth']
        ax0.plot(bin_centers[gt.index.values], gt.values,
                 'k--', linewidth=LINE_WIDTH, label='Ground Truth (Real Q)')

    for i, name in enumerate(algos):
        if name not in stats:
            continue
        s = stats[name]

        # Q-value curve
        q = s['q_pred']
        ax0.plot(bin_centers[q.index.values], q.values,
                 color=colors[name], linewidth=LINE_WIDTH,
                 marker=markers[name], markersize=MARKER_SIZE,
                 label=labels[name])

        # MAE curve
        m = s['mae']
        ax1.plot(bin_centers[m.index.values], m.values,
                 color=colors[name], linewidth=LINE_WIDTH,
                 marker=markers[name], markersize=MARKER_SIZE,
                 label=f'{labels[name]} MAE', zorder=5)

        # Density bars (Normalized to Percentage)
        d = s['density']
        total_data_points = s.get('total_count', d.sum())
        density_percentage = (d.values / total_data_points) * 100 if total_data_points > 0 else d.values
        
        ax2.bar(bin_centers[d.index.values] + offsets[i] * bin_width,
                density_percentage,
                width=bar_width_ratio * bin_width,
                alpha=0.15, color=colors[name],
                label=f'{labels[name]} Density (%)')

    # ── Formatting ───────────────────────────────────────────────────────────
    ax0.set_title('Mean Q-Value Predictions vs Ground Truth (Variance Matching)',
                  fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax0.set_ylabel('Q-Value', fontsize=LABEL_FONTSIZE)
    ax0.legend(loc='upper right', fontsize=LEGEND_FONTSIZE)
    ax0.tick_params(labelsize=TICK_FONTSIZE, labelbottom=True)
    ax0.grid(alpha=0.3)

    ax1.set_title('Oracle Q-Error (MAE) & Data Density',
                  fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax1.set_xlabel(f'Mahalanobis Rareness (Clipped at {global_r_clip})',
                   fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('Mean Absolute Error of Best Head', fontsize=LABEL_FONTSIZE)
    ax1.axhline(0, color='red', linestyle='-', alpha=0.3)
    ax1.tick_params(labelsize=TICK_FONTSIZE)
    ax1.grid(alpha=0.3)

    ax2.set_ylabel('Data Density (%)', color='gray', fontsize=LABEL_FONTSIZE)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=TICK_FONTSIZE)

    lines1, lbs1 = ax1.get_legend_handles_labels()
    lines2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbs1 + lbs2,
               loc='upper left', fontsize=LEGEND_FONTSIZE)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=DPI)
    plt.close()
    print(f"Saved restyled plot to {OUT_PATH}")


if __name__ == '__main__':
    main()
