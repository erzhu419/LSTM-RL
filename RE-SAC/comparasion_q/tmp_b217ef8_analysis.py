"""
Mahalanobis-distance based Rareness analysis.

Rareness = Mahalanobis distance of (forward_headway, backward_headway)
           conditioned on (station_id, direction).

For each group (station_id, direction), fit a 2D Gaussian on all data points,
then compute the distance of each point from the group center.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import LinearRegression


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all_data(data_dir, prefix):
    files = sorted(glob.glob(os.path.join(data_dir, f'data_{prefix}_*.pkl')))
    print(f"Loading {len(files)} files for {prefix}...")
    all_results = []
    for f in tqdm(files, desc=f"Loading {prefix}"):
        with open(f, 'rb') as fh:
            chunk = pickle.load(fh)
        ckpt_idx = int(os.path.basename(f).split('_')[-1].replace('.pkl', ''))
        events = chunk.get('data', chunk.get('events', []))
        results = process_chunk(events, ckpt_idx)
        all_results.extend(results)
    return pd.DataFrame(all_results)


def process_chunk(events, ckpt_idx, gamma=0.99):
    df = pd.DataFrame(events)
    if df.empty:
        return []
    df = df.sort_values(['episode_sub_idx', 'bus_id', 'time'])
    results = []
    grouped = df.groupby(['episode_sub_idx', 'bus_id'])
    for _, group in grouped:
        evts = group.to_dict('records')
        preds = [e for e in evts if e['event'] == 'predict']
        rewards = [e for e in evts if e['event'] == 'reward']
        n = min(len(preds), len(rewards))
        # Monte Carlo Q_real
        running_q = 0
        q_reals = [0] * n
        for i in range(n - 1, -1, -1):
            running_q = rewards[i]['reward'] + gamma * running_q
            q_reals[i] = running_q
        for i in range(n):
            rec = preds[i]
            q_vals = np.array(rec['q_vals'])
            state = rec['state_raw']
            results.append({
                'checkpoint': ckpt_idx,
                'station_id': int(state[1]),
                'direction': int(state[3]),
                'forward_headway': float(state[4]),
                'backward_headway': float(state[5]),
                'q_pred_mean': float(np.mean(q_vals)),
                'q_pred_std': float(np.std(q_vals)),
                'q_real': q_reals[i],
                'q_vals': q_vals.tolist(),
            })
    return results


# ── Mahalanobis Computation ──────────────────────────────────────────────────

def compute_mahalanobis_rareness(df, feature_cols=['forward_headway', 'backward_headway'],
                                  group_cols=['station_id', 'direction']):
    """Compute Mahalanobis distance per row, conditioned on group."""
    df = df.copy()
    df['rareness'] = np.nan

    for name, grp in tqdm(df.groupby(group_cols), desc="Computing Mahalanobis"):
        idx = grp.index
        X = grp[feature_cols].values  # (N, 2)

        if len(X) < 3:
            # Too few samples to estimate covariance
            df.loc[idx, 'rareness'] = 0.0
            continue

        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)

        # Regularize covariance to avoid singularity
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            df.loc[idx, 'rareness'] = 0.0
            continue

        distances = np.array([mahalanobis(x, mean, cov_inv) for x in X])
        df.loc[idx, 'rareness'] = distances

    return df


# ── Alignment & Oracle Error ─────────────────────────────────────────────────

def get_aligned_error(df, pred_col='q_pred_mean'):
    X = df[pred_col].values.reshape(-1, 1)
    y = df['q_real'].values
    reg = LinearRegression().fit(X, y)
    aligned_pred = reg.predict(X)
    return aligned_pred - y, reg


def calculate_best_head_error(df, reg):
    errs = []
    alpha = reg.coef_[0]
    beta = reg.intercept_
    for q_heads, q_r in zip(df['q_vals'].values, df['q_real'].values):
        aligned = np.array(q_heads) * alpha + beta
        diffs = aligned - q_r
        errs.append(diffs[np.argmin(np.abs(diffs))])
    return np.array(errs)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(df_ens, df_sac, output_dir='analysis_results'):
    os.makedirs(output_dir, exist_ok=True)

    # Align
    print("Aligning Ensemble...")
    df_ens['error'], reg_ens = get_aligned_error(df_ens)
    print(f"  Coef={reg_ens.coef_[0]:.2f}, Intercept={reg_ens.intercept_:.2f}")

    print("Aligning SAC...")
    df_sac['error'], reg_sac = get_aligned_error(df_sac)
    print(f"  Coef={reg_sac.coef_[0]:.2f}, Intercept={reg_sac.intercept_:.2f}")

    # Oracle best-head error
    print("Computing Oracle errors...")
    df_ens['error_oracle'] = calculate_best_head_error(df_ens, reg_ens)
    df_sac['error_oracle'] = calculate_best_head_error(df_sac, reg_sac)

    # ── Bin by Mahalanobis rareness (equal-width bins) ──
    num_bins = 30

    # Clip extreme rareness for cleaner bins
    r_clip_ens = np.percentile(df_ens['rareness'].dropna(), 99)
    r_clip_sac = np.percentile(df_sac['rareness'].dropna(), 99)
    r_clip = max(r_clip_ens, r_clip_sac)

    df_ens['rareness_clipped'] = df_ens['rareness'].clip(upper=r_clip)
    df_sac['rareness_clipped'] = df_sac['rareness'].clip(upper=r_clip)

    df_ens['rare_bin'] = pd.cut(df_ens['rareness_clipped'], bins=num_bins, labels=False)
    df_sac['rare_bin'] = pd.cut(df_sac['rareness_clipped'], bins=num_bins, labels=False)

    # Compute per-bin MAE
    ens_mae = df_ens.groupby('rare_bin')['error_oracle'].apply(lambda x: x.abs().mean())
    sac_mae = df_sac.groupby('rare_bin')['error_oracle'].apply(lambda x: x.abs().mean())

    # Data density
    ens_density = df_ens.groupby('rare_bin').size()
    sac_density = df_sac.groupby('rare_bin').size()

    # ── Print stats ──
    print(f"\nEnsemble: Overall Oracle MAE = {df_ens['error_oracle'].abs().mean():.1f}")
    print(f"Vanilla:  Overall Oracle MAE = {df_sac['error_oracle'].abs().mean():.1f}")
    print(f"\nRareness range: 0 — {r_clip:.1f} (99th pct)")
    print(f"Ensemble points: {len(df_ens)}, Vanilla points: {len(df_sac)}")

    # ── Plot ──
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Density bars (secondary y-axis)
    ax2 = ax1.twinx()
    bar_width = 0.35
    bins_ens = ens_density.index.values
    bins_sac = sac_density.index.values
    ax2.bar(bins_ens - bar_width / 2, ens_density.values, width=bar_width,
            alpha=0.15, color='blue', label='Ensemble Data Count')
    ax2.bar(bins_sac + bar_width / 2, sac_density.values, width=bar_width,
            alpha=0.15, color='orange', label='Vanilla Data Count')
    ax2.set_ylabel('Data Count per Bin', color='gray', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='gray')

    # MAE lines (primary y-axis, in front)
    ax1.plot(sac_mae.index, sac_mae.values, label='Vanilla SAC (Best of 2)',
             color='orange', linewidth=2.5, marker='x', zorder=5)
    ax1.plot(ens_mae.index, ens_mae.values, label='Ensemble (Best of 10)',
             color='#4488FF', linewidth=2.5, marker='o', markersize=5, zorder=5)

    ax1.set_title('Oracle Q-Error vs Mahalanobis Rareness\n'
                   '(Rareness = distance of headways from per-station distribution)',
                   fontsize=13, fontweight='bold')
    ax1.set_xlabel('Mahalanobis Rareness Bin (Low → High)', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error of Best Head', fontsize=12)
    ax1.axhline(0, color='red', linestyle='-', alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    ax1.grid(alpha=0.2)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'comparison_mahalanobis_rareness.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\nSaved to {out_path}")

    # ── Also plot rareness distribution histogram ──
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_ens['rareness'].dropna().values, bins=100, color='blue', alpha=0.6, edgecolor='none')
    axes[0].set_title('Ensemble: Mahalanobis Rareness Distribution')
    axes[0].set_xlabel('Mahalanobis Distance')
    axes[0].set_ylabel('Count')
    axes[0].axvline(r_clip, color='red', linestyle='--', label=f'99th pct = {r_clip:.1f}')
    axes[0].legend()

    axes[1].hist(df_sac['rareness'].dropna().values, bins=100, color='orange', alpha=0.6, edgecolor='none')
    axes[1].set_title('Vanilla SAC: Mahalanobis Rareness Distribution')
    axes[1].set_xlabel('Mahalanobis Distance')
    axes[1].set_ylabel('Count')
    axes[1].axvline(r_clip, color='red', linestyle='--', label=f'99th pct = {r_clip:.1f}')
    axes[1].legend()

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, 'mahalanobis_distribution.png')
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Saved distribution to {out_path2}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_DIR = 'offline_dataset_full'

    print("Loading Ensemble Data...")
    df_ens = load_all_data(DATA_DIR, 'ensemble')
    print(f"  {len(df_ens)} rows")

    print("Loading SAC Data...")
    df_sac = load_all_data(DATA_DIR, 'sac')
    print(f"  {len(df_sac)} rows")

    print("\nComputing Mahalanobis Rareness for Ensemble...")
    df_ens = compute_mahalanobis_rareness(df_ens)

    print("Computing Mahalanobis Rareness for SAC...")
    df_sac = compute_mahalanobis_rareness(df_sac)

    print(f"\nEnsemble rareness: mean={df_ens['rareness'].mean():.2f}, "
          f"median={df_ens['rareness'].median():.2f}, max={df_ens['rareness'].max():.2f}")
    print(f"Vanilla  rareness: mean={df_sac['rareness'].mean():.2f}, "
          f"median={df_sac['rareness'].median():.2f}, max={df_sac['rareness'].max():.2f}")

    plot_comparison(df_ens, df_sac)
