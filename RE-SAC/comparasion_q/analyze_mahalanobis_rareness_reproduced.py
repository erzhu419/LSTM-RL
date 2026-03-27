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

import sys

def load_all_data(data_dir, prefix):
    files = sorted(glob.glob(os.path.join(data_dir, f'data_{prefix}_*.pkl')))
    # Sort numerically by checkpoint index
    files = sorted(files, key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.pkl','')))
    # Exactly 1000 episodes (50 checkpoints of 20 eps each out of 500 total files)
    if len(files) > 50:
        files = files[::(len(files)//50)]
    print(f"Loading {len(files)} files for {prefix}...", flush=True)
    dfs = []
    for f in tqdm(files, desc=f"Loading {prefix}"):
        with open(f, 'rb') as fh:
            chunk = pickle.load(fh)
        ckpt_idx = int(os.path.basename(f).split('_')[-1].replace('.pkl', ''))
        events = chunk.get('data', chunk.get('events', []))
        print(f"Loaded {len(events)} events from {f}. Processing...", flush=True)
        df = process_chunk_vectorized(events, ckpt_idx)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def process_chunk_vectorized(events, ckpt_idx, gamma=0.99):
    if not events: return pd.DataFrame()
    
    # 1. Convert to DataFrame (only necessary columns)
    df = pd.DataFrame(events)
    
    # Filter for necessary events
    # Important: Drop the reward col from preds to avoid merge collision
    preds = df[df['event'] == 'predict'].drop(columns=['reward'], errors='ignore').copy()
    rewards = df[df['event'] == 'reward'].copy()
    
    if preds.empty or rewards.empty: return pd.DataFrame()

    # Assign a 'sequence_id' within each (episode_sub_idx, bus_id)
    preds['seq'] = preds.groupby(['episode_sub_idx', 'bus_id']).cumcount()
    rewards['seq'] = rewards.groupby(['episode_sub_idx', 'bus_id']).cumcount()
    
    # Merge preds and rewards on (ep, bus, seq)
    merged = pd.merge(preds, rewards[['episode_sub_idx', 'bus_id', 'seq', 'reward']], 
                      on=['episode_sub_idx', 'bus_id', 'seq'], how='inner')
    
    if merged.empty: return pd.DataFrame()
    
    # Sort for reward accumulation (Reverse seq order)
    merged = merged.sort_values(['episode_sub_idx', 'bus_id', 'seq'], ascending=[True, True, False])
    
    def compute_returns_for_array(r):
        # merged is sorted in REVERSE seq order
        # Calculate returns: Q_t = r_t + gamma * Q_{t+1}
        qs = np.zeros_like(r)
        curr = 0
        for i in range(len(r)):
            curr = r[i] + gamma * curr
            qs[i] = curr
        return qs

    merged['q_real'] = merged.groupby(['episode_sub_idx', 'bus_id'], group_keys=False)['reward'].transform(lambda r: compute_returns_for_array(r.values))

    
    # Extract features from state_raw list
    # Ensure they are all the same length
    state_list = merged['state_raw'].values.tolist()
    states = np.array(state_list)
    
    final_df = pd.DataFrame({
        'checkpoint': ckpt_idx,
        'station_id': states[:, 1].astype(int),
        'direction': states[:, 3].astype(int),
        'forward_headway': states[:, 4].astype(float),
        'backward_headway': states[:, 5].astype(float),
        'q_pred_mean': merged['q_vals'].apply(np.mean).values,
        'q_pred_std': merged['q_vals'].apply(np.std).values,
        'q_real': merged['q_real'].values,
        'q_vals': merged['q_vals'].values
    })
    
    return final_df


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

    # Reconstruct the Oracle Predict Q value
    df_ens['q_pred_oracle'] = df_ens['q_real'] + df_ens['error_oracle']
    df_sac['q_pred_oracle'] = df_sac['q_real'] + df_sac['error_oracle']

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

    # Compute per-bin mean Q-values
    ens_q_pred = df_ens.groupby('rare_bin')['q_pred_oracle'].mean()
    sac_q_pred = df_sac.groupby('rare_bin')['q_pred_oracle'].mean()
    q_real_mean = df_ens.groupby('rare_bin')['q_real'].mean()

    # Data density
    ens_density = df_ens.groupby('rare_bin').size()
    sac_density = df_sac.groupby('rare_bin').size()

    # ── Print stats ──
    print(f"\nEnsemble: Overall Oracle MAE = {df_ens['error_oracle'].abs().mean():.1f}")
    print(f"Vanilla:  Overall Oracle MAE = {df_sac['error_oracle'].abs().mean():.1f}")
    print(f"\nRareness range: 0 — {r_clip:.1f} (99th pct)")
    print(f"Ensemble points: {len(df_ens)}, Vanilla points: {len(df_sac)}")

    # ── Plot ──
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # --- Top Subplot: Q-Values ---
    ax0.plot(q_real_mean.index, q_real_mean.values, 'k--', linewidth=2, label='Ground Truth (Real Q)')
    ax0.plot(ens_q_pred.index, ens_q_pred.values, color='#4488FF', linewidth=2.5, marker='o', markersize=5, label='Ensemble (Oracle Best Predict)')
    ax0.plot(sac_q_pred.index, sac_q_pred.values, color='orange', linewidth=2.5, marker='x', label='Vanilla SAC (Oracle Best Predict)')
    ax0.set_title('Mean Q-Value Predictions vs Ground Truth', fontsize=13, fontweight='bold')
    ax0.set_ylabel('Q-Value', fontsize=12)
    ax0.legend(loc='upper left', fontsize=11)
    ax0.grid(alpha=0.3)

    # --- Bottom Subplot: MAE & Density ---
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
    ax1.plot(sac_mae.index, sac_mae.values, label='Vanilla SAC (Best of 2) MAE',
             color='orange', linewidth=2.5, marker='x', zorder=5)
    ax1.plot(ens_mae.index, ens_mae.values, label='Ensemble (Best of 10) MAE',
             color='#4488FF', linewidth=2.5, marker='o', markersize=5, zorder=5)

    ax1.set_title('Oracle Q-Error (MAE) & Data Density', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Mahalanobis Rareness Bin (Low → High)', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error of Best Head', fontsize=12)
    ax1.axhline(0, color='red', linestyle='-', alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'comparison_mahalanobis_rareness_reproduced.png')
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
    DATA_DIR = 'offline_timeline_data_20eps'

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
