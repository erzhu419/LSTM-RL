"""
Reproduction of comparison_mahalanobis_rareness_fixed_v3.png.
Uses data from offline_timeline_data_20eps (subsampled to 25 checkpoints per model).
Initial version: SAC and Ensemble only.
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

def load_data_subsampled(data_dir, prefix, step=20):
    all_files = sorted(glob.glob(os.path.join(data_dir, f'data_{prefix}_*.pkl')), 
                       key=lambda x: int(os.path.basename(x).split('_')[-1].replace('.pkl', '')))
    
    # Load every 'step' files
    target_files = all_files[::step]
    if not target_files:
        return pd.DataFrame()
        
    print(f"Loading {len(target_files)} files (subsampled from {len(all_files)}) for {prefix}...", flush=True)
    dfs = []
    for f in tqdm(target_files, desc=f"Loading {prefix}"):
        with open(f, 'rb') as fh:
            chunk = pickle.load(fh)
        ckpt_idx = int(os.path.basename(f).split('_')[-1].replace('.pkl', ''))
        events = chunk.get('data', chunk.get('events', []))
        df = process_chunk_vectorized(events, ckpt_idx)
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def process_chunk_vectorized(events, ckpt_idx, gamma=0.99):
    if not events: return pd.DataFrame()
    df = pd.DataFrame(events)
    preds = df[df['event'] == 'predict'].drop(columns=['reward'], errors='ignore').copy()
    rewards = df[df['event'] == 'reward'].copy()
    if preds.empty or rewards.empty: return pd.DataFrame()

    preds['seq'] = preds.groupby(['episode_sub_idx', 'bus_id']).cumcount()
    rewards['seq'] = rewards.groupby(['episode_sub_idx', 'bus_id']).cumcount()
    
    merged = pd.merge(preds, rewards[['episode_sub_idx', 'bus_id', 'seq', 'reward']], 
                      on=['episode_sub_idx', 'bus_id', 'seq'], how='inner')
    if merged.empty: return pd.DataFrame()
    
    merged = merged.sort_values(['episode_sub_idx', 'bus_id', 'seq'], ascending=[True, True, False])
    
    def compute_returns_for_group(grp):
        r = grp['reward'].values
        qs = np.zeros_like(r)
        curr = 0
        for i in range(len(r)):
            curr = r[i] + gamma * curr
            qs[i] = curr
        grp['q_real'] = qs
        return grp

    merged = merged.groupby(['episode_sub_idx', 'bus_id'], group_keys=False).apply(compute_returns_for_group)
    
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
    df = df.copy()
    df['rareness'] = np.nan
    groups = df.groupby(group_cols)
    for name, grp in tqdm(groups, desc="Computing Mahalanobis"):
        idx = grp.index
        X = grp[feature_cols].values
        if len(X) < 3:
            df.loc[idx, 'rareness'] = 0.0
            continue
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            df.loc[idx, 'rareness'] = 0.0
            continue
        # Vectorized internal loop would be better but this is fine for subsampled data
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

def plot_comparison_repro(df_ens, df_sac, output_path):
    print("Aligning & Computing Oracle Errors...")
    df_ens['error'], reg_ens = get_aligned_error(df_ens)
    df_sac['error'], reg_sac = get_aligned_error(df_sac)
    
    df_ens['error_oracle'] = calculate_best_head_error(df_ens, reg_ens)
    df_sac['error_oracle'] = calculate_best_head_error(df_sac, reg_sac)
    
    df_ens['q_pred_oracle'] = df_ens['q_real'] + df_ens['error_oracle']
    df_sac['q_pred_oracle'] = df_sac['q_real'] + df_sac['error_oracle']

    num_bins = 30
    r_clip = max(np.percentile(df_ens['rareness'].dropna(), 99), 
                 np.percentile(df_sac['rareness'].dropna(), 99))

    df_ens['rare_bin'] = pd.cut(df_ens['rareness'].clip(upper=r_clip), bins=num_bins, labels=False)
    df_sac['rare_bin'] = pd.cut(df_sac['rareness'].clip(upper=r_clip), bins=num_bins, labels=False)

    ens_mae = df_ens.groupby('rare_bin')['error_oracle'].apply(lambda x: x.abs().mean())
    sac_mae = df_sac.groupby('rare_bin')['error_oracle'].apply(lambda x: x.abs().mean())
    ens_q_pred = df_ens.groupby('rare_bin')['q_pred_oracle'].mean()
    sac_q_pred = df_sac.groupby('rare_bin')['q_pred_oracle'].mean()
    q_real_mean = df_ens.groupby('rare_bin')['q_real'].mean()
    ens_density = df_ens.groupby('rare_bin').size()
    sac_density = df_sac.groupby('rare_bin').size()

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Top: Q-Values
    ax0.plot(q_real_mean.index, q_real_mean.values, 'k--', linewidth=2, label='Ground Truth (Real Q)')
    ax0.plot(ens_q_pred.index, ens_q_pred.values, color='#4488FF', linewidth=2.5, marker='o', markersize=5, label='Ensemble (Oracle Best Predict)')
    ax0.plot(sac_q_pred.index, sac_q_pred.values, color='orange', linewidth=2.5, marker='x', label='Vanilla SAC (Oracle Best Predict)')
    ax0.set_title('Mean Q-Value Predictions vs Ground Truth (20eps Aggregate)', fontsize=13, fontweight='bold')
    ax0.set_ylabel('Q-Value', fontsize=12)
    ax0.legend(loc='upper left')
    ax0.grid(alpha=0.3)

    # Bottom: MAE & Density
    ax2 = ax1.twinx()
    bar_width = 0.35
    ax2.bar(ens_density.index - bar_width/2, ens_density.values, width=bar_width, alpha=0.15, color='blue', label='Ensemble Count')
    ax2.bar(sac_density.index + bar_width/2, sac_density.values, width=bar_width, alpha=0.15, color='orange', label='Vanilla Count')
    ax2.set_ylabel('Data Count per Bin', color='gray')
    
    ax1.plot(sac_mae.index, sac_mae.values, label='Vanilla SAC MAE', color='orange', linewidth=2.5, marker='x', zorder=5)
    ax1.plot(ens_mae.index, ens_mae.values, label='Ensemble MAE', color='#4488FF', linewidth=2.5, marker='o', markersize=5, zorder=5)
    ax1.set_title('Oracle Q-Error (MAE) & Data Density', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Mahalanobis Rareness Bin (Low → High)', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error of Best Head', fontsize=12)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    DATA_DIR = 'offline_timeline_data_20eps'
    # Loading Ensemble/SAC
    df_ens = load_data_subsampled(DATA_DIR, 'ensemble', step=20)
    df_sac = load_data_subsampled(DATA_DIR, 'sac', step=20)
    
    print("\nComputing Mahalanobis Rareness...")
    df_ens = compute_mahalanobis_rareness(df_ens)
    df_sac = compute_mahalanobis_rareness(df_sac)
    
    OUT_DIR = 'analysis_results'
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_comparison_repro(df_ens, df_sac, os.path.join(OUT_DIR, 'comparison_mahalanobis_rareness_repro_v3.png'))
