"""
Mahalanobis-distance based Rareness analysis for SAC, Ensemble, BAC, and DSAC.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis

# ── Data Loading ──────────────────────────────────────────────────────────────

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
        'q_real': merged['q_real'].values,
        'q_vals': merged['q_vals'].values
    })
    return final_df

def load_and_process_alg(data_dir, prefix, subsample=1):
    files = sorted(glob.glob(os.path.join(data_dir, f'data_{prefix}_*.pkl')))
    if subsample > 1:
        files = files[::subsample]
    print(f"Loading {len(files)} files for {prefix}...")
    dfs = []
    for f in tqdm(files, desc=f"Loading {prefix}"):
        try:
            with open(f, 'rb') as fh:
                chunk = pickle.load(fh)
        except:
            continue
        ckpt_idx = int(os.path.basename(f).split('_')[-1].replace('.pkl', ''))
        events = chunk.get('data', chunk.get('events', []))
        df = process_chunk_vectorized(events, ckpt_idx)
        dfs.append(df)
    if not dfs: return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

# ── Computation ──────────────────────────────────────────────────────────────

def compute_mahalanobis_rareness(df):
    df = df.copy()
    df['rareness'] = np.nan
    group_cols = ['station_id', 'direction']
    feature_cols = ['forward_headway', 'backward_headway']
    
    for name, grp in tqdm(df.groupby(group_cols), desc="Computing Mahalanobis"):
        X = grp[feature_cols].values
        if len(X) < 3:
            df.loc[grp.index, 'rareness'] = 0.0
            continue
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + np.eye(2) * 1e-6
        cov_inv = np.linalg.inv(cov)
        distances = np.array([mahalanobis(x, mean, cov_inv) for x in X])
        df.loc[grp.index, 'rareness'] = distances
    return df

def get_aligned_error(df):
    X = df['q_pred_mean'].values
    y = df['q_real'].values
    x_mean, x_std = X.mean(), X.std() + 1e-8
    y_mean, y_std = y.mean(), y.std() + 1e-8
    aligned_pred = ((X - x_mean) / x_std) * y_std + y_mean
    return aligned_pred - y, (x_mean, x_std, y_mean, y_std)

def calculate_best_head_error(df, align_params):
    x_mean, x_std, y_mean, y_std = align_params
    errs = []
    for q_heads, q_r in zip(df['q_vals'].values, df['q_real'].values):
        aligned = ((np.array(q_heads) - x_mean) / x_std) * y_std + y_mean
        diffs = aligned - q_r
        errs.append(diffs[np.argmin(np.abs(diffs))])
    return np.array(errs)

# ── Processing & Plotting ───────────────────────────────────────────────────

def process_algorithm(data_dir, prefix, subsample=1):
    df = load_and_process_alg(data_dir, prefix, subsample)
    if df.empty: return None
    df = compute_mahalanobis_rareness(df)
    df['error_aligned'], params = get_aligned_error(df)
    df['error_oracle'] = calculate_best_head_error(df, params)
    df['q_pred_oracle'] = df['q_real'] + df['error_oracle']
    return df

def main():
    DATA_DIR = 'offline_timeline_data_20eps'
    OUT_DIR = 'analysis_results_reproduced'
    os.makedirs(OUT_DIR, exist_ok=True)
    
    algs = ['sac', 'ensemble', 'bac', 'dsac']
    # Subsample to speed up (500 files -> 20 files if subsample=25)
    # SAC/Ensemble have 500 each. BAC/DSAC have 100 each.
    results = {}
    for alg in algs:
        sub = 25 if alg in ['sac', 'ensemble'] else 5
        res = process_algorithm(DATA_DIR, alg, subsample=sub)
        if res is not None:
            results[alg] = res
            
    if not results:
        print("No data loaded!")
        return

    # Binning and Plotting
    num_bins = 20
    all_rareness = pd.concat([df['rareness'] for df in results.values()])
    r_clip = np.percentile(all_rareness.dropna(), 99)
    
    colors = {
        'sac': 'orange',
        'ensemble': '#4488FF',
        'bac': 'mediumseagreen',
        'dsac': 'indianred'
    }
    labels = {
        'sac': 'Vanilla SAC',
        'ensemble': 'Ensemble (10 heads)',
        'bac': 'BAC (Twin Q)',
        'dsac': 'DSAC (20 Quantiles)'
    }

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    for alg, df in results.items():
        df['rareness_clipped'] = df['rareness'].clip(upper=r_clip)
        df['rare_bin'] = pd.cut(df['rareness_clipped'], bins=num_bins, labels=False)
        
        bin_stats = df.groupby('rare_bin').agg({
            'q_real': 'mean',
            'q_pred_oracle': 'mean',
            'error_oracle': lambda x: x.abs().mean(),
            'rareness': 'count'
        }).rename(columns={'error_oracle': 'mae', 'rareness': 'count'})
        
        # Plot Q-values
        if alg == 'sac': # Ground truth 
            ax0.plot(bin_stats.index, bin_stats['q_real'], 'k--', alpha=0.5, label='Ground Truth')
            
        ax0.plot(bin_stats.index, bin_stats['q_pred_oracle'], color=colors[alg], 
                 marker='o', markersize=4, label=f'{labels[alg]} (Oracle)')
        
        # Plot MAE
        ax1.plot(bin_stats.index, bin_stats['mae'], color=colors[alg], 
                 linewidth=2, marker='s', markersize=5, label=f'{labels[alg]} MAE')

    ax0.set_title('Q-Value Prediction Accuracy by Rareness', fontsize=14, fontweight='bold')
    ax0.set_ylabel('Q-Value', fontsize=12)
    ax0.legend()
    ax0.grid(alpha=0.3)
    
    ax1.set_title('Oracle Q-Error (MAE)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mahalanobis Rareness Bin (Low → High)', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'comparison_mahalanobis_rareness_reproduced_final.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved final plot to {out_path}")

if __name__ == '__main__':
    main()
