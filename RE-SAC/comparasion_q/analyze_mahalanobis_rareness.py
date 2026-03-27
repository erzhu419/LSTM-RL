"""
Mahalanobis-distance based Rareness analysis for 4 Algorithms (SAC, Ensemble, BAC, DSAC-v1).

Rareness = Mahalanobis distance of (forward_headway, backward_headway)
           conditioned on (station_id, direction).
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
from multiprocessing import Pool, cpu_count


# ── Data Loading ──────────────────────────────────────────────────────────────

def _load_one_file(args):
    """Top-level worker function (required for multiprocessing pickling)."""
    f, ckpt_idx = args
    try:
        with open(f, 'rb') as fh:
            chunk = pickle.load(fh)
        events = chunk.get('data', chunk.get('events', []))
        return process_chunk_vectorized(events, ckpt_idx)
    except Exception as e:
        print(f"Skipping corrupted file {f}: {e}", flush=True)
        return pd.DataFrame()


def load_all_data(data_dir, prefix, subsample_step=5, n_workers=None):
    files = sorted(glob.glob(os.path.join(data_dir, f'data_{prefix}_*.pkl')))
    if len(files) == 0:
        return pd.DataFrame()

    # Standardize to roughly the same number of data points
    if len(files) > 100:
        files = files[::subsample_step]

    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    print(f"Loading {len(files)} files for {prefix} with {n_workers} workers...", flush=True)
    args = [
        (f, int(os.path.basename(f).split('_')[-1].replace('.pkl', '')))
        for f in files
    ]

    dfs = []
    with Pool(processes=n_workers) as pool:
        for df in tqdm(pool.imap_unordered(_load_one_file, args, chunksize=4),
                       total=len(args), desc=f"Loading {prefix}"):
            if not df.empty:
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
    
    # High performance vectorization to avoid Pandas apply() blowing up memory over millions of rows
    q_vals_array = np.array(merged['q_vals'].tolist())
    q_pred_mean = np.mean(q_vals_array, axis=1)
    q_pred_std = np.std(q_vals_array, axis=1)
    
    final_df = pd.DataFrame({
        'checkpoint': ckpt_idx,
        'station_id': states[:, 1].astype(int),
        'direction': states[:, 3].astype(int),
        'forward_headway': states[:, 4].astype(float),
        'backward_headway': states[:, 5].astype(float),
        'q_pred_mean': q_pred_mean,
        'q_pred_std': q_pred_std,
        'q_real': merged['q_real'].values,
        'q_vals': merged['q_vals'].values
    })
    
    return final_df


# ── Mahalanobis Computation ──────────────────────────────────────────────────

def compute_mahalanobis_rareness(df, feature_cols=['forward_headway', 'backward_headway'],
                                  group_cols=['station_id', 'direction']):
    if df.empty: return df
    # IMPORTANT: Do NOT use df.copy() here — that doubles peak memory on large datasets.
    # Instead modify in-place (rareness is a new column, no existing data is overwritten).
    df['rareness'] = 0.0

    for name, grp in tqdm(df.groupby(group_cols), desc="Computing Mahalanobis"):
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

        # Vectorized Mahalanobis distance calculation to avoid OOM and speed up
        diff = X - mean
        distances = np.sqrt(np.sum(np.dot(diff, cov_inv) * diff, axis=1))
        df.loc[idx, 'rareness'] = distances

    return df


# ── Alignment & Oracle Error ─────────────────────────────────────────────────

def get_aligned_error_variance_matching(df, pred_col='q_pred_mean'):
    """Variance Matching alignment to avoid flat lines."""
    X = df[pred_col].values
    y = df['q_real'].values
    
    x_mean = X.mean()
    x_std = X.std() + 1e-8
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    
    aligned_pred = ((X - x_mean) / x_std) * y_std + y_mean
    return aligned_pred - y, (x_mean, x_std, y_mean, y_std)


def calculate_best_head_error(df, align_params):
    errs = []
    x_mean, x_std, y_mean, y_std = align_params
    
    for q_heads, q_r in zip(df['q_vals'].values, df['q_real'].values):
        aligned = ((np.array(q_heads) - x_mean) / x_std) * y_std + y_mean
        diffs = aligned - q_r
        # Oracle: pick closest head
        errs.append(diffs[np.argmin(np.abs(diffs))])
    return np.array(errs)

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_all_sequential(data_dir, output_dir='analysis_results'):
    import gc
    import pickle
    os.makedirs(output_dir, exist_ok=True)
    bin_stats_all = {}  # accumulate per-algo binned stats for later replotting
    
    algos = [
        ('ensemble', 1),
        ('sac', 1),
        ('dsac', 1),
        ('bac', 5)
    ]
    
    labels = {
        'ensemble': 'Ensemble SAC',
        'sac': 'Vanilla SAC',
        'bac': 'BAC',
        'dsac': 'DSAC-v1'
    }
    
    colors = {
        'ensemble': '#4488FF', 
        'sac': 'orange',
        'bac': '#2ca02c', # Green
        'dsac': '#d62728' # Red
    }
    
    markers = {
        'ensemble': 'o',
        'sac': 's',
        'bac': '^',
        'dsac': 'D'
    }

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(18, 16), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    ax2 = ax1.twinx()
    
    bar_width_ratio = 0.8 / len(algos)
    offsets = np.linspace(-bar_width_ratio * (len(algos) - 1) / 2, bar_width_ratio * (len(algos) - 1) / 2, len(algos))
    
    global_r_clip = 6.0
    num_bins = 30
    bin_edges = np.linspace(0, global_r_clip, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    ground_truth_plotted = False

    for i, (name, subsample) in enumerate(algos):
        print(f"\n==============================================")
        print(f"               Processing {name.upper()}")
        print(f"==============================================")

        # ── Per-algo cache: skip heavy compute if already done ──
        cache_path = os.path.join(output_dir, f'_cache_{name}.pkl')
        if os.path.exists(cache_path):
            print(f"[{name.upper()}] Cache found! Loading from {cache_path}")
            with open(cache_path, 'rb') as _f:
                cached = pickle.load(_f)
            mae_data      = cached['mae']
            q_pred_data   = cached['q_pred']
            density_data  = cached['density']
            q_real_cached = cached.get('q_real_mean')  # only for first algo
        else:
            df = load_all_data(data_dir, name, subsample_step=subsample)
            if df.empty:
                print(f"Skipping {name} - Dataframe is empty")
                continue

            print(f"[{name.upper()}] Computing Rareness (Mahalanobis)...")
            df = compute_mahalanobis_rareness(df)

            print(f"[{name.upper()}] Aligning (Variance Matching)...")
            df['error'], params = get_aligned_error_variance_matching(df)

            print(f"[{name.upper()}] Computing Oracle Errors...")
            df['error_oracle'] = calculate_best_head_error(df, params)
            df['q_pred_oracle'] = df['q_real'] + df['error_oracle']
            # Free the large q_vals column (list-of-arrays, several GB) immediately after use
            df.drop(columns=['q_vals', 'q_pred_mean', 'q_pred_std', 'error'], inplace=True, errors='ignore')

            print(f"\n{name.upper()} Overall Oracle MAE = {df['error_oracle'].abs().mean():.2f}")

            # Binning
            df['rareness_clipped'] = df['rareness'].clip(upper=global_r_clip)
            df['rare_bin'] = pd.cut(df['rareness_clipped'], bins=bin_edges, labels=False, include_lowest=True)

            mae_data      = df.groupby('rare_bin')['error_oracle'].apply(lambda x: x.abs().mean())
            q_pred_data   = df.groupby('rare_bin')['q_pred_oracle'].mean()
            density_data  = df.groupby('rare_bin').size()
            q_real_cached = df.groupby('rare_bin')['q_real'].mean() if not ground_truth_plotted else None

            # Save per-algo cache immediately
            with open(cache_path, 'wb') as _f:
                pickle.dump({
                    'mae': mae_data,
                    'q_pred': q_pred_data,
                    'density': density_data,
                    'total_count': len(df),
                    'q_real_mean': q_real_cached,
                }, _f)
            print(f"[{name.upper()}] Cache saved to {cache_path}")

            del df
            gc.collect()

        print(f"[{name.upper()}] Plotting curves...")
        # Plot ground truth once (from first algo)
        if not ground_truth_plotted and q_real_cached is not None:
            idx = q_real_cached.index.values
            ax0.plot(bin_centers[idx], q_real_cached.values, 'k--', linewidth=2.5, label='Ground Truth (Real Q)')
            ground_truth_plotted = True
            bin_stats_all['ground_truth'] = q_real_cached

        # Plot algorithm curves
        idx_pred = q_pred_data.index.values
        ax0.plot(bin_centers[idx_pred], q_pred_data.values, color=colors[name], linewidth=2.5, marker=markers[name], markersize=6, label=f'{labels[name]}')

        idx_mae = mae_data.index.values
        ax1.plot(bin_centers[idx_mae], mae_data.values, color=colors[name], linewidth=2.5, marker=markers[name], markersize=6, label=f'{labels[name]} MAE', zorder=5)

        idx_den = density_data.index.values
        # Normalize density to percentage of total data for fair comparison
        total_data_points = density_data.sum()
        density_percentage = (density_data.values / total_data_points) * 100 if total_data_points > 0 else density_data.values
        
        ax2.bar(bin_centers[idx_den] + offsets[i] * bin_width, density_percentage, width=bar_width_ratio * bin_width, alpha=0.15, color=colors[name], label=f'{labels[name]} Density (%)')

        # Accumulate for combined pkl
        bin_stats_all[name] = {'mae': mae_data, 'q_pred': q_pred_data, 'density': density_data, 'total_count': total_data_points}

        print(f"[{name.upper()}] Finished. Memory released.")
        del mae_data, q_pred_data, density_data
        gc.collect()

    # Formatting 
    print("\nFormatting and saving final plot...")
    ax0.set_title('Mean Q-Value Predictions vs Ground Truth (Variance Matching Alignment)', fontsize=20, fontweight='bold')
    ax0.set_ylabel('Q-Value', fontsize=16)
    ax0.legend(loc='upper right', fontsize=14)
    ax0.tick_params(labelsize=13, labelbottom=True)
    ax0.grid(alpha=0.3)
    
    ax2.set_ylabel('Data Density (%)', color='gray', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=13)
    
    ax1.set_title('Oracle Q-Error (MAE) & Data Density', fontsize=20, fontweight='bold')
    ax1.set_xlabel(f'Mahalanobis Rareness (Clipped at {global_r_clip})', fontsize=16)
    ax1.set_ylabel('Mean Absolute Error of Best Head', fontsize=16)
    ax1.tick_params(labelsize=13)
    ax1.axhline(0, color='red', linestyle='-', alpha=0.3)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'comparison_mahalanobis_rareness_reproduced_final.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Success! Saved final comparison plot to {out_path}")

    # Save bin stats so we can replot cheaply without reprocessing data
    stats_path = os.path.join(output_dir, 'bin_stats_all.pkl')
    meta = {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'bin_width': bin_width,
        'global_r_clip': global_r_clip,
        'num_bins': num_bins,
        'algos': [a for a, _ in algos],
        'labels': labels,
        'colors': colors,
        'markers': markers,
        'bar_width_ratio': bar_width_ratio,
    }
    with open(stats_path, 'wb') as f:
        pickle.dump({'meta': meta, 'stats': bin_stats_all}, f)
    print(f"Saved bin stats cache to {stats_path} (for cheap replotting)")

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Use the directory specifically requested by the user
    DATA_DIR = 'offline_timeline_data_20eps'
    plot_all_sequential(DATA_DIR)
