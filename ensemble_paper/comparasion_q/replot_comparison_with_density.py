"""
Re-plot comparison_rareness_error.png with data density bars.
Loads existing offline dataset, computes Oracle best-head error,
and adds a secondary y-axis showing data count per uncertainty bin.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

def load_all_data(data_dir, prefix):
    files = sorted(glob.glob(os.path.join(data_dir, f'data_{prefix}_*.pkl')))
    print(f"Loading {len(files)} files for {prefix}...")
    all_results = []
    for f in tqdm(files, desc=f"Loading {prefix}"):
        with open(f, 'rb') as fh:
            chunk = pickle.load(fh)
        ckpt_idx = int(os.path.basename(f).split('_')[-1].replace('.pkl', ''))
        norm_stats = chunk.get('norm_stats', None)
        events = chunk.get('data', chunk.get('events', []))
        results = process_chunk(events, ckpt_idx, norm_stats)
        all_results.extend(results)
    return pd.DataFrame(all_results)

def process_chunk(events, ckpt_idx, norm_stats, gamma=0.99):
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
        running_q = 0
        q_reals = [0] * n
        for i in range(n-1, -1, -1):
            running_q = rewards[i]['reward'] + gamma * running_q
            q_reals[i] = running_q
        for i in range(n):
            rec = preds[i]
            q_vals = np.array(rec['q_vals'])
            results.append({
                'checkpoint': ckpt_idx,
                'q_pred_mean': np.mean(q_vals),
                'q_pred_std': np.std(q_vals),
                'q_pred_min': np.min(q_vals),
                'q_real': q_reals[i],
                'q_vals': q_vals.tolist()
            })
    return results

def main():
    DATA_DIR = 'offline_dataset_full'
    OUTPUT_DIR = 'analysis_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Ensemble Data...")
    df_ens = load_all_data(DATA_DIR, 'ensemble')
    print("Loading SAC Data...")
    df_sac = load_all_data(DATA_DIR, 'sac')

    from sklearn.linear_model import LinearRegression

    def get_aligned_error(df, pred_col='q_pred_mean'):
        X = df[pred_col].values.reshape(-1, 1)
        y = df['q_real'].values
        reg = LinearRegression().fit(X, y)
        aligned_pred = reg.predict(X)
        return aligned_pred - y, reg

    print("Aligning Ensemble...")
    df_ens['error'], reg_ens = get_aligned_error(df_ens)
    print(f"Ens Align: Coef={reg_ens.coef_[0]:.2f}")

    print("Aligning SAC...")
    df_sac['error'], reg_sac = get_aligned_error(df_sac)
    print(f"SAC Align: Coef={reg_sac.coef_[0]:.2f}")

    num_bins = 50
    df_ens['unc_bin'] = pd.qcut(df_ens['q_pred_std'], num_bins, labels=False, duplicates='drop')
    try:
        df_sac['unc_bin'] = pd.qcut(df_sac['q_pred_std'], num_bins, labels=False, duplicates='drop')
        sac_has_unc = True
    except:
        sac_has_unc = False

    if not sac_has_unc:
        print("SAC has no valid uncertainty bins. Cannot plot comparison.")
        return

    def calculate_best_head_error(df, reg):
        errs = []
        alpha = reg.coef_[0]
        beta = reg.intercept_
        q_vals_col = df['q_vals'].values
        q_reals = df['q_real'].values
        for q_heads, q_r in zip(q_vals_col, q_reals):
            aligned_heads = np.array(q_heads) * alpha + beta
            diffs = aligned_heads - q_r
            abs_diffs = np.abs(diffs)
            min_idx = np.argmin(abs_diffs)
            errs.append(diffs[min_idx])
        return np.array(errs)

    print("Calculating Oracle Errors...")
    df_sac['error_oracle'] = calculate_best_head_error(df_sac, reg_sac)
    df_ens['error_oracle'] = calculate_best_head_error(df_ens, reg_ens)

    # Group by Uncertainty Bin
    sac_oracle_mae = df_sac.groupby('unc_bin')['error_oracle'].apply(lambda x: x.abs().mean())
    ens_oracle_mae = df_ens.groupby('unc_bin')['error_oracle'].apply(lambda x: x.abs().mean())

    # Data density (count per bin)
    ens_density = df_ens.groupby('unc_bin').size()

    # Plot
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Density bars on secondary y-axis (behind)
    ax2 = ax1.twinx()
    ax2.bar(ens_density.index, ens_density.values, alpha=0.18, color='gray', label='Data Count per Bin', width=0.8)
    ax2.set_ylabel('Data Count per Bin', color='gray', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='gray')

    # MAE lines on primary y-axis (in front)
    ax1.plot(sac_oracle_mae.index, sac_oracle_mae.values, label='Vanilla SAC (Best of 2)',
             color='orange', linewidth=2.5, marker='x', zorder=5)
    ax1.plot(ens_oracle_mae.index, ens_oracle_mae.values, label='Ensemble (Best of 10)',
             color='#4488FF', linewidth=2.5, marker='o', markersize=5, zorder=5)

    ax1.set_title('Comparison: Oracle Accuracy (Error of Best Head) vs Uncertainty', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Uncertainty Decile (Rareness)', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error of Best Head', fontsize=12)
    ax1.axhline(0, color='red', linestyle='-', alpha=0.3)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    ax1.grid(alpha=0.2)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'comparison_rareness_error.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
