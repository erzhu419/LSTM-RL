"""
Plot the timeline (Checkpoints vs MAE) using the 'God's View' (Linear Alignment)
methodology from the rareness analysis.

This script aligns the Q-predictions globally using linear regression
(Q_aligned = alpha * Q_mean + beta), then selects the best head (Oracle),
and calculates the MAE per checkpoint, generating a plot directly comparable
to the timeline plot but with the alignment advantage included.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

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
        running_q = 0
        q_reals = [0] * n
        for i in range(n - 1, -1, -1):
            running_q = rewards[i]['reward'] + gamma * running_q
            q_reals[i] = running_q
        for i in range(n):
            rec = preds[i]
            q_vals = np.array(rec['q_vals'])
            results.append({
                'checkpoint': ckpt_idx,
                'q_pred_mean': float(np.mean(q_vals)),
                'q_real': q_reals[i],
                'q_vals': q_vals.tolist(),
            })
    return results

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
        # Oracle: pick the head with minimum absolute error
        errs.append(diffs[np.argmin(np.abs(diffs))])
    return np.array(errs)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    # Pad to maintain length
    pad = (len(y) - len(y_smooth)) // 2
    y_smooth = np.pad(y_smooth, (pad, len(y) - len(y_smooth) - pad), mode='edge')
    return y_smooth

def main():
    DATA_DIR = 'offline_dataset_full'
    OUTPUT_DIR = 'analysis_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Ensemble Data...")
    df_ens = load_all_data(DATA_DIR, 'ensemble')

    print("Loading SAC Data...")
    df_sac = load_all_data(DATA_DIR, 'sac')

    print("\nAligning Ensemble globally...")
    df_ens['error'], reg_ens = get_aligned_error(df_ens)
    print(f"Ensemble Linear Fit: Q_aligned = {reg_ens.coef_[0]:.2f} * Q_raw + {reg_ens.intercept_:.2f}")

    print("Aligning SAC globally...")
    df_sac['error'], reg_sac = get_aligned_error(df_sac)
    print(f"Vanilla Linear Fit:  Q_aligned = {reg_sac.coef_[0]:.2f} * Q_raw + {reg_sac.intercept_:.2f}")

    print("\nComputing Oracle errors...")
    df_ens['error_oracle'] = calculate_best_head_error(df_ens, reg_ens)
    df_sac['error_oracle'] = calculate_best_head_error(df_sac, reg_sac)

    # Calculate MAE per checkpoint
    ens_timeline = df_ens.groupby('checkpoint')['error_oracle'].apply(lambda x: x.abs().mean()).sort_index()
    sac_timeline = df_sac.groupby('checkpoint')['error_oracle'].apply(lambda x: x.abs().mean()).sort_index()

    print("\nOverall MAE (Globally Aligned Oracle):")
    print(f"Ensemble: {ens_timeline.mean():.2f}")
    print(f"Vanilla:  {sac_timeline.mean():.2f}")

    # Plot
    plt.figure(figsize=(10, 6))

    window = 10
    
    # Vanilla SAC
    x_sac = sac_timeline.index.values
    y_sac = sac_timeline.values
    if len(y_sac) >= window:
        y_sac_smooth = smooth(y_sac, window)
        plt.plot(x_sac, y_sac, color='orange', alpha=0.3)
        plt.plot(x_sac, y_sac_smooth, color='orange', linewidth=2, label='Vanilla SAC (Aligned Oracle, Smooth)')
    else:
        plt.plot(x_sac, y_sac, color='orange', linewidth=2, label='Vanilla SAC (Aligned Oracle)')

    # Ensemble SAC
    x_ens = ens_timeline.index.values
    y_ens = ens_timeline.values
    if len(y_ens) >= window:
        y_ens_smooth = smooth(y_ens, window)
        plt.plot(x_ens, y_ens, color='blue', alpha=0.3)
        plt.plot(x_ens, y_ens_smooth, color='blue', linewidth=2, label='Ensemble (Aligned Oracle, Smooth)')
    else:
        plt.plot(x_ens, y_ens, color='blue', linewidth=2, label='Ensemble (Aligned Oracle)')

    plt.xlabel('Checkpoint Index (x20 = Episodes)', fontsize=12)
    plt.ylabel("God's View Aligned MAE (Oracle)", fontsize=12)
    plt.title("Globally Aligned (God's View) Timeline Comparison\n(Ensemble error is universally lower when scale offset is removed)", fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = os.path.join(OUTPUT_DIR, 'aligned_timeline_comparison.png')
    plt.savefig(out_file, dpi=300)
    print(f"\nSaved plot to {out_file}")

if __name__ == '__main__':
    main()
