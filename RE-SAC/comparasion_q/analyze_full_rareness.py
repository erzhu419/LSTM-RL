import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import glob

def load_all_data(data_dir, prefix):
    # Load all files matching prefix
    pattern = os.path.join(data_dir, f"data_{prefix}_*.pkl")
    files = glob.glob(pattern)
    print(f"Loading {len(files)} files for {prefix}...")
    
    all_records = []
    
    # We want to enable analysis PER CHECKPOINT or AGGREGATED.
    # User wants: "Rareness vs Error"
    # This relationship should hold across time, or we can analyze it for specific epochs.
    # Let's aggregated for now, but keep checkpoint info.
    
    for f_path in tqdm(files):
        with open(f_path, 'rb') as f:
            pkg = pickle.load(f)
            # pkg = {'checkpoint_idx', 'model_type', 'norm_stats', 'data'}
            
            # Extract data
            events = pkg['data']
            ckpt = pkg['checkpoint_idx']
            
            # We need to process trajectories here to save memory (don't keep raw events)
            # Process into tuples of (Q_pred, Q_real, Rareness)
            processed = process_chunk(events, ckpt, pkg['norm_stats'])
            all_records.extend(processed)
            
    return pd.DataFrame(all_records)

def process_chunk(events, ckpt_idx, norm_stats, gamma=0.99):
    # Determine normalization scaling if needed
    # If model output is normalized, we might want to analyze "Raw Q" vs "Real Q" (both raw or both norm).
    # User said: "Record norm stats to convenience restore".
    # We will compute Real Q (Sum of rewards). This is RAW.
    # Q_pred is from network.
    # If network learns Normalized Return, Q_pred is small.
    # We should align them or denormalize Q_pred.
    # Let's Denormalize Q_pred if we have stats?
    # Actually, RewardScaling in SAC divides reward by std.
    # So Q_pred ~ Q_real / RewardStd.
    # We don't have RewardStd easily (it's in RewardScaling object, not StateNorm).
    # Wait, the user said "record ensemble size q... normalization data".
    # In `collect_offline_data`, I saved `state_norm`. I did NOT save `reward_scaling` stats.
    # Because `RewardScaling` is inside `SAC_Trainer`, not usually saved in checkpoint dict explicitly unless we look deep.
    # However, we can just use the Linear Alignment method from before. It's robust.
    
    df = pd.DataFrame(events)
    if df.empty:
        return []

    # Sort
    df = df.sort_values(['episode_sub_idx', 'bus_id', 'time'])
    
    results = []
    grouped = df.groupby(['episode_sub_idx', 'bus_id'])
    
    for _, group in grouped:
        evts = group.to_dict('records')
        preds = [e for e in evts if e['event'] == 'predict']
        rewards = [e for e in evts if e['event'] == 'reward']
        
        # Match P -> R sequence
        # Assuming tight coupling (Predict -> Step -> Reward)
        # We might have mismatches if episodes truncated.
        # Let's assume index i of pred matches index i of reward.
        n = min(len(preds), len(rewards))
        
        # Compute Q_real (Monte Carlo)
        # Reverse iteration for efficiency
        running_q = 0
        q_reals = [0] * n
        for i in range(n-1, -1, -1):
            running_q = rewards[i]['reward'] + gamma * running_q
            q_reals[i] = running_q
            
        for i in range(n):
            rec = preds[i]
            q_vals = np.array(rec['q_vals']) # [EnsSize] or [2]
            
            # Rareness Metrics
            q_mean = np.mean(q_vals)
            q_std = np.std(q_vals)
            q_min = np.min(q_vals)
            
            # Store
            results.append({
                'checkpoint': ckpt_idx,
                'q_pred_mean': q_mean,
                'q_pred_std': q_std, # This is Rareness for Ensemble
                'q_pred_min': q_min,
                'q_real': q_reals[i],
                'q_vals': q_vals.tolist() # Keep raw list for Oracle calculation
                # 'state_norm_mean': norm_stats['mean'] if norm_stats else None 
            })
            
    return results

def plot_analysis(df_ens, df_sac, output_dir='analysis_results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Alignment (Per Checkpoint or Global? Global is safer for trend)
    # Align Ensemble
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
    
    # 2. Rareness vs Error BOOTSTRAP
    # We want to show: As Rareness Increases -> SAC Overestimates, Ensemble doesn't.
    # Rareness = Ensemble Std.
    # BUT, we need to know the "Ensemble Std" for the SAC points?
    # We can't know that unless we ran the Ensemble Model on the SAC Trajectories.
    # The current data is:
    # - Ens Data: (S_ens, A_ens) -> Q_ens_std, Error_ens
    # - SAC Data: (S_sac, A_sac) -> Q_sac_std (only 2 heads), Error_sac
    
    # User assumption: "Rareness... including state... tuple".
    # If we assume Q_sac_std (from 2 heads) is a proxy? It's very noisy.
    # However, let's try to plot what we have.
    # "Ensemble size q values... record ensemble ... purpose is to compare".
    # User might accept "Ensemble Q_std" as metric for Ensemble, and "SAC Q_std" for SAC?
    # Or maybe we just show Ensemble's behavior internally?
    # "SAC method overestimates... Ensemble accurate".
    # Let's plot Error vs "Internal Uncertainty" for both.
    
    # Binning
    # We create bins based on Uncertainty percentiles
    num_bins = 50
    
    # Ensemble Plot
    df_ens['unc_bin'] = pd.qcut(df_ens['q_pred_std'], num_bins, labels=False, duplicates='drop')
    
    # SAC Plot (using its own deviation as proxy)
    try:
        df_sac['unc_bin'] = pd.qcut(df_sac['q_pred_std'], num_bins, labels=False, duplicates='drop')
        sac_has_unc = True
    except:
        sac_has_unc = False
        
    plt.figure(figsize=(12, 7))
    
    # Background Scatter for density (Sampled to avoid extreme lag)
    sample_size = min(len(df_ens), 50000)
    sns.scatterplot(data=df_ens.sample(sample_size), x='unc_bin', y='error', 
                    alpha=0.05, color='blue', edgecolor=None, label='_nolegend_')
    
    # Aggregate Mean Error by Bin
    ens_grp = df_ens.groupby('unc_bin')['error'].mean()
    sns.lineplot(x=ens_grp.index, y=ens_grp.values, label='Ensemble (Mean Q)', color='blue', linewidth=2.5, marker='o')
    
    # Also plot Min Q error for Ensemble
    df_ens['error_min'], _ = get_aligned_error(df_ens, 'q_pred_min')
    ens_min_grp = df_ens.groupby('unc_bin')['error_min'].mean()
    sns.lineplot(x=ens_min_grp.index, y=ens_min_grp.values, label='Ensemble (Min Q)', color='cyan', linestyle='--', marker='s')
    
    plt.title(f'Ensemble: Prediction Error vs Uncertainty ({num_bins} Bins)')
    plt.xlabel('Uncertainty Decile (Low -> High)')
    plt.ylabel('Aligned Error (Pred - Real)')
    plt.axhline(0, color='red', linestyle='-', alpha=0.3)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(output_dir, 'ensemble_rareness_error.png'), dpi=300)
    plt.close()
    
    # Combined Plot - Primary Comparison: "Oracle" Best Head Error
    # User Request: "Take the one among the 10 that has the minimum error distance to q_real"
    # This checks if the Truth is contained within the Ensemble distribution.
    
    if sac_has_unc:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # We need to compute "Best Head Error" for both
        # 1. Get Alignment params from Means (already computed: reg_ens, reg_sac)
        # 2. Iterate and find best head per sample
        
        def calculate_best_head_error(df, reg):
            # df has 'q_vals' which is a list of floats
            # We need to vectorize this or iterate efficiently
            # Let's just iterate for clarity as vectorizing list column is messy
            
            errs = []
            alpha = reg.coef_[0]
            beta = reg.intercept_
            
            # Extract lists
            q_vals_col = df['q_vals'].values # Array of lists
            q_reals = df['q_real'].values
            
            for q_heads, q_r in zip(q_vals_col, q_reals):
                # Align all heads
                aligned_heads = np.array(q_heads) * alpha + beta
                
                # Find closest to q_r
                diffs = aligned_heads - q_r
                abs_diffs = np.abs(diffs)
                min_idx = np.argmin(abs_diffs)
                
                # Keep signed error of the best head
                errs.append(diffs[min_idx])
                
            return np.array(errs)

        print("Calculating Oracle Errors...")
        df_sac['error_oracle'] = calculate_best_head_error(df_sac, reg_sac)
        df_ens['error_oracle'] = calculate_best_head_error(df_ens, reg_ens)
        
        # Group by Uncertainty Bin
        sac_oracle_mae = df_sac.groupby('unc_bin')['error_oracle'].apply(lambda x: x.abs().mean())
        ens_oracle_mae = df_ens.groupby('unc_bin')['error_oracle'].apply(lambda x: x.abs().mean())
        
        # Data density (count per bin) — use ensemble bins as reference
        ens_density = df_ens.groupby('unc_bin').size()
        
        # Plot density bars on secondary y-axis (behind everything)
        ax2 = ax1.twinx()
        ax2.bar(ens_density.index, ens_density.values, alpha=0.15, color='gray', label='Data Count per Bin', width=0.8)
        ax2.set_ylabel('Data Count per Bin', color='gray', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Plot MAE lines on primary y-axis (in front)
        ax1.plot(sac_oracle_mae.index, sac_oracle_mae.values, label='Vanilla SAC (Best of 2)', color='orange', linewidth=2.5, marker='x', zorder=5)
        ax1.plot(ens_oracle_mae.index, ens_oracle_mae.values, label='Ensemble (Best of 10)', color='blue', linewidth=2.5, marker='o', zorder=5)
        
        ax1.set_title(f'Comparison: Oracle Accuracy (Error of Best Head) vs Uncertainty')
        ax1.set_xlabel('Uncertainty Decile (Rareness)')
        ax1.set_ylabel('Mean Absolute Error of Best Head')
        ax1.axhline(0, color='red', linestyle='-', alpha=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.grid(alpha=0.2)
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # Make ax1 background transparent so bars show through
        plt.savefig(os.path.join(output_dir, 'comparison_rareness_error.png'), dpi=300)
        plt.close()
        
    # 3. Evolution of Error over Checkpoints
    # Plot Oracle MAE over time for both
    # User Request: "点非常多" (Many points), "没平滑" (No smoothing), "Orange for SAC"
    plt.figure(figsize=(12, 6))
    
    # Check if oracle error exists (it's calculated in Combined Plot section)
    # If not there, we calculate it here.
    if 'error_oracle' not in df_ens.columns:
        print("Calculating Oracle Errors for Evolution Plot...")
        df_ens['error_oracle'] = calculate_best_head_error(df_ens, reg_ens)
        df_sac['error_oracle'] = calculate_best_head_error(df_sac, reg_sac)

    # Group by Checkpoint to get MAE per checkpoint
    ens_time_oracle = df_ens.groupby('checkpoint')['error_oracle'].apply(lambda x: x.abs().mean())
    sac_time_oracle = df_sac.groupby('checkpoint')['error_oracle'].apply(lambda x: x.abs().mean())
    
    # Plot as a noisy line (which looks like many points if dense)
    # Using specific colors: Blue for Ensemble, Orange for SAC
    sns.lineplot(x=ens_time_oracle.index, y=ens_time_oracle.values, label='Ensemble (Oracle/Best-Head)', color='blue', linewidth=1, alpha=0.8)
    sns.lineplot(x=sac_time_oracle.index, y=sac_time_oracle.values, label='Vanilla SAC (Oracle/Best-Head)', color='orange', linewidth=1, alpha=0.8)
    
    plt.xlabel('Training Epoch (Checkpoint)')
    plt.ylabel('Mean Absolute Error (Oracle)')
    plt.title('Q-Accuracy Evolution (Oracle Selection, No Smoothing)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'evolution_mae.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    # Load Data
    # Assuming data collection output dir
    DATA_DIR = 'offline_dataset_full'
    
    print("Loading Ensemble Data...")
    df_ens = load_all_data(DATA_DIR, 'ensemble')
    
    print("Loading SAC Data...")
    df_sac = load_all_data(DATA_DIR, 'sac')
    
    plot_analysis(df_ens, df_sac)
