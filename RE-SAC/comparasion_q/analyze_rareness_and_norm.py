import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} records from {path}")
    return data

def process_trajectories(data, gamma=0.99):
    """
    Reconstruct trajectories and calculate Q_real (discounted return).
    Data is a list of events (dicts).
    """
    # Group by (Episode, BusID)
    # We need to sort by time within each group
    df = pd.DataFrame(data)
    
    # Filter valid events
    df = df.sort_values(['episode', 'bus_id', 'time'])
    
    results = []
    
    # Iterate over each bus trajectory in each episode
    grouped = df.groupby(['episode', 'bus_id'])
    
    for (ep, bus), group in tqdm(grouped, desc="Processing Trajectories"):
        # group contains 'predict' and 'reward' events
        # We need to match predict[t] with reward[t], reward[t+1]...
        
        events = group.to_dict('records')
        
        # We need to forward-calculate returns
        # But rewards are received AFTER actions.
        # A 'predict' event at T is followed by 'reward' events at T+1, T+2...
        # Wait, in this env, 'reward' is received when a bus arrives at next stop.
        # The 'predict' happens when bus leaves a stop.
        # So: Predict -> (driving) -> Reward -> Predict -> ...
        # The sequence should be P, R, P, R...
        # Let's align them.
        
        preds = [e for e in events if e.get('event') == 'predict']
        rewards = [e for e in events if e.get('event') == 'reward']
        
        # We assume 1-to-1 mapping roughly? 
        # Actually, if done=True, we might have a reward without next predict.
        # Let's align by index if possible, or just strict sequence.
        
        # Simple Discounted Sum:
        # For prediction at index i, Q_real = r_i + gamma * r_{i+1} + ...
        
        # Extract reward values
        r_values = [r['reward'] for r in rewards]
        
        # Calculate Q_real for each prediction
        # We assume prediction i leads to reward i.
        # (Check logic: Predict -> Step -> RewardDict. Yes.)
        
        m = len(preds)
        n = len(r_values)
        count = min(m, n)
        
        # If simulation cut off (3000 steps), we might have missing tail rewards.
        # We treat them as 0 for now (or bootstrapping if we had V(s')). 
        # Since we want Pure Q (Monte Carlo), we just sum what we have.
        
        q_reals = []
        for i in range(count):
            q = 0
            discount = 1.0
            for j in range(i, n):
                q += r_values[j] * discount
                discount *= gamma
                if discount < 1e-6: # Truncate for speed
                    break
            q_reals.append(q)
            
        # Store processed data items
        for i in range(count):
            rec = preds[i]
            rec['q_real'] = q_reals[i]
            rec['q_pred_mean'] = np.mean(rec['q_vals'])
            rec['q_pred_std'] = np.std(rec['q_vals'])
            rec['q_pred_min'] = np.min(rec['q_vals'])
            # Keeping raw q_vals if needed
            results.append(rec)
            
    return pd.DataFrame(results)

def align_scales(df, target_col='q_real', pred_col='q_pred_mean'):
    """
    Fit linear regression: Q_real ~ a * Q_pred + b
    Return aligned Q_pred.
    """
    from sklearn.linear_model import LinearRegression
    X = df[pred_col].values.reshape(-1, 1)
    y = df[target_col].values
    
    reg = LinearRegression().fit(X, y)
    print(f"Alignment: Q_real = {reg.coef_[0]:.4f} * Q_pred + {reg.intercept_:.4f}")
    
    aligned = reg.predict(X)
    return aligned, reg

def analyze_and_plot(ens_file, sac_file):
    # 1. Load and Process
    print("Processing Ensemble Data...")
    df_ens = process_trajectories(load_data(ens_file))
    print("Processing SAC Data...")
    df_sac = process_trajectories(load_data(sac_file))
    
    # 2. Alignment
    # Align Ensemble Q-pred to Real Q
    df_ens['q_pred_aligned'], reg_ens = align_scales(df_ens, 'q_real', 'q_pred_mean')
    df_ens['error_aligned'] = df_ens['q_pred_aligned'] - df_ens['q_real']
    df_ens['abs_error_aligned'] = np.abs(df_ens['error_aligned'])
    
    # Align SAC Q-pred to Real Q (using its own best fit? or same as Ensemble?)
    # To be fair, if we claim Ensemble is better *structurally*, we should allow SAC to scale itself too.
    # Otherwise we just compare normalization constants.
    df_sac['q_pred_aligned'], reg_sac = align_scales(df_sac, 'q_real', 'q_pred_mean')
    df_sac['error_aligned'] = df_sac['q_pred_aligned'] - df_sac['q_real']
    df_sac['abs_error_aligned'] = np.abs(df_sac['error_aligned'])
    
    # 3. Rareness Analysis (Ensemble Only? Or apply Ensemble Uncertainty to SAC?)
    # "Rareness" is defined by Ensemble Variance (Std).
    # We can bin the data by 'q_pred_std'.
    
    # We use df_ens because it has the 'q_pred_std' (uncertainty) metric naturally.
    # To compare SAC, we need to know if SAC errors are correlated with *Ensemble's* uncertainty?
    # Or just analyze Ensemble's behavior?
    # The user asked: "Over/under estimation is correlated with rareness".
    # So we plot: Error vs Uncertainty.
    
    plt.figure(figsize=(12, 6))
    
    # Binning Uncertainty
    df_ens['uncertainty_bin'] = pd.qcut(df_ens['q_pred_std'], q=10, labels=False)
    
    sns.lineplot(data=df_ens, x='uncertainty_bin', y='error_aligned', label='Ensemble Signed Error', marker='o')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Uncertainty (Rareness) Decile')
    plt.ylabel('Aligned Prediction Error (Pred - Real)')
    plt.title('Ensemble: Overestimation vs Uncertainty')
    plt.savefig('rareness_vs_error_ensemble.png')
    plt.close()
    
    # 4. Comparative Plot of Error Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_ens['error_aligned'], label=f"Ensemble (MAE={df_ens['abs_error_aligned'].mean():.2f})", fill=True)
    sns.kdeplot(df_sac['error_aligned'], label=f"Vanilla SAC (MAE={df_sac['abs_error_aligned'].mean():.2f})", fill=True)
    plt.xlabel('Aligned Prediction Error (Pred - Real)')
    plt.title('Error Distribution: Ensemble vs Vanilla SAC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-5000, 5000) # Zoom in
    plt.savefig('error_distribution_comparison.png')
    plt.close()

if __name__ == '__main__':
    analyze_and_plot('offline_data_ensemble.pkl', 'offline_data_sac.pkl')
