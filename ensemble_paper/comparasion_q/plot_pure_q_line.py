import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def load_data(results, key):
    if key not in results or not results[key]:
        return None, None, None, None
        
    data = results[key]
    sorted_eps = sorted(data.keys())
    
    q_pred = np.array([data[ep]['q_pred'] for ep in sorted_eps])
    q_real = np.array([data[ep]['q_real'] for ep in sorted_eps])
    episodes = np.array(sorted_eps)
    
    abs_error = np.abs(q_pred - q_real)
    
    return episodes, q_pred, q_real, abs_error

def smooth_data(data, window=5):
    if len(data) < window:
        return data
    # Simple moving average for Real Q (which is high variance)
    # Exponential moving average for Pred Q to show trend
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    # Pad to maintain length
    padding = [smoothed[0]] * (window - 1)
    return np.concatenate([padding, smoothed])

def plot_comparison(results_path, save_dir, smooth_window=10):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
        
    results = np.load(results_path, allow_pickle=True).item()
    
    ens_eps, ens_pred, ens_real, ens_err = load_data(results, 'ensemble')
    van_eps, van_pred, van_real, van_err = load_data(results, 'vanilla')
    
    # Setup Plot
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    colors = {'ens': '#1f77b4', 'van': '#d62728', 'real': '#7f7f7f'}
    
    # Panel 1: Q-Values
    if ens_eps is not None:
        # Smooth the lines
        s_real = smooth_data(ens_real, window=smooth_window)
        s_pred = smooth_data(ens_pred, window=smooth_window)
        
        ax1.plot(ens_eps, ens_real, color=colors['ens'], alpha=0.15, label='_nolegend_') # Raw
        ax1.plot(ens_eps, s_real, '--', color=colors['real'], alpha=0.8, label='Ensemble Real Q (Smoothed)')
        ax1.plot(ens_eps, s_pred, '-', color=colors['ens'], label='Ensemble Oracle-Best Q (Smoothed)', linewidth=2.5)
        
    if van_eps is not None:
        s_real_v = smooth_data(van_real, window=smooth_window)
        s_pred_v = smooth_data(van_pred, window=smooth_window)
        
        ax1.plot(van_eps, van_real, color=colors['van'], alpha=0.15, label='_nolegend_') # Raw
        ax1.plot(van_eps, s_pred_v, '-', color=colors['van'], label='Vanilla Min(Q1,Q2) (Smoothed)', linewidth=2.5)
        
    ax1.set_ylabel('Q-Value')
    ax1.set_title('Pure Q-Value Estimation Accuracy (Oracle Best-Head)', fontweight='bold', pad=20)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Panel 2: Errors
    if ens_eps is not None:
        s_err_e = smooth_data(ens_err, window=smooth_window)
        ax2.plot(ens_eps, ens_err, color=colors['ens'], alpha=0.15, label='_nolegend_')
        ax2.plot(ens_eps, s_err_e, '-', color=colors['ens'], label='Ensemble Oracle-Best MAE (Smoothed)', linewidth=2.5)
        
    if van_eps is not None:
        s_err_v = smooth_data(van_err, window=smooth_window)
        ax2.plot(van_eps, van_err, color=colors['van'], alpha=0.15, label='_nolegend_')
        ax2.plot(van_eps, s_err_v, '-', color=colors['van'], label='Vanilla Min(Q1,Q2) MAE (Smoothed)', linewidth=2.5)
        
    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Estimation Error Trend', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', frameon=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    output_png = os.path.join(save_dir, 'pure_q_line_comparison_smooth.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Smoothed plot saved to {output_png}")
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default='results/pure_q_results_line.npy')
    parser.add_argument('--save_dir', type=str, default='pic')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    plot_comparison(args.results_path, args.save_dir)
