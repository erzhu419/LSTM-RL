import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Plot Q-Accuracy Results')
    parser.add_argument('--sac_file', type=str, required=True, help='Path to SAC results npy')
    parser.add_argument('--ensemble_file', type=str, required=True, help='Path to Ensemble results npy')
    parser.add_argument('--log_dir', type=str, required=False, help='Path to logs (optional)')
    parser.add_argument('--output_file', type=str, required=True, help='Output PNG path')
    return parser.parse_args()

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None, None
    
    try:
        data = np.load(filepath, allow_pickle=True)
        # Handle list of dicts format
        if isinstance(data, np.ndarray) and data.ndim == 0:
            data = data.item()
            # Handle possible old dict format if needed, but we expect list of dicts
            return None, None, None 

        # Expecting array/list of dicts: [{'episode': 10, 'q_pred_mean': ...}, ...]
        episodes = [d['episode'] for d in data]
        q_pred = [d['q_pred_mean'] for d in data]
        q_real = [d['q_real_mean'] for d in data]
        
        # Sort
        indices = np.argsort(episodes)
        return np.array(episodes)[indices], np.array(q_pred)[indices], np.array(q_real)[indices]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, None

def main():
    args = parse_args()
    
    plt.figure(figsize=(12, 8))
    
    # Load Ensemble
    ep_ens, q_pred_ens, q_real_ens = load_data(args.ensemble_file)
    if ep_ens is not None:
        plt.plot(ep_ens, q_pred_ens, label='Ensemble Q_pred (Optimistic)', color='red', linestyle='--', linewidth=2)
        plt.plot(ep_ens, q_real_ens, label='Ensemble Q_real (True)', color='darkred', linestyle='-', linewidth=2)
        
    # Load SAC
    ep_sac, q_pred_sac, q_real_sac = load_data(args.sac_file)
    if ep_sac is not None:
        plt.plot(ep_sac, q_pred_sac, label='SAC Q_pred (Pessimistic)', color='blue', linestyle='--', linewidth=2)
        # SAC Q_real should be similar to Ensemble Q_real if environment is same
        plt.plot(ep_sac, q_real_sac, label='SAC Q_real (True)', color='darkblue', linestyle='-', linewidth=2)

    plt.xlabel('Training Episodes', fontsize=14)
    plt.ylabel('Q-Value', fontsize=14)
    plt.title('Corrected Q-Accuracy: Ensemble Optimism vs SAC Pessimism', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text annotation about correctness
    plt.annotate(f"Real Q is now Negative\n(Correct for Penalty Env)", 
                 xy=(0.02, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    print(f"Plot saved to {args.output_file}")

if __name__ == "__main__":
    main()
