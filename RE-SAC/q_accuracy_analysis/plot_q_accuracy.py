import numpy as np
import matplotlib.pyplot as plt
import os
import re
import argparse

def parse_episode(filename):
    # extract number from filename, assuming it's the last number before extension or suffixes
    # Example: checkpoint_100 or model_100_q1
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def plot_results(sac_file, ensemble_file, output_path):
    plt.figure(figsize=(12, 8))

    # SAC
    if os.path.exists(sac_file):
        print(f"Loading SAC data from {sac_file}")
        try:
            sac_data = np.load(sac_file, allow_pickle=True)
            # Check if it is a 0-d array wrapping a dict (old format) or array of dicts (new format)
            if sac_data.ndim == 0:
                sac_data = sac_data.item()
                # Handle old format if needed, but assuming new format based on debug
                # If old format: {'checkpoints': [], 'q_pred': [] ...}
                if 'checkpoints' in sac_data:
                    checkpoints = sac_data['checkpoints']
                    episodes = [parse_episode(x) for x in checkpoints] # This depends on filename format
                    q_pred = sac_data['q_pred']
                    q_real = sac_data['q_real']
            else:
                # New format: List [dict, dict, ...]
                episodes = [d['episode'] for d in sac_data]
                q_pred = [d['q_pred_mean'] for d in sac_data]
                q_real = [d['q_real_mean'] for d in sac_data]

            if len(episodes) > 0:
                # Sort by episode
                sorted_indices = np.argsort(episodes)
                episodes = np.array(episodes)[sorted_indices]
                q_pred = np.array(q_pred)[sorted_indices]
                q_real = np.array(q_real)[sorted_indices]
                
                plt.plot(episodes, q_pred, label='SAC Q_pred', color='blue', linestyle='--')
                plt.plot(episodes, q_real, label='SAC Q_real', color='blue', linestyle='-')
        except Exception as e:
            print(f"Error loading SAC data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"SAC file not found: {sac_file}")

    # Ensemble
    if os.path.exists(ensemble_file):
        print(f"Loading Ensemble data from {ensemble_file}")
        try:
            ens_data = np.load(ensemble_file, allow_pickle=True)
             # Check if it is a 0-d array wrapping a dict (old format) or array of dicts (new format)
            if ens_data.ndim == 0:
                ens_data = ens_data.item()
                if 'checkpoints' in ens_data:
                    checkpoints = ens_data['checkpoints']
                    episodes = [parse_episode(x) for x in checkpoints]
                    q_pred = ens_data['q_pred']
                    q_real = ens_data['q_real']
            else:
                 # New format
                episodes = [d['episode'] for d in ens_data]
                q_pred = [d['q_pred_mean'] for d in ens_data]
                q_real = [d['q_real_mean'] for d in ens_data]
            
            if len(episodes) > 0:
                # Sort by episode
                sorted_indices = np.argsort(episodes)
                episodes = np.array(episodes)[sorted_indices]
                q_pred = np.array(q_pred)[sorted_indices]
                q_real = np.array(q_real)[sorted_indices]
                
                plt.plot(episodes, q_pred, label='Ensemble Q_pred', color='red', linestyle='--')
                plt.plot(episodes, q_real, label='Ensemble Q_real', color='red', linestyle='-')
        except Exception as e:
            print(f"Error loading Ensemble data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Ensemble file not found: {ensemble_file}")

    plt.xlabel('Training Episodes')
    plt.ylabel('Q-Value')
    plt.title('Q-Accuracy: Predicted vs Real Q-Values (Solid: Real, Dashed: Pred)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sac_file", default="/home/erzhu419/mine_code/LSTM-RL/RE-SAC/q_comparasion/sac_q_accuracy_norm.npy")
    parser.add_argument("--ensemble_file", default="/home/erzhu419/mine_code/LSTM-RL/RE-SAC/q_comparasion/ensemble_q_accuracy_norm.npy")
    parser.add_argument("--output_file", default="/home/erzhu419/mine_code/LSTM-RL/RE-SAC/q_comparasion/q_accuracy_comparison_norm.png")
    args = parser.parse_args()
    
    plot_results(args.sac_file, args.ensemble_file, args.output_file)
