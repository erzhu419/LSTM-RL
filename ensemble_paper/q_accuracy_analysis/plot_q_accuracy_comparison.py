import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return None
    data = np.load(file_path, allow_pickle=True).item()
    # Sort by episode
    sorted_eps = sorted(data.keys())
    q_pred = [data[ep]['q_pred'] for ep in sorted_eps]
    q_real = [data[ep]['q_real'] for ep in sorted_eps]
    abs_error = [data[ep]['abs_error'] for ep in sorted_eps]
    rel_error = [data[ep]['rel_error'] for ep in sorted_eps]
    return np.array(sorted_eps), np.array(q_pred), np.array(q_real), np.array(abs_error), np.array(rel_error)

def plot_comparison():
    sac_path = '/home/erzhu419/mine_code/LSTM-RL/第一篇论文的模型和图/sac_v2_bus/sac_q_accuracy_results.npy'
    ensemble_path = '/home/erzhu419/mine_code/LSTM-RL/ensemble_paper/ensemble_10/model/ensemble_q_accuracy_results.npy'
    
    sac_data = load_data(sac_path)
    ens_data = load_data(ensemble_path)
    
    if sac_data is None or ens_data is None:
        print("Data files missing or incomplete.")
        return

    # Use a clean, professional style
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Panel 1: Q-Value Comparison
    # SAC
    ax1.plot(sac_data[0], sac_data[2], 'k--', alpha=0.5, label='Real Q (Ground Truth)')
    ax1.plot(sac_data[0], sac_data[1], 'r-o', markersize=4, label='SAC Predicted Q', linewidth=2)
    # Ensemble
    ax1.plot(ens_data[0], ens_data[1], 'b-s', markersize=4, label='Ensemble Predicted Q', linewidth=2)
    
    ax1.set_ylabel('Q-Value')
    ax1.set_title('Q-Value Estimation Accuracy: SAC vs Ensemble SAC', fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Panel 2: Mean Absolute Error (MAE)
    ax2.plot(sac_data[0], sac_data[3], 'r-o', markersize=4, label='SAC MAE', linewidth=2)
    ax2.plot(ens_data[0], ens_data[3], 'b-s', markersize=4, label='Ensemble MAE', linewidth=2)
    
    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Estimation Error over Training Time', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_png = 'q_accuracy_comparison_results.png'
    plt.savefig(output_png, dpi=300)
    print(f"Comparison plot saved to {output_png}")
    plt.show()

if __name__ == '__main__':
    plot_comparison()
