import numpy as np
import matplotlib.pyplot as plt

def plot_q_accuracy(file_path, output_pic):
    data = np.load(file_path, allow_pickle=True)
    if len(data) == 0:
        print("Empty data file.")
        return

    # Extract data
    episodes = [item['episode'] for item in data]
    q_pred_means = [item['q_pred_mean'] for item in data]
    q_real_means = [item['q_real_mean'] for item in data]
    q_pred_stds = [item['q_pred_std'] for item in data]
    q_real_stds = [item['q_real_std'] for item in data]

    plt.figure(figsize=(10, 6))
    
    # Plot Mean with Error Bars
    plt.errorbar(episodes, q_pred_means, yerr=q_pred_stds, label='Q_predict', fmt='-o', capsize=5, alpha=0.7)
    plt.errorbar(episodes, q_real_means, yerr=q_real_stds, label='Q_real (Monte Carlo)', fmt='-s', capsize=5, alpha=0.7)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    plt.xlabel('Training Episode')
    plt.ylabel('Q-Value')
    plt.title('Corrected Q-Accuracy: Predicted vs. Real Q (Gated & Normalized)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_pic)
    print(f"Plot saved to {output_pic}")

if __name__ == '__main__':
    plot_q_accuracy('q_accuracy_ensemble_10_fixed.npy', 'q_accuracy_fixed_plot.png')
