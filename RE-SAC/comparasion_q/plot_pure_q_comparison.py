import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def plot_q_comparison(results_path, save_dir):
    data = np.load(results_path, allow_pickle=True).item()
    
    ensemble_pred = np.array(data['ensemble_pred'])
    ensemble_real = np.array(data['ensemble_real'])
    vanilla_pred = np.array(data['vanilla_pred'])
    vanilla_real = np.array(data['vanilla_real'])
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Scatter Plot with Ideal Line
    plt.figure(figsize=(12, 6))
    
    # Ensemble
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=ensemble_real, y=ensemble_pred, alpha=0.5, label='Ensemble Pure Q')
    min_val = min(ensemble_real.min(), ensemble_pred.min())
    max_val = max(ensemble_real.max(), ensemble_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('Real Pure Q (Monte Carlo)')
    plt.ylabel('Predicted Pure Q (Stripped)')
    plt.title(f'Ensemble SAC\nMSE: {mean_squared_error(ensemble_real, ensemble_pred):.2f}, R2: {r2_score(ensemble_real, ensemble_pred):.2f}')
    plt.legend()
    
    # Vanilla
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=vanilla_real, y=vanilla_pred, color='orange', alpha=0.5, label='Vanilla SAC Pure Q')
    min_val = min(vanilla_real.min(), vanilla_pred.min())
    max_val = max(vanilla_real.max(), vanilla_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('Real Pure Q (Monte Carlo)')
    plt.ylabel('Predicted Pure Q (Stripped)')
    plt.title(f'Vanilla SAC\nMSE: {mean_squared_error(vanilla_real, vanilla_pred):.2f}, R2: {r2_score(vanilla_real, vanilla_pred):.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pure_q_scatter_comparison.png'))
    plt.close()
    
    # 2. Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(ensemble_pred - ensemble_real, label='Ensemble Error (Pred - Real)', fill=True)
    sns.kdeplot(vanilla_pred - vanilla_real, label='Vanilla Error (Pred - Real)', fill=True)
    plt.xlabel('Q-Value Error')
    plt.title('Distribution of Q-Value Prediction Errors (Pure Q)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'pure_q_error_distribution.png'))
    plt.close()

    print(f"Plots saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', type=str, default='/home/erzhu419/mine_code/LSTM-RL/RE-SAC/comparasion_q/results/q_comparison_results.npy')
    parser.add_argument('--save_dir', type=str, default='/home/erzhu419/mine_code/LSTM-RL/RE-SAC/comparasion_q/pic')
    args = parser.parse_args()
    
    plot_q_comparison(args.results_path, args.save_dir)
