import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Explicit data paths mapping
DATA_PATHS = {
    "SAC": {
        "reward": "/home/erzhu419/mine_code/LSTM-RL/logs/sac_v2_bus_sigma1p5_embed-full_wreg0p0_exp_sac_amax0p3_20251025_221923/rewards.npy",
        "std": None  # No matching std found in that dir, or not requested to search
    },
    "DDPG": {
        "reward": "/home/erzhu419/mine_code/LSTM-RL/logs/ddpg_bus_sigma1p5_embed-full_plot500/rewards.npy",
        "std": None
    },
    "TD3": {
        "reward": "/home/erzhu419/mine_code/LSTM-RL/logs/td3_bus_sigma1p5_embed-full_plot500/rewards.npy",
        "std": None
    },
    "MADDPG NPS": {
        "reward": "/home/erzhu419/mine_code/LSTM-RL/model/MADDPG/rewards_individual.npy",
        "std": "/home/erzhu419/mine_code/LSTM-RL/model/MADDPG/eval_reward_stds_individual.npy"
    },
    "MADDPG PS": {
        "reward": "/home/erzhu419/mine_code/LSTM-RL/model/MADDPG/rewards_ps.npy",
        "std": "/home/erzhu419/mine_code/LSTM-RL/model/MADDPG/eval_reward_stds_ps.npy"
    }
}

COLORS = {
    "SAC": "tab:blue",
    "DDPG": "tab:red",
    "TD3": "tab:purple",
    "MADDPG NPS": "tab:orange",
    "MADDPG PS": "tab:green",
}

def smooth_series(values: np.ndarray, window: int = 10, alpha: float = 0.3):
    df = pd.DataFrame(values, columns=["values"])
    return {
        "rolling": df["values"].rolling(window=window, min_periods=1).mean(),
        "ewm": df["values"].ewm(alpha=alpha).mean(),
    }

def smooth_std(values, window: int = 10, alpha: float = 0.3):
    if values is None:
        return {"rolling": None, "ewm": None}
    df = pd.DataFrame(values, columns=["values"])
    return {
        "rolling": df["values"].rolling(window=window, min_periods=1).mean(),
        "ewm": df["values"].ewm(alpha=alpha).mean(),
    }

def load_data():
    datasets = {}
    for label, paths in DATA_PATHS.items():
        if not os.path.exists(paths["reward"]):
            print(f"[WARN] Reward file not found for {label}: {paths['reward']}")
            continue
        
        try:
            rewards = np.load(paths["reward"])
            stds = None
            if paths["std"] and os.path.exists(paths["std"]):
                stds = np.load(paths["std"])
                
                # Align lengths if necessary
                if len(stds) != len(rewards):
                    min_len = min(len(rewards), len(stds))
                    rewards = rewards[:min_len]
                    stds = stds[:min_len]
            
            datasets[label] = (rewards, stds)
            print(f"Loaded {label}: {len(rewards)} points")
        except Exception as e:
            print(f"[ERROR] Failed to load {label}: {e}")
            
    return datasets

def plot_full_comparison():
    datasets = load_data()
    
    if not datasets:
        print("No datasets loaded. Exiting.")
        return

    plt.figure(figsize=(14, 7))
    plt.axhline(y=-980000, color='gray', linestyle='--', label='No control average')

    # Sorting to ensure consistent legend order if possible, or just iterate
    # Order: SAC, DDPG, TD3, MADDPG NPS, MADDPG PS
    plot_order = ["SAC", "DDPG", "TD3", "MADDPG NPS", "MADDPG PS"]
    
    for label in plot_order:
        if label not in datasets:
            continue
            
        rewards, stds = datasets[label]
        # Using default smoothing params matching plot_reward.py
        window = 10
        alpha = 0.3
        
        reward_smooth = smooth_series(rewards, window=window, alpha=alpha)
        std_smooth = smooth_std(stds, window=window, alpha=alpha)

        x_axis = np.arange(len(rewards))
        color = COLORS.get(label, "black")
        
        # Plot Rolling Mean (Solid Line)
        plt.plot(x_axis, reward_smooth["rolling"], label=label, color=color, linewidth=2)

        # Plot Std Error Band (if available) - using rolling std
        if std_smooth["rolling"] is not None:
            lower = reward_smooth["rolling"] - std_smooth["rolling"]
            upper = reward_smooth["rolling"] + std_smooth["rolling"]
            plt.fill_between(x_axis, lower, upper, color=color, alpha=0.2)

        # Optional: Plot EWM (Dashed) - mimicing args.show_ewm=False behavior by default to keep clean,
        # but original code had it as option. Let's stick to Rolling as primary.
        # If user wants EWM, we can add it. For now, solid lines look cleaner.

    plt.title("Reward Comparison (Reproduced)", fontsize=16)
    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Rewards", fontsize=16)
    plt.legend()
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    output_filename = 'reproduced_five_algo_comparison_full.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    plot_full_comparison()
