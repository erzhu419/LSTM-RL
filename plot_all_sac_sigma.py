import numpy as np
import matplotlib.pyplot as plt
import os
import glob

base_dir = "/home/erzhu419/mine_code/LSTM-RL/logs"
pattern = os.path.join(base_dir, "sac_v2_bus_sigma*")
directories = glob.glob(pattern)
directories.sort()

plt.figure(figsize=(14, 8))

found_data = False

for directory in directories:
    reward_file = os.path.join(directory, "rewards.npy")
    if os.path.exists(reward_file):
        try:
            rewards = np.load(reward_file)
            print(f"Loaded {reward_file} with shape: {rewards.shape}")
            label_name = os.path.basename(directory).replace("sac_v2_bus_", "")
            
            # Smoothing for better visualization if data is long enough
            if len(rewards) > 100:
                window_size = 50
                rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(rewards_smooth, label=label_name + " (Smoothed)")
                plt.plot(rewards, alpha=0.3, label=label_name + " (Raw)")
            else:
                plt.plot(rewards, label=label_name)
            
            found_data = True
        except Exception as e:
            print(f"Error loading {reward_file}: {e}")
    else:
        print(f"No rewards.npy found in {directory}")

if found_data:
    plt.title("Comparison of Rewards for SAC V2 Bus Sigma Runs")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()
    output_path = "all_sac_sigma_rewards.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
else:
    print("No valid reward files found to plot.")
