import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the "Blue Line" data
blue_line_path = "/home/erzhu419/mine_code/LSTM-RL/logs/sac_v2_bus_sigma1p5_embed-full_wreg0p0_exp_sac_amax0p3_20251025_221923/rewards.npy"

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

if not os.path.exists(blue_line_path):
    print(f"Error: File not found {blue_line_path}")
    exit(1)

rewards = np.load(blue_line_path)
# plot_reward.py uses window=10, alpha=0.3 by default
reward_smooth = smooth_series(rewards, window=10, alpha=0.3)

plt.figure(figsize=(14, 7))
plt.axhline(y=-980000, color='gray', linestyle='--', label='No control average')

# SAC is usually "tab:blue" in plot_reward.py
plt.plot(np.arange(len(rewards)), reward_smooth["rolling"], label="SAC Rolling (amax0p3)", color="tab:blue")

# Assuming EWM might be shown too
plt.plot(np.arange(len(rewards)), reward_smooth["ewm"], linestyle='--', color="tab:blue", label="SAC EWM (amax0p3)")

plt.title(f"Blue Line Reproduction (amax0p3) - Compare with Five Algo Plot", fontsize=16)
plt.xlabel("Episodes", fontsize=16)
plt.ylabel("Rewards", fontsize=16)
plt.legend()
plt.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig('blue_line_reproduction.png')
print("Plot saved to blue_line_reproduction.png")
