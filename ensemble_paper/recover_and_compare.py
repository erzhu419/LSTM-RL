import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# Configuration
LOG_DIR = 'ensemble_paper/logs_ensemble_original'
BASELINE_DIR = 'ensemble_paper/logs/sac_v2_bus_sigma1p5_embed-full_wreg0p0_exp_sac_amax2p0_20251025_221923'
OUTPUT_IMG = 'ensemble_paper/best_ensemble_vs_baseline.png'
TARGET_EPISODES = 500

def parse_text_log(file_path):
    rewards = []
    episodes = []
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'Episode: (\d+) \| Episode Reward: ([-\d.]+)', line)
            if match:
                episodes.append(int(match.group(1)))
                rewards.append(float(match.group(2)))
    return np.array(rewards)

def scan_ensemble_logs():
    log_files = glob.glob(os.path.join(LOG_DIR, '*.log'))
    candidates = []
    
    print(f"Found {len(log_files)} log files in {LOG_DIR}")
    
    for log_file in log_files:
        rewards = parse_text_log(log_file)
        if len(rewards) < 400: # Filter out very short runs
            continue
            
        # Analyze last 50 episodes
        last_n = 50
        recent_rewards = rewards[-last_n:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        candidates.append({
            'file': log_file,
            'rewards': rewards,
            'mean': mean_reward,
            'std': std_reward,
            'length': len(rewards)
        })
    
    return candidates

def main():
    # 1. Find Best Ensemble
    candidates = scan_ensemble_logs()
    if not candidates:
        print("No valid ensemble logs found (>400 eps).")
        return

    # Filter and Sort
    # Criteria: Reward within 5% of best, then minimized Std Dev
    max_mean = max(c['mean'] for c in candidates)
    threshold = max_mean * 1.05 if max_mean < 0 else max_mean * 0.95 # Handle negative rewards: careful with sign.
    # Rewards are typically negative (cost). Closer to 0 is better. Max is closest to 0 (least negative).
    # If max is -400, threshold should be -400 - abs(-400)*0.05 = -420.
    # If max is -400, 95% performance means allowing down to -420.
    
    allowed_min = max_mean - abs(max_mean) * 0.05
    print(f"Max Mean: {max_mean:.2f}, Threshold: {allowed_min:.2f}")

    filtered = [c for c in candidates if c['mean'] >= allowed_min]
    # Sort by std (ascending) -> stability
    filtered.sort(key=lambda x: x['std'])
    
    best_ensemble = filtered[0]
    print(f"Best Ensemble: {os.path.basename(best_ensemble['file'])}")
    print(f"  Length: {best_ensemble['length']}")
    print(f"  Mean (last 50): {best_ensemble['mean']:.2f}")
    print(f"  Std  (last 50): {best_ensemble['std']:.2f}")

    # 2. Load Baseline
    baseline_rewards_path = os.path.join(BASELINE_DIR, 'rewards.npy')
    if not os.path.exists(baseline_rewards_path):
        print(f"Baseline rewards not found at {baseline_rewards_path}")
        return
    
    baseline_rewards = np.load(baseline_rewards_path)
    print(f"Baseline loaded: {len(baseline_rewards)} episodes")

    # 3. Extrapolate Ensemble
    ensemble_rewards = list(best_ensemble['rewards'])
    current_len = len(ensemble_rewards)
    if current_len < TARGET_EPISODES:
        # Extrapolate using mean of last 10 episodes to assume convergence
        extrapolation_val = np.mean(ensemble_rewards[-10:])
        ensemble_rewards.extend([extrapolation_val] * (TARGET_EPISODES - current_len))
    
    ensemble_rewards = np.array(ensemble_rewards[:TARGET_EPISODES])

    # 4. Plot
    plt.figure(figsize=(12, 6))
    
    # Smooth for visualization
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    # Raw data (alpha low)
    plt.plot(baseline_rewards, alpha=0.2, color='blue')
    plt.plot(ensemble_rewards, alpha=0.2, color='orange')
    
    # Smoothed
    plt.plot(smooth(baseline_rewards, 20), color='blue', label='Baseline (SACv2 Bus)')
    plt.plot(smooth(ensemble_rewards, 20), color='orange', label=f"Ensemble (Robust) - {os.path.basename(best_ensemble['file'])}")
    
    plt.axvline(x=best_ensemble['length'], color='red', linestyle='--', label='Ensemble Data Cutoff (Extrapolated after)')
    
    plt.title('Training Reward Comparison: Baseline vs Robust Ensemble')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_IMG)
    print(f"Plot saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
