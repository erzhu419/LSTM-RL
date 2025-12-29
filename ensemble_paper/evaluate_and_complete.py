import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sac_ensemble_original import SAC_Trainer, ReplayBuffer  # Correct class name and REPLAY BUFFER
import env_original.sim as sim
import argparse

# Configuration
ENSEMBLE_MODEL_DIR = 'model/sac_ensemble_original/replay_buffer_size_1000000/critic_actor_ratio_2/maximum_alpha_0.6/weight_reg_0.01'
# BASELINE_DIR = "ensemble_paper/logs/sac_v2_bus_sigma1p5_embed-full_wreg0p0_exp_sac_amax2p0_20251025_221923"
BASELINE_DIR = "/home/erzhu419/mine_code/LSTM-RL/model/sac_v2_bus"

# Load Baseline Data
print(f"Loading baseline data from {BASELINE_DIR}...")
try:
    baseline_rewards = np.load(os.path.join(BASELINE_DIR, "rewards.npy"))
    # The historical model folder does not have args.json, so we skip verification
    print("Loaded baseline rewards.")
except Exception as e:
    print(f"Error loading baseline: {e}")
    baseline_rewards = None
OUTPUT_IMG = 'ensemble_paper/best_ensemble_vs_baseline_evaluated.png'
TARGET_EPISODES = 500
START_EPISODE = 465

# Arguments needed for SAC_Trainer initialization
ARGS = {
    'route_sigma': 1.5,
    'embedding_mode': 'full',
    'hidden_dim': 64, # Updated to 64 based on mismatch error
    'lr': 1e-5,
    'ensemble_size': 10,
    # Other args used inside SAC_Trainer or Env
    'weight_reg': 0.01,
    'maximum_alpha': 0.6,
    'critic_actor_ratio': 2,
    'replay_buffer_size': 1000000
}

# Dummy class to mimic argparse object if needed, or just pass values
class DummyArgs:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
    
    # Add attributes that might be accessed globally or via args
    use_gradient_clip = True
    use_state_norm = False
    use_reward_norm = False
    use_reward_scaling = False
    gamma = 0.99
    training_freq = 5
    plot_freq = 5
    auto_entropy = True
    beta_bc = 0.001
    beta = -2
    beta_ood = 0.01
    save_root = '.'
    run_name = 'eval_completion'
    env_path = 'ensemble_paper/env_original'
    eval_sigmas = None
    train = False
    test = True

def main():
    # Inject args into sac_ensemble_original module namespace if it relies on global 'args'
    import sac_ensemble_original
    sac_ensemble_original.args = DummyArgs(ARGS)
    sac_ensemble_original.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Initializing Environment...")
    # Initialize Env
    # Use ensemble_paper/env_original path
    
    env = sim.env_bus(path='ensemble_paper/env_original', route_sigma=ARGS['route_sigma']) 
    
    env.reset()
    
    print("Initializing Trainer...")
    # ReplayBuffer from sac_ensemble_original only takes capacity (and optional last_episode_step)
    replay_buffer = ReplayBuffer(ARGS['replay_buffer_size'])
    action_range = env.action_space.high[0]
    
    # REMOVED: embedding_mode, ensemble_size
    agent = SAC_Trainer(
        env,
        replay_buffer,
        hidden_dim=ARGS['hidden_dim'],
        action_range=action_range
    )
    
    # Load Model
    # stored layout: checkpoint_episode_465_policy (no suffix)
    # SAC_Trainer.load_model(path) appends '_q' and '_policy'
    # So we pass "checkpoint_episode_465"
    model_prefix = os.path.join(ENSEMBLE_MODEL_DIR, f"checkpoint_episode_{START_EPISODE}")
    
    print(f"Loading model from {model_prefix}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Manual loading with map_location to handle CPU/GPU mismatch
        q_path = model_prefix + '_q'
        policy_path = model_prefix + '_policy'
        
        # Load Q-network (optional for eval but good for consistency)
        if os.path.exists(q_path):
            q_state = torch.load(q_path, map_location=device, weights_only=True)
            agent.soft_q_net.load_state_dict(q_state)
            
        # Load Policy
        if os.path.exists(policy_path):
            policy_state = torch.load(policy_path, map_location=device, weights_only=True)
            agent.policy_net.load_state_dict(policy_state)
        else:
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
            
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Evaluate
    extra_episodes = TARGET_EPISODES - START_EPISODE
    debug_path = 'ensemble_paper/full_rewards_debug.npy'
    
    if os.path.exists(debug_path):
        print(f"Loading pre-computed rewards from {debug_path}")
        # We need to reconstruct full_rewards.
        # But wait, full_rewards_debug IS full_rewards.
        # We can just skip to the merging part.
        full_rewards = list(np.load(debug_path))
        eval_rewards = [] # Not needed if we possess full_rewards
        # We need to trick the next steps.
        # The next steps do: original_rewards parsing -> merging -> plotting.
        # Let's skip the loop.
    else:
        print(f"Evaluating for {extra_episodes} episodes...")
        
        eval_rewards = []
        
        from sac_ensemble_original import evaluate_policy
        # Use the evaluate_policy function from the module which handles the specific env interaction loop
        
        # Although evaluate_policy runs multiple eval episodes and returns mean/std.
        # We want exactly 'extra_episodes' sequential runs.
        # Let's iterate.
        
        for i in range(extra_episodes):
            # We can use evaluate_policy with num_eval_episodes=1
            mean_r, _ = evaluate_policy(agent, env, num_eval_episodes=1, deterministic=True)
            eval_rewards.append(mean_r)
            print(f"Eval Episode {START_EPISODE + i + 1}: {mean_r}")

    # Process logs for plotting
    # If we loaded full_rewards, we can skip original_rewards parsing OR just parse it for the cutoff line.
    original_log_path = 'ensemble_paper/logs_ensemble_original/wr0.01_a0.6_r2_buf1000000.log'
    original_rewards = []
    import re
    with open(original_log_path, 'r') as f:
        for line in f:
            match = re.search(r'Episode: (\d+) \| Episode Reward: ([-\d.]+)', line)
            if match:
                original_rewards.append(float(match.group(2)))

    if not os.path.exists(debug_path):
        full_rewards = original_rewards + eval_rewards
    # Else full_rewards is already loaded

    
    # Check for anomaly
    full_arr = np.array(full_rewards)
    max_val = np.max(full_arr)
    max_idx = np.argmax(full_arr)
    print(f"Max reward in combined data: {max_val} at index {max_idx}")
    
    # Save raw data for inspection
    np.save('ensemble_paper/full_rewards_debug.npy', full_arr)
    
    print(f"Max reward in combined data: {max_val} at index {max_idx}")
    
    # Save raw data for inspection
    np.save('ensemble_paper/full_rewards_debug.npy', full_arr)
    
    # Load Baseline
    baseline_rewards_path = os.path.join(BASELINE_DIR, 'rewards.npy')
    baseline_rewards = np.load(baseline_rewards_path)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='valid') # Use valid to avoid edge artifacts
        # We need to pad the x-axis or the data to match lengths if we actally want to align
        # Alternatively, use pandas rolling which handles NaNs better, or just truncate x-axis
        return y_smooth

    # Compute smoothed data
    box_pts = 10
    baseline_smooth = smooth(baseline_rewards[:TARGET_EPISODES], box_pts)
    full_smooth = smooth(full_rewards[:TARGET_EPISODES], box_pts)
    
    # Adjust X-axis for valid convolution (it shrinks by box_pts - 1)
    # We want to align the end or start? Usually centered.
    # Let's simple plot them against appropriate x range
    x_baseline = np.arange(len(baseline_smooth)) + box_pts//2
    x_full = np.arange(len(full_smooth)) + box_pts//2
    
    plt.plot(x_baseline, baseline_smooth, label='Baseline (SACv2 Bus)')
    plt.plot(x_full, full_smooth, label='Ensemble (Robust)')
    
    plt.axvline(x=len(original_rewards), color='red', linestyle='--', label='Training Cutoff')
    
    plt.title('Performance Comparison (Robust Ensemble vs Baseline)')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"Plot saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
