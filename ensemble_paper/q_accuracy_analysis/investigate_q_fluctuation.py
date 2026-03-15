import argparse
import numpy as np
import torch
import os
import glob
import re
from tqdm import tqdm
import sys

# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.sim import env_bus
from normalization import Normalization, RunningMeanStd
from model_loader import ModelLoader

# Configuration
parser = argparse.ArgumentParser(description='Investigate Q_real fluctuation')
parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory containing model checkpoints')
parser.add_argument('--env_path', type=str, default='env', help='Path to env config')
parser.add_argument('--episodes', type=int, nargs='+', default=[200, 205], help='Episodes to investigate')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
parser.add_argument('--ensemble_size', type=int, default=10, help='Ensemble size')
parser.add_argument('--horizon', type=int, default=50)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

def calculate_q_real_debug(env, policy_net, action_dict, target_bus_id, gamma, horizon, device):
    """
    Calculate Q_real with Debug Prints
    """
    snapshot = env.get_snapshot()
    
    total_reward = 0.0
    discount = 1.0
    
    print(f"\n--- Debug Q_real Rollout for Bus {target_bus_id} ---")

    # 1. Immediate Step
    state_dict, reward_dict, done = env.step(action_dict)
    
    step_reward = 0.0
    if target_bus_id in reward_dict:
        step_reward = reward_dict[target_bus_id]
        total_reward += step_reward
        
    print(f"Step 0: Reward={step_reward:.4f}, Total={total_reward:.4f}, Done={done}")
        
    if done:
        env.load_snapshot(snapshot)
        return total_reward

    # 2. Rollout
    for t in range(1, horizon + 1):
        if done:
            break
            
        discount *= gamma
        
        next_action_dict = {key: 0.0 for key in range(env.max_agent_num)}
        
        for key in state_dict:
            if len(state_dict[key]) == 0: continue
            raw_state = np.array(state_dict[key][0])
            a = policy_net.get_action(raw_state, deterministic=True, device=device)
            next_action_dict[key] = a
            
        state_dict, reward_dict, done = env.step(next_action_dict)
        
        step_reward = 0.0
        if target_bus_id in reward_dict:
            step_reward = reward_dict[target_bus_id]
            total_reward += discount * step_reward
            
        print(f"Step {t}: Reward={step_reward:.4f}, Disc_Reward={discount*step_reward:.4f}, Total={total_reward:.4f}, Done={done}")

    env.load_snapshot(snapshot)
    return total_reward

def main():
    env_config_path = os.path.abspath(args.env_path)
    env = env_bus(env_config_path, debug=False)
    
    loader = ModelLoader(env, args.hidden_dim, env.action_space.high[0], device)
    
    for episode in args.episodes:
        print(f"\n================ Investigating Episode {episode} ================")
        
        # Construct pattern to find file
        # Ensemble: checkpoint_episode_X_policy
        policy_path = os.path.join(args.checkpoint_dir, f"checkpoint_episode_{episode}_policy")
        q_path = os.path.join(args.checkpoint_dir, f"checkpoint_episode_{episode}_q")
        
        model_name = os.path.join(args.checkpoint_dir, f"checkpoint_episode_{episode}")
        
        if not os.path.exists(policy_path):
            print(f"Checkpoint not found: {policy_path}")
            continue
            
        try:
            model = loader.load_ensemble(model_name, args.ensemble_size)
            policy = model['policy']
        except Exception as e:
            print(f"Error loading model: {e}")
            continue
            
        # Run multiple samples to find a non-zero Q_real
        found_nonzero = False
        for sample_idx in range(20):
            if found_nonzero: break
            
            env.reset()
            state_dict, _, _ = env.initialize_state(render=False)
            done = False
            
            # Advance a bit to get into the episode
            for i in range(50): 
                if done: break
                
                action_dict = {key: 0.0 for key in range(env.max_agent_num)}
                available_buses = [k for k in state_dict if len(state_dict[k]) > 0]
                
                # Policy actions
                for key in state_dict:
                    if len(state_dict[key]) == 0: continue
                    raw_state = np.array(state_dict[key][0])
                    a = policy.get_action(raw_state, deterministic=True, device=device)
                    action_dict[key] = a
                
                # Randomly attempt to debug a bus
                if available_buses and i > 5 and np.random.random() < 0.2:
                    target_bus_id = np.random.choice(available_buses)
                    
                    # Run Q real debug silently first
                    # We need a quiet version of calculate_q_real or just capture output?
                    # Let's just run the debug version, if it's 0 it will just print 0s. 
                    # To reduce noise, maybe we only print if we find something bad?
                    # For now, let's just print.
                    
                    q_real = calculate_q_real_debug(env, policy, action_dict, target_bus_id, args.gamma, args.horizon, device)
                    
                    if q_real != 0:
                        print(f"!!! Found significant Q_real for Ep {episode} Sample {sample_idx} Bus {target_bus_id}: {q_real} !!!")
                        found_nonzero = True
                        break
                    else:
                        print(f"Sample {sample_idx} Bus {target_bus_id}: Q_real=0")
                    
                state_dict, _, done = env.step(action_dict)
        if not found_nonzero:
            print(f"Could not find non-zero Q_real for Episode {episode} in 20 samples.")

if __name__ == "__main__":
    main()
