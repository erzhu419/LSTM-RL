import argparse
import numpy as np
import torch
import os
import glob
import re
from tqdm import tqdm
import sys
import time

# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.sim import env_bus
from normalization import Normalization, RunningMeanStd
from model_loader import ModelLoader

# Configuration
parser = argparse.ArgumentParser(description='Analyze Q-Accuracy (Q_pred vs Q_real)')
parser.add_argument('--env_path', type=str, default='/home/erzhu419/mine_code/LSTM-RL/env',
                    help='Path to environment data directory')
parser.add_argument('--checkpoint_dir', type=str, required=True,
                    help='Directory containing model checkpoints')
parser.add_argument('--model_type', type=str, choices=['sac', 'ensemble'], required=True,
                    help='Model type to load (sac or ensemble)')
parser.add_argument('--ensemble_size', type=int, default=10,
                    help='Number of Q-networks in ensemble')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='Hidden dimension of networks')
parser.add_argument('--action_range', type=float, default=15.0,
                    help='Action range (max holding time)')
parser.add_argument('--num_eval_episodes', type=int, default=1,
                    help='Number of evaluation episodes per checkpoint')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor')
parser.add_argument('--step_interval', type=int, default=1,
                    help='Evaluate every N-th checkpoint')

def get_checkpoints(directory, model_type, interval=1):
    """Find all policy checkpoints and sort them by episode number."""
    if model_type == 'sac':
        pattern = os.path.join(directory, "sac_v2_bus_episode_*_policy")
    else:
        pattern = os.path.join(directory, "ensemble_*_episode_*_policy")
        if not glob.glob(pattern):
            pattern = os.path.join(directory, "*_episode_*_policy")
        
    checkpoints = glob.glob(pattern)
    
    # Sort by episode number
    def extract_episode(path):
        match = re.search(r"episode_(\d+)", path)
        return int(match.group(1)) if match else 999999
        
    checkpoints.sort(key=extract_episode)
    
    # Subset
    if interval > 1:
        checkpoints = checkpoints[::interval]
        
    # Return prefix (remove _policy)
    return [c.replace("_policy", "") for c in checkpoints]

def main():
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load environment
    env = env_bus(args.env_path)

    # Load initial model (we will reload weights for each checkpoint)
    loader = ModelLoader(env, args.hidden_dim, args.action_range, device)
    
    # Get all checkpoints
    checkpoints = get_checkpoints(args.checkpoint_dir, args.model_type, args.step_interval)
    print(f"Found {len(checkpoints)} checkpoints.")

    results = {}

    for cp_prefix in tqdm(checkpoints, desc="Analyzing checkpoints"):
        match = re.search(r"episode_(\d+)", cp_prefix)
        if not match: continue
        episode_num = match.group(1)
        
        # Load model for this checkpoint
        try:
            if args.model_type == 'sac':
                model = loader.load_sac(cp_prefix)
            else:
                model = loader.load_ensemble(cp_prefix, args.ensemble_size)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {cp_prefix}: {e}")
            continue
            
        policy = model['policy']
        state_norm_obj = model['state_norm']

        q_predictions = []
        q_reals = []

        # Run Evaluation Episodes
        start_t = time.time()
        for _ in range(args.num_eval_episodes):
            env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=False)
            done = False
            # bus_trajectories[bus_id] = [{'type': 'q_pred', 'val': v}, {'type': 'reward', 'val': v}, ...]
            bus_trajectories = {i: [] for i in range(env.max_agent_num)}
            
            # 1. Rollout the full episode
            while not done:
                action_dict = {key: None for key in range(env.max_agent_num)}
                
                for bus_id in state_dict:
                    agent_states = state_dict[bus_id]
                    new_arrival = False
                    
                    if len(agent_states) == 1:
                        if action_dict[bus_id] is None:
                            new_arrival = True
                    elif len(agent_states) == 2:
                        # Transaction event
                        if agent_states[0][1] != agent_states[1][1]:
                             # Station change. Record reward.
                             bus_trajectories[bus_id].append({'type': 'reward', 'val': reward_dict[bus_id]})
                        
                        # Move to new state
                        state_dict[bus_id] = agent_states[1:]
                        new_arrival = True
                    
                    if new_arrival:
                        raw_state = np.array(state_dict[bus_id][0])
                        state_input = state_norm_obj(raw_state, update=False)
                        state_tensor = torch.FloatTensor(state_input).unsqueeze(0).to(device)
                        
                        # Policy action
                        a = policy.get_action(state_input, deterministic=True, device=device)
                        action_dict[bus_id] = a
                        
                        # Q prediction
                        a_val = float(a) if isinstance(a, (np.ndarray, list, torch.Tensor)) else a
                        action_tensor = torch.FloatTensor([[a_val]]).to(device)
                        
                        with torch.no_grad():
                            if args.model_type == 'ensemble':
                                q_preds_ensemble = model['q_net'](state_tensor, action_tensor)
                                q_pred = torch.min(q_preds_ensemble, dim=0).values.item()
                            else:
                                q_p1 = model['q1'](state_tensor, action_tensor).item()
                                q_p2 = model['q2'](state_tensor, action_tensor).item()
                                q_pred = min(q_p1, q_p2)
                        
                        bus_trajectories[bus_id].append({'type': 'q_pred', 'val': q_pred})

                # Step the environment
                state_dict, reward_dict, done = env.step(action_dict)

            # 2. Back-calculate Q_real
            for bus_id in bus_trajectories:
                traj = bus_trajectories[bus_id]
                q_real_accum = 0
                for item in reversed(traj):
                    if item['type'] == 'reward':
                        q_real_accum = item['val'] + args.gamma * q_real_accum
                    elif item['type'] == 'q_pred':
                        q_predictions.append(item['val'])
                        q_reals.append(q_real_accum)

        duration = time.time() - start_t
        if q_predictions:
            results[int(episode_num)] = {
                'q_pred': np.mean(q_predictions),
                'q_real': np.mean(q_reals),
                'abs_error': np.mean(np.abs(np.array(q_predictions) - np.array(q_reals))),
                'rel_error': np.mean(np.abs(np.array(q_predictions) - np.array(q_reals)) / (np.abs(np.array(q_reals)) + 1.0))
            }
            print(f"Ep {episode_num}: Q_pred={results[int(episode_num)]['q_pred']:.2f}, Q_real={results[int(episode_num)]['q_real']:.2f}, Error={results[int(episode_num)]['abs_error']:.2f}, Time={duration:.1f}s")

            # Save results incrementally
            output_path = os.path.join(args.checkpoint_dir, f"{args.model_type}_q_accuracy_results.npy")
            np.save(output_path, results)

if __name__ == '__main__':
    main()
