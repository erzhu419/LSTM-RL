import torch
import numpy as np
import os
import argparse
import sys
import multiprocessing
import pickle
import glob
import re
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RE-SAC.q_accuracy_analysis.model_loader import ModelLoader
from env.sim import env_bus

# Global variable for worker-level persistence
_worker_env = None

def worker_init():
    global _worker_env
    # Prevent OpenMP thread contention when multiprocessing
    torch.set_num_threads(1)
    
    # Pre-initialize env once per process
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../env'))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    _worker_env = env_bus(env_path, debug=False, route_sigma=1.5)
    _worker_env.enable_plot = False

def collect_checkpoint_data(args):
    """
    Worker function to collect data for a SPECIFIC CHECKPOINT.
    Runs N episodes for this checkpoint.
    """
    try:
        ckpt_path, ckpt_idx, model_type, device, hidden_dim, action_range, num_episodes, output_dir = args
        
        global _worker_env
        if _worker_env is None:
            worker_init()
        
        env = _worker_env
        loader = ModelLoader(env, hidden_dim, action_range, device)
        
        # Load Model & Norm Stats
        if model_type == 'sac':
            model = loader.load_sac(ckpt_path)
        elif model_type == 'dsac':
            model = loader.load_dsac(ckpt_path)
        elif model_type == 'bac':
            model = loader.load_bac(ckpt_path)
        else:
            model = loader.load_ensemble(ckpt_path)
        
        # Extract Normalization Stats
        norm_stats = None
        if 'state_norm' in model and model['state_norm'] is not None:
            # Assuming RunningMeanStd structure
            rms = model['state_norm'].running_ms
            norm_stats = {
                'mean': rms.mean.tolist() if isinstance(rms.mean, np.ndarray) else rms.mean,
                'std': rms.std.tolist() if isinstance(rms.std, np.ndarray) else rms.std,
                'count': rms.n
            }
        
        checkpoint_data = []
        
        for ep_i in range(num_episodes):
            env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=False)
            done = False
            action_dict = {key: None for key in list(range(env.max_agent_num))}
            
            step = 0
            max_steps = 40000 # Collect full trajectories
            
            while not done and step < max_steps:
                # 1. Identify states needing action
                states_to_predict = []
                bus_keys = []
                raw_states_batch = []
                
                # Helper to process state list
                for key in state_dict:
                    needs_action = False
                    raw_state = None
                    
                    if len(state_dict[key]) == 1 and action_dict[key] is None:
                        raw_state = np.array(state_dict[key][0])
                        needs_action = True
                    elif len(state_dict[key]) == 2:
                        # Transition finished, consume old state
                        # We have completed a transition (S, A, R, S'). 
                        # The reward_dict[key] holds the reward resulting from the previous action.
                        if reward_dict[key] is not None:
                            checkpoint_data.append({
                                'event': 'reward',
                                'checkpoint_idx': ckpt_idx,
                                'episode_sub_idx': ep_i,
                                'bus_id': key,
                                'time': env.current_time,
                                'reward': float(reward_dict[key])
                            })
                        
                        state_dict[key] = state_dict[key][1:]
                        action_dict[key] = None # Reset action so we can predict a new one
                        raw_state = np.array(state_dict[key][0])
                        needs_action = True
                    
                    if needs_action:
                        # Disable Normalization to reproduce the exact old plot data
                        # state_input = raw_state
                        # Disable Normalization to reproduce the exact old plot data
                        state_input = raw_state
                        # if 'state_norm' in model and model['state_norm'] is not None:
                        #      state_input = model['state_norm'](raw_state, update=False)
                        # else:
                        #     state_input = raw_state
                        bus_keys.append(key)
                        states_to_predict.append(state_input)
                        raw_states_batch.append(raw_state)

                # 2. Batch Predict
                if states_to_predict:
                    states_tensor = torch.FloatTensor(np.array(states_to_predict)).to(device)
                    with torch.no_grad():
                        if model['type'] == 'ensemble':
                            mean, _ = model['policy'](states_tensor)
                            actions_tensor = torch.tanh(mean) * action_range/2 + action_range/2
                            # Record ALL Q-values (Ensemble Size)
                            q_vals = model['q_net'](states_tensor, actions_tensor)
                            q_vals_np = q_vals.cpu().numpy().T # [Batch, Ens_Size]
                        elif model['type'] == 'dsac':
                            mean, _ = model['policy'](states_tensor)
                            actions_tensor = torch.tanh(mean) * action_range/2 + action_range/2
                            z1 = model['z1'](states_tensor, actions_tensor)
                            z2 = model['z2'](states_tensor, actions_tensor)
                            # Record ALL 10+10 quantiles
                            q_vals_np = torch.cat([z1, z2], dim=1).cpu().numpy() # [Batch, 20]
                        elif model['type'] == 'bac':
                            mean, _ = model['policy'](states_tensor)
                            actions_tensor = torch.tanh(mean) * action_range/2 + action_range/2
                            q1, q2 = model['q_net'](states_tensor, actions_tensor)
                            q_vals_np = torch.cat([q1, q2], dim=1).cpu().numpy() # [Batch, 2]
                        else:
                            mean, _ = model['policy'](states_tensor)
                            actions_tensor = torch.tanh(mean) * action_range/2 + action_range/2
                            q1 = model['q1'](states_tensor, actions_tensor)
                            q2 = model['q2'](states_tensor, actions_tensor)
                            q_vals_np = torch.cat([q1, q2], dim=1).cpu().numpy() # [Batch, 2]
                        
                        actions_np = actions_tensor.cpu().numpy()

                    # Store Predictions
                    for i, key in enumerate(bus_keys):
                        action = float(actions_np[i][0])
                        action_dict[key] = action
                        
                        checkpoint_data.append({
                            'event': 'predict',
                            'checkpoint_idx': ckpt_idx,
                            'episode_sub_idx': ep_i,
                            'bus_id': key,
                            'time': env.current_time,
                            'state_raw': raw_states_batch[i].tolist(),
                            'action': action,
                            'q_vals': q_vals_np[i].tolist()
                        })

                # 3. Step Env
                state_dict, reward_dict, done = env.step(action_dict, debug=False, render=False)
                step += 1
                
        # Save formatted data for this checkpoint
        result_pkg = {
            'checkpoint_idx': ckpt_idx,
            'model_type': model_type,
            'norm_stats': norm_stats,
            'data': checkpoint_data
        }
        
        # Save to disk directly to avoid massive memory usage in main process
        save_path = os.path.join(output_dir, f"data_{model_type}_{ckpt_idx}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(result_pkg, f)
            
        return ckpt_idx
    except Exception as e:
        import traceback
        print(f"WORKER CRASHED: {e}")
        traceback.print_exc()
        return None

def get_checkpoints(model_dir, prefix, suffix='_policy'):
    # Find all matching files
    files = glob.glob(os.path.join(model_dir, f"{prefix}*"))
    # Filter for policy/q files to identify unique checkpoints
    checkpoints = []
    
    # Regex to extract episode number
    pattern = re.compile(rf"{re.escape(prefix)}_episode_(\d+){re.escape(suffix)}")
    
    seen_eps = set()
    for f in files:
        match = pattern.search(f)
        if match:
            ep = int(match.group(1))
            if ep not in seen_eps:
                seen_eps.add(ep)
                # Reconstruct base path (without suffix)
                base_path = os.path.join(model_dir, f"{prefix}_episode_{ep}")
                checkpoints.append((ep, base_path))
    
    return sorted(checkpoints, key=lambda x: x[0])

def main():
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_dir', type=str, default=None)
    parser.add_argument('--sac_dir', type=str, default=None)
    parser.add_argument('--dsac_dir', type=str, default=None)
    parser.add_argument('--bac_dir', type=str, default=None)
    parser.add_argument('--ensemble_prefix', type=str, default='checkpoint')
    parser.add_argument('--sac_prefix', type=str, default='sac_v2')
    parser.add_argument('--dsac_prefix', type=str, default='dsac_bus')
    parser.add_argument('--bac_prefix', type=str, default='checkpoint')
    
    parser.add_argument('--episodes_per_ckpt', type=int, default=2, help="Eps per checkpoint")
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='offline_dataset')
    parser.add_argument('--only_latest', action='store_true', help="Only collect data for the latest checkpoint")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Find Checkpoints
    print("Scanning for checkpoints...")
    tasks = []
    
    # Hidden dims
    hid_256 = 256
    hid_64 = 64
    
    # Ensemble
    if args.ensemble_dir:
        ckpts = get_checkpoints(args.ensemble_dir, args.ensemble_prefix, '_policy')
        if args.only_latest: ckpts = ckpts[-1:]
        print(f"Found {len(ckpts)} Ensemble checkpoints.")
        for ep, path in ckpts:
            if not os.path.exists(os.path.join(args.output_dir, f"data_ensemble_{ep}.pkl")):
                tasks.append((path, ep, 'ensemble', args.device, hid_256, 1.0, args.episodes_per_ckpt, args.output_dir))

    # SAC
    if args.sac_dir:
        ckpts = get_checkpoints(args.sac_dir, args.sac_prefix, '_policy')
        if args.only_latest: ckpts = ckpts[-1:]
        print(f"Found {len(ckpts)} SAC checkpoints.")
        for ep, path in ckpts:
            if not os.path.exists(os.path.join(args.output_dir, f"data_sac_{ep}.pkl")):
                tasks.append((path, ep, 'sac', args.device, hid_256, 1.0, args.episodes_per_ckpt, args.output_dir))

    # DSAC
    if args.dsac_dir:
        ckpts = get_checkpoints(args.dsac_dir, args.dsac_prefix, '_z1')
        if args.only_latest: ckpts = ckpts[-1:]
        print(f"Found {len(ckpts)} DSAC checkpoints.")
        for ep, path in ckpts:
            if not os.path.exists(os.path.join(args.output_dir, f"data_dsac_{ep}.pkl")):
                tasks.append((path, ep, 'dsac', args.device, hid_64, 1.0, args.episodes_per_ckpt, args.output_dir))

    # BAC
    if args.bac_dir:
        ckpts = get_checkpoints(args.bac_dir, args.bac_prefix, '_q')
        if args.only_latest: ckpts = ckpts[-1:]
        print(f"Found {len(ckpts)} BAC checkpoints.")
        for ep, path in ckpts:
            if not os.path.exists(os.path.join(args.output_dir, f"data_bac_{ep}.pkl")):
                tasks.append((path, ep, 'bac', args.device, hid_64, 1.0, args.episodes_per_ckpt, args.output_dir))
            
    print(f"Total tasks pending: {len(tasks)}")
    
    # 3. Run
    with multiprocessing.Pool(processes=args.workers, initializer=worker_init) as pool:
        list(tqdm(pool.imap_unordered(collect_checkpoint_data, tasks), total=len(tasks)))
        
    print("Data collection complete.")

if __name__ == '__main__':
    main()
