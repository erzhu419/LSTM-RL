import os
import sys
import torch
import numpy as np
import argparse
import glob
import re
import copy
from tqdm import tqdm
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.distributions import Normal
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from env.sim import env_bus
from normalization import Normalization, RunningMeanStd
from bus_feature_utils import create_embedding_layer

# ----------------- Network Definitions (Local to avoid import issues & support adaptive sizing) -----------------

class SAC_PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, action_range=1.):
        super(SAC_PolicyNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.action_range = action_range

    def forward(self, state):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
        x = F.relu(self.linear1(state_with_embeddings))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def get_action(self, state, deterministic, device):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range/2 * torch.tanh(mean + std * z) + self.action_range/2
        action = self.action_range/2 * torch.tanh(mean).detach().cpu().numpy()[0] + self.action_range/2 if deterministic else action.detach().cpu().numpy()[0]
        return action

class SAC_SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer):
        super(SAC_SoftQNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
        x = torch.cat([state_with_embeddings, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)

# Ensemble Components
class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()
    def reset_parameters(self):
        for i in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x):
        return x @ self.weight + self.bias

class Ensemble_SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, embedding_layer, ensemble_size=10):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.ensemble_size = ensemble_size
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, ensemble_size),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, ensemble_size),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, ensemble_size),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, ensemble_size),
        )
    def forward(self, state, action):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
        state_action = torch.cat([state_with_embeddings, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.ensemble_size, dim=0)
        return self.critic(state_action).squeeze(-1)

# ----------------- Helper Functions -----------------

def detect_hidden_dim(state_dict):
    """Detect hidden dimension from policy state dict."""
    if 'linear1.weight' in state_dict:
        return state_dict['linear1.weight'].shape[0]
    return 256 # Default fallback

def get_checkpoints(directory, prefix="checkpoint_episode_"):
    """Find all checkpoints and sort by episode."""
    pattern = os.path.join(directory, f"{prefix}*")
    files = glob.glob(pattern)
    # Filter for policy or q files and dedup
    files = [f for f in files if '_policy' in f or '_q' in f]
    
    eps_map = {}
    for f in files:
        # Match episode number e.g. ...episode_100_policy
        match = re.search(r"episode_(\d+)", f)
        if match:
            ep = int(match.group(1))
            base_path = f[:f.rfind('_')] # Strip suffix
            eps_map[ep] = base_path
            
    sorted_eps = sorted(eps_map.keys())
    return [(ep, eps_map[ep]) for ep in sorted_eps]

def load_model(env, base_path, model_type, device, ensemble_size=10):
    # Load state dicts
    try:
        policy_sd = torch.load(base_path + '_policy', map_location=device, weights_only=True)
        if model_type == 'ensemble':
            q_sd = torch.load(base_path + '_q', map_location=device, weights_only=True)
        else:
            q1_sd = torch.load(base_path + '_q1', map_location=device, weights_only=True)
            q2_sd = torch.load(base_path + '_q2', map_location=device, weights_only=True)
    except Exception as e:
        print(f"Error loading {base_path}: {e}")
        return None

    # Detect Hidden Dim
    hidden_dim = detect_hidden_dim(policy_sd)
    
    # Detect Embedding Dims AND Cardinality from State Dict
    emb_update_dim = {}
    emb_update_card = {}
    for key, name in [
        ('embedding_layer.embeddings.bus_id.weight', 'bus_id'),
        ('embedding_layer.embeddings.station_id.weight', 'station_id'),
        ('embedding_layer.embeddings.time_period.weight', 'time_period'),
        ('embedding_layer.embeddings.direction.weight', 'direction')
    ]:
        if key in policy_sd:
             emb_update_dim[name] = policy_sd[key].shape[1]
             emb_update_card[name] = policy_sd[key].shape[0]
             
    # Init Models with detected embeddings
    # We start with default from env, but override if checkpoint differs
    cat_code_dict = {
        'bus_id': {i: i for i in range(25)},
        'station_id': {i: i for i in range(23)},
        'time_period': {i: i for i in range(15)},
        'direction': {0: 0, 1: 1}
    }
    cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']
    
    # Check if we need to adjust cat_code_dict to match cardinality?
    # If checkpoint has 22 stations, we should probably just set the embedding layer to 22.
    # The forward pass will clamp indices to max anyway? 
    # bus_feature_utils.py: indices = torch.clamp(indices, 0, max_index) where max_index = cardinality - 1.
    # So if we reduce cardinality to 22, and env produces station_id 22, it will be clamped to 21. 
    # This is "safe" enough for loading.
    
    embedding_layer = create_embedding_layer('full', cat_code_dict, cat_cols)
    
    # Force overwrite attributes based on checkpoint
    for col in cat_cols:
        expected_dim = emb_update_dim.get(col)
        expected_card = emb_update_card.get(col)
        
        current_dim = embedding_layer.embeddings[col].embedding_dim
        current_card = embedding_layer.embeddings[col].num_embeddings
        
        if (expected_dim and expected_dim != current_dim) or (expected_card and expected_card != current_card):
             # print(f"Fixing {col}: ({current_card}, {current_dim}) -> ({expected_card}, {expected_dim})")
             new_card = expected_card if expected_card else current_card
             new_dim = expected_dim if expected_dim else current_dim
             embedding_layer.embeddings[col] = nn.Embedding(new_card, new_dim)
             embedding_layer.cardinalities[col] = new_card # Update cardinality record too
    
    # Re-calculate output dim
    total_emb_dim = sum([l.embedding_dim for l in embedding_layer.embeddings.values()])
    embedding_layer.output_dim = total_emb_dim
    
    num_cont_features = env.state_dim - len(cat_cols)
    state_dim = embedding_layer.output_dim + num_cont_features
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    policy = SAC_PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_layer, action_range).to(device)
    policy.load_state_dict(policy_sd)
    policy.eval()

    if model_type == 'ensemble':
        q_net = Ensemble_SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_layer, ensemble_size).to(device)
        q_net.load_state_dict(q_sd)
        q_net.eval()
        return {'type': 'ensemble', 'policy': policy, 'q_net': q_net, 'state_norm': None} # Norm handled separately
    else:
        q1 = SAC_SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_layer).to(device)
        q2 = SAC_SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_layer).to(device)
        q1.load_state_dict(q1_sd)
        q2.load_state_dict(q2_sd)
        q1.eval()
        q2.eval()
        return {'type': 'sac', 'policy': policy, 'q1': q1, 'q2': q2, 'state_norm': None}

def robust_state_norm(obs, norm_obj):
    if norm_obj is None: return obs
    try:
        return norm_obj(np.array(obs), update=False)
    except:
        # Fallback for shape mismatch (basic fix)
        return np.array(obs)

def compute_reg_norm(model):
    weight_norm, bias_norm = [], []
    
    net = model.get('q_net') or model.get('q1') 
    
    for name, param in net.named_parameters():
        # Exclude embeddings explicitly (Ensemble only regularizes critic)
        if 'embedding' in name: continue
        
        # For Ensemble, we only care about 'critic' submodule or Linear layers in VectorizedCritic
        # For Vanilla, we don't use reg, so it doesn't matter, but let's be robust.
        
        if len(param.shape) == 3: # Vectorized [Ens, In, Out]
             weight_norm.append(torch.norm(param, p=1, dim=[1, 2])) if 'weight' in name else bias_norm.append(torch.norm(param, p=1, dim=[1, 2]))
        elif len(param.shape) == 2: # Linear Weight [Out, In]
             weight_norm.append(torch.norm(param, p=1)) if 'weight' in name else None 
        elif len(param.shape) == 1: # Linear Bias [Out]
             bias_norm.append(torch.norm(param, p=1)) if 'bias' in name else None

    # Robust Stack
    if not weight_norm: return torch.tensor(0.0)
    
    # Check if we have tensors of same size or mix
    # Ensemble: All should be [10]
    # Vanilla: All should be scalar
    
    if len(weight_norm[0].shape) > 0: # Vectorized
        w_stack = torch.stack(weight_norm)
        r_norm = torch.sum(w_stack, dim=0)
        if bias_norm:
             b_stack = torch.stack(bias_norm[:-1]) # Match original logic of dropping last bias?
             if b_stack.shape[0] > 0:
                 r_norm += torch.sum(b_stack, dim=0)
    else: # Scalar
        r_norm = torch.sum(torch.stack(weight_norm))
        if bias_norm:
             r_norm += torch.sum(torch.stack(bias_norm[:-1])) # Assuming logic matches
             
    return r_norm

def evaluate_checkpoint(env, model, device, num_episodes=1, gamma=0.99, max_steps=1000, alpha=0.2, weight_reg=0.01, beta_ood=0.0, oracle_mode='per_episode'):
    q_preds_pure = [] # Corrected
    q_preds_raw = []  # Original
    q_reals = []
    
    # Calculate Reg Norm — keep as per-head vector for ensemble
    reg_norm_vec = None  # numpy array: [num_heads] for ensemble, scalar for vanilla
    if weight_reg > 0:
        with torch.no_grad():
            r_val = compute_reg_norm(model)
            reg_norm_vec = r_val.cpu().numpy()
            if reg_norm_vec.ndim == 0:
                reg_norm_vec = np.array([reg_norm_vec.item()])

    # Init Normalization 
    num_cat = 4
    num_cont = env.state_dim - num_cat
    initial_mean = np.zeros(num_cont); initial_mean[:3] = [360., 360., 90.]
    initial_std = np.ones(num_cont); initial_std[:3] = [165., 133., 45.]
    running_ms = RunningMeanStd(shape=(num_cont,), init_mean=initial_mean, init_std=initial_std)
    state_norm = Normalization(num_categorical=num_cat, num_numerical=num_cont, running_ms=running_ms)

    for _ in range(num_episodes):
        env.reset()
        state_dict, reward_dict, _ = env.initialize_state(render=False)
        done = False
        step_count = 0
        bus_trajectories = {i: [] for i in range(env.max_agent_num)}
        
        while not done and step_count < max_steps:
            action_dict = {key: None for key in range(env.max_agent_num)}
            
            # 1. Identify which buses need an action this step
            active_buses = []
            for bus_id in state_dict:
                agent_states = state_dict[bus_id]
                new_arrival = False
                
                if len(agent_states) == 2:
                    if agent_states[0][1] != agent_states[1][1]:
                         bus_trajectories[bus_id].append({'type': 'reward', 'val': reward_dict[bus_id]})
                    state_dict[bus_id] = agent_states[1:]
                    new_arrival = True
                elif len(agent_states) == 1:
                    if action_dict[bus_id] is None:
                        new_arrival = True
                
                if new_arrival:
                    active_buses.append(bus_id)
            
            # 2. Batched Inference
            if active_buses:
                states_input = []
                for bus_id in active_buses:
                    raw_state = state_dict[bus_id][0]
                    states_input.append(np.array(raw_state))
                
                states_tensor = torch.FloatTensor(np.stack(states_input)).to(device)
                
                with torch.no_grad():
                    # Get Action & Log Prob (Optimized math)
                    mean, log_std = model['policy'].forward(states_tensor)
                    std = log_std.exp()
                    z = torch.randn_like(mean)
                    action_0 = torch.tanh(mean + std * z)
                    
                    # Action scaling
                    a_range = model['policy'].action_range
                    actions = a_range/2 * action_0 + a_range/2
                    
                    # Log Prob (Bias Corrected Version)
                    epsilon = 1e-6
                    # Fast log_prob without Normal object
                    # log_prob = -((z**2)/2) - log_std - log(sqrt(2*pi))
                    log_probs = -0.5 * (z**2) - log_std - 0.5 * np.log(2 * np.pi)
                    # Tanh correction
                    log_probs -= torch.log(1. - action_0.pow(2) + epsilon)
                    # Range scaling correction
                    log_probs -= np.log(a_range)
                    log_probs = log_probs.sum(dim=1)
                    
                    # Q Prediction
                    actions_input = actions.cpu().numpy() 
                    actions_tensor = torch.FloatTensor(actions_input).to(device)
                    # If actions_input is (batch,) make it (batch, 1)
                    if actions_tensor.dim() == 1:
                        actions_tensor = actions_tensor.unsqueeze(-1)

                    if model['type'] == 'ensemble':
                        q_vals = model['q_net'](states_tensor, actions_tensor)
                        q_preds_raw_batch = q_vals.detach().cpu().numpy().T # [EnsSize, Batch] -> [Batch, EnsSize]
                    else:
                        q1 = model['q1'](states_tensor, actions_tensor)
                        q2 = model['q2'](states_tensor, actions_tensor)
                        q_preds_raw_batch = torch.cat([q1, q2], dim=1).detach().cpu().numpy() # [Batch, 2]
                
                # 3. Store results and populate action_dict
                log_probs_np = log_probs.cpu().numpy()
                for i, bus_id in enumerate(active_buses):
                    act_val = float(actions_input[i])
                    action_dict[bus_id] = act_val
                    
                    q_p_raw_vec = q_preds_raw_batch[i] # Vector
                    lp_val = float(log_probs_np[i])
                    
                    # STEADY-STATE BIAS CORRECTION
                    # Q learned in normalized-reward space includes entropy + reg + ood bias.
                    # Ensemble Bellman: target = r + γ(Q' - α·lp - w·rn)
                    # Ensemble Loss:   L = MSE + β_ood·std(Q)
                    # Vanilla Bellman:  target = r + γ(min(Q1',Q2') - α·lp)
                    # bias_per_step = α·|lp| + w·rn + β·std(Q)  (ensemble)
                    #                = α·|lp|                    (vanilla)
                    # Steady-state: bias = bias_per_step / (1 - γ)
                    num_heads = len(q_p_raw_vec)
                    entropy_bias = alpha * abs(lp_val)
                    if reg_norm_vec is not None and len(reg_norm_vec) == num_heads:
                        bias_term = entropy_bias + weight_reg * reg_norm_vec  # Per-head [num_heads]
                    else:
                        bias_term = np.full(num_heads, entropy_bias)
                    # Add OOD std penalty (ensemble only, beta_ood > 0)
                    if beta_ood > 0 and num_heads > 1:
                        q_std = np.std(q_p_raw_vec)
                        bias_term = bias_term + beta_ood * q_std
                    bias_accum = bias_term / (1 - gamma)
                    
                    # Store Vector and steady-state bias
                    bus_trajectories[bus_id].append({'type': 'q_pred', 'raw_vec': q_p_raw_vec, 'bias': bias_accum})

            # 4. Step Environment
            state_dict, reward_dict, done = env.step(action_dict)
            step_count += 1
            
        # Back-calculate with Oracle Best-Head Selection (per episode/bus)
        for bus_id in bus_trajectories:
            traj = bus_trajectories[bus_id]
            
            # Pass 1: back-calculate g_t for each q_pred step
            g_t = 0
            step_data = []  # list of (q_pure_vec, bias_vec, g_t) for each q_pred step
            for item in reversed(traj):
                if item['type'] == 'reward':
                    g_t = item['val'] + gamma * g_t
                elif item['type'] == 'q_pred':
                    q_pure_vec = item['raw_vec'] - item['bias']
                    step_data.append((q_pure_vec, item['bias'], g_t))
            
            if not step_data:
                continue
            
            if oracle_mode == 'per_step':
                # Per-step oracle: each step independently picks the closest head
                for q_pure_vec, bias_vec, gt_val in step_data:
                    abs_diffs = np.abs(q_pure_vec - gt_val)
                    best_idx = np.argmin(abs_diffs)
                    q_preds_pure.append(q_pure_vec[best_idx])
                    raw_val = q_pure_vec[best_idx] + (bias_vec[best_idx] if np.ndim(bias_vec) > 0 else bias_vec)
                    q_preds_raw.append(raw_val)
                    q_reals.append(gt_val)
            else:
                # Per-episode oracle: find best head (lowest MAE across ALL steps)
                num_heads = len(step_data[0][0])
                head_maes = np.zeros(num_heads)
                for q_pure_vec, _, gt_val in step_data:
                    head_maes += np.abs(q_pure_vec - gt_val)
                head_maes /= len(step_data)
                best_head = np.argmin(head_maes)
                
                # Use the chosen head for all steps
                for q_pure_vec, bias_vec, gt_val in step_data:
                    q_preds_pure.append(q_pure_vec[best_head])
                    raw_val = q_pure_vec[best_head] + (bias_vec[best_head] if np.ndim(bias_vec) > 0 else bias_vec)
                    q_preds_raw.append(raw_val)
                    q_reals.append(gt_val)
                    
    if not q_preds_pure: return None, None
    return np.mean(q_preds_pure), np.mean(q_reals)

# Global variable for worker-level persistence
_worker_env = None

def worker_init():
    global _worker_env
    # Pre-initialize env once per process
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../env'))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    _worker_env = env_bus(env_path, debug=False, route_sigma=1.5)
    _worker_env.enable_plot = False

def worker_fn(ep, path, model_type, alpha, eval_episodes, max_steps, device_str, oracle_mode='per_episode'):
    try:
        global _worker_env
        # Set single thread to avoid CPU oversubscription
        torch.set_num_threads(1)
        
        # Set device within worker
        device = torch.device(device_str)
        
        env = _worker_env
        model = load_model(env, path, model_type, device)
        if not model: return ep, model_type, None, None
        
        weight_reg = 0.01 if model_type == 'ensemble' else 0.0
        beta_ood = 0.01 if model_type == 'ensemble' else 0.0
        q_pred, q_real = evaluate_checkpoint(env, model, device, eval_episodes, max_steps=max_steps, alpha=alpha, weight_reg=weight_reg, beta_ood=beta_ood, oracle_mode=oracle_mode)
        return ep, model_type, q_pred, q_real
    except Exception as e:
        print(f"Worker failed for {model_type} episode {ep}: {e}")
        import traceback
        traceback.print_exc()
        return ep, model_type, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_dir', type=str, default='/home/erzhu419/mine_code/LSTM-RL/RE-SAC/ensemble_10/model')
    parser.add_argument('--vanilla_dir', type=str, default='/home/erzhu419/mine_code/LSTM-RL/第一篇论文的模型和图/sac_v2_bus')
    parser.add_argument('--save_dir', type=str, default='/home/erzhu419/mine_code/LSTM-RL/RE-SAC/comparasion_q/results')
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--step_interval', type=int, default=1, help='Process every N-th checkpoint')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel processes")
    parser.add_argument("--device", type=str, default="cpu", help="Device for workers (cuda or cpu)")
    parser.add_argument("--oracle_mode", type=str, default="per_episode", choices=['per_episode', 'per_step'], help="Oracle selection: per_episode or per_step")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    results = {'ensemble': {}, 'vanilla': {}}

    # Helper to load alphas
    def load_alphas(dir_path):
        try:
            log_dir = dir_path.replace('model', 'logs')
            alpha_path = os.path.join(log_dir, 'alpha_values_episode.npy')
            if not os.path.exists(alpha_path):
                alpha_path = os.path.join(dir_path, 'alpha_values_episode.npy')
            if os.path.exists(alpha_path):
                alphas = np.load(alpha_path)
                return {i: alphas[i] for i in range(len(alphas))}
        except:
            pass
        return {}

    alpha_e = load_alphas(args.ensemble_dir)
    alpha_v = load_alphas(args.vanilla_dir)

    # Process Ensemble checkpoints
    e_checkpoints = get_checkpoints(args.ensemble_dir, "ensemble_") 
    if not e_checkpoints:
         e_checkpoints = get_checkpoints(args.ensemble_dir, "checkpoint_episode_")
    if args.step_interval > 1:
        e_checkpoints = e_checkpoints[::args.step_interval]

    # Process Vanilla checkpoints
    v_checkpoints = get_checkpoints(args.vanilla_dir, "sac_v2_episode_")
    if not v_checkpoints:
         v_checkpoints = get_checkpoints(args.vanilla_dir, "checkpoint_episode_")
    if args.step_interval > 1:
        v_checkpoints = v_checkpoints[::args.step_interval]

    tasks = []
    for ep, path in e_checkpoints:
        tasks.append((ep, path, 'ensemble', alpha_e.get(ep, 0.2), args.eval_episodes, args.max_steps, args.device, args.oracle_mode))
    for ep, path in v_checkpoints:
        tasks.append((ep, path, 'vanilla', alpha_v.get(ep, 0.2), args.eval_episodes, args.max_steps, args.device, args.oracle_mode))

    print(f"Starting parallel evaluation with {args.num_workers} workers. Total tasks: {len(tasks)}")
    
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=worker_init) as executor:
        futures = {executor.submit(worker_fn, *task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Evaluating"):
            ep, m_type, q_p, q_r = future.result()
            if q_p is not None:
                results[m_type][ep] = {'q_pred': q_p, 'q_real': q_r}

    out_name = f'pure_q_results_line_{args.oracle_mode}.npy'
    np.save(os.path.join(args.save_dir, out_name), results)
    print(f"Saved results to {os.path.join(args.save_dir, out_name)}")

if __name__ == "__main__":
    try:
        from multiprocessing import set_start_method
        set_start_method('spawn')
    except RuntimeError:
        pass
    main()
