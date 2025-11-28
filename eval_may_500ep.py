#!/usr/bin/env python3
"""
Evaluate May 2024 500-episode ensemble model
Uses the EXACT same architecture as sac_ensemble_original.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.sim import env_bus

# Device setup
GPU = True
device = torch.device("cuda:0" if GPU and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Exact copy of EmbeddingLayer from sac_ensemble_original.py
# ============================================================================

class EmbeddingLayer(nn.Module):
    """
    Embedding layer for categorical features
    EXACT COPY from sac_ensemble_original.py (lines 97-117)
    """
    def __init__(self, cat_cols, cat_code_dict):
        super(EmbeddingLayer, self).__init__()
        self.cat_cols = cat_cols
        # Use the ORIGINAL formula: min(50, cardinality // 2)
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(len(cat_code_dict[col]), min(50, len(cat_code_dict[col]) // 2))
            for col in cat_cols
        })
        self.output_dim = sum([emb.embedding_dim for emb in self.embeddings.values()])

    def forward(self, cat_features):
        # cat_features shape: (batch_size, num_cat_features)
        embedded = []
        for i, col in enumerate(self.cat_cols):
            embedded.append(self.embeddings[col](cat_features[:, i]))
        return torch.cat(embedded, dim=1)


# ============================================================================
# Build categorical info (same as sac_ensemble_original.py)
# ============================================================================

def build_categorical_info(env):
    """Build categorical column info - EXACT copy from sac_ensemble_original.py"""
    cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']

    cat_code_dict = {
        'bus_id': {i: i for i in range(env.max_agent_num)},
        'station_id': {i: i for i in range(len(env.route['station']))},
        'time_period': {i: i for i in range(len(env.time_period))},
        'direction': {i: i for i in range(2)}
    }

    return cat_cols, cat_code_dict


# ============================================================================
# PolicyNetwork - EXACT copy from sac_ensemble_original.py
# ============================================================================

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, embedding_layer, action_range=1., init_w=3e-3,
                 log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.embedding_layer = embedding_layer

        # Input dimension after embedding
        adjusted_state_dim = state_dim

        self.linear1 = nn.Linear(adjusted_state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = action_dim

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Split categorical and continuous features
        cat_features = state[:, :4].long()
        cont_features = state[:, 4:]

        # Get embeddings
        cat_embedded = self.embedding_layer(cat_features)

        # Concatenate
        x = torch.cat([cat_embedded, cont_features], dim=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range / 2 * torch.tanh(mean + std * z) + self.action_range / 2

        if deterministic:
            action = self.action_range / 2 * torch.tanh(mean) + self.action_range / 2

        return action.detach().cpu().numpy()[0]


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model_path, env, policy_net, num_episodes=5, deterministic=True):
    """Evaluate a single model checkpoint"""

    # Load model
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        policy_net.eval()
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

    rewards = []

    for ep in range(num_episodes):
        print(f"  Episode {ep+1}/{num_episodes}...", end=' ', flush=True)
        env.reset()
        state_dict, reward_dict, _ = env.initialize_state(render=False)

        done = False
        episode_reward = 0
        action_dict = {key: None for key in list(range(env.max_agent_num))}

        while not done:
            for key in state_dict:
                if len(state_dict[key]) == 1:
                    if action_dict[key] is None:
                        raw_state = np.array(state_dict[key][0])
                        action = policy_net.get_action(raw_state, deterministic=deterministic)
                        action_dict[key] = action

                elif len(state_dict[key]) == 2:
                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        episode_reward += reward_dict[key]

                    state_dict[key] = state_dict[key][1:]
                    raw_state = np.array(state_dict[key][0])
                    action_dict[key] = policy_net.get_action(raw_state, deterministic=deterministic)

            state_dict, reward_dict, done = env.step(action_dict, render=False)

        rewards.append(episode_reward)
        print(f"Reward: {episode_reward:,.0f}")

    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'all': rewards
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("Evaluating May 2024 500-Episode Ensemble Model")
    print("Using EXACT architecture from sac_ensemble_original.py")
    print("="*80)

    # Initialize environment
    print("\nInitializing environment...")
    base_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_path, 'env')
    env = env_bus(env_path, debug=False, route_sigma=1.5)
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    # Build categorical info and embedding layer (ORIGINAL formula)
    cat_cols, cat_code_dict = build_categorical_info(env)
    embedding_layer = EmbeddingLayer(cat_cols, cat_code_dict)

    # Print embedding dimensions
    print("\nEmbedding layer configuration (ORIGINAL formula: min(50, cardinality//2)):")
    for col in cat_cols:
        cardinality = len(cat_code_dict[col])
        emb_dim = min(50, cardinality // 2)
        print(f"  {col:15} cardinality={cardinality:3} -> embedding_dim={emb_dim:2}")
    print(f"  Total embedding dimension: {embedding_layer.output_dim}")

    # Calculate state dimension
    embedding_dim = embedding_layer.output_dim
    num_cont_features = 24  # 4 categorical + 24 continuous = 28 total features
    state_dim = embedding_dim + num_cont_features

    print(f"\nNetwork configuration:")
    print(f"  State dimension: {state_dim} (embedding: {embedding_dim} + continuous: {num_cont_features})")
    print(f"  Action dimension: {action_dim}")
    print(f"  Action range: [0, {action_range}]")

    # Initialize policy network
    hidden_dim = 64
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_layer, action_range).to(device)

    # Model path
    MODEL_PATH = "/home/erzhu419/mine_code/LSTM-RL/model/sac_v2_bus_ensemble/replay_buffer_size_1000000/critic_actor_ratio_2/maximum_alpha_0.3/weight_reg_0.03 499"

    print(f"\nModel to evaluate:")
    print(f"  Path: {MODEL_PATH}")
    print(f"  Configuration: buffer=1M, ratio=2, alpha=0.3, wreg=0.03")
    print(f"  Episode: 500 (index 499)")

    if not os.path.exists(MODEL_PATH):
        print(f"\n✗ Error: Model file not found!")
        return

    # Evaluate
    print("\n" + "="*80)
    print("Starting evaluation (5 episodes)...")
    print("="*80 + "\n")

    result = evaluate_model(MODEL_PATH, env, policy_net, num_episodes=5, deterministic=True)

    if result:
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"Mean reward:  {result['mean']:>12,.0f} ± {result['std']:,.0f}")
        print(f"Min reward:   {result['min']:>12,.0f}")
        print(f"Max reward:   {result['max']:>12,.0f}")
        print(f"\nNo-control baseline: -980,000")
        improvement = (result['mean'] + 980000) / 980000 * 100
        print(f"Improvement: {improvement:>6.1f}%")
        print("="*80)

        # Save results
        np.save('may_500ep_eval_result.npy', result)
        print("\nResults saved to: may_500ep_eval_result.npy")


if __name__ == '__main__':
    main()
