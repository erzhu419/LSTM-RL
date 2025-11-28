#!/usr/bin/env python3
"""
Evaluate final checkpoints of different May 2024 configurations
Uses the same architecture as sac_ensemble_original.py
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
from bus_feature_utils import create_embedding_layer, build_bus_categorical_info

# Device setup
GPU = True
device = torch.device("cuda:0" if GPU and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Network Definitions (copied from sac_ensemble_original.py)
# ============================================================================

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, embedding_layer, action_range=1., init_w=3e-3,
                 log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.embedding_layer = embedding_layer

        # Adjust input dimension
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

    def evaluate(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action_0 = torch.tanh(mean + std * z)
        action = self.action_range / 2 * action_0 + self.action_range / 2

        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1. - action_0.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, z, mean, log_std


# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model_path, env, policy_net, num_episodes=5, deterministic=True):
    """Evaluate a single model checkpoint"""

    # Load model
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        policy_net.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    rewards = []

    for ep in range(num_episodes):
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

    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'all': rewards
    }


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    print("="*80)
    print("Evaluating May 2024 Ensemble Final Checkpoints")
    print("="*80)

    # Initialize environment
    print("\nInitializing environment...")
    # env_bus expects path to the env directory
    base_path = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_path, 'env')
    env = env_bus(env_path, debug=False, route_sigma=1.5)
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    # Build categorical info and embedding layer
    cat_cols, cat_code_dict = build_bus_categorical_info(env)
    embedding_layer = create_embedding_layer('full', cat_code_dict, cat_cols)

    # Calculate state dimension
    embedding_dim = embedding_layer.output_dim
    num_cont_features = 24  # Based on the model structure
    state_dim = embedding_dim + num_cont_features

    print(f"State dimension: {state_dim} (embedding: {embedding_dim} + continuous: {num_cont_features})")
    print(f"Action dimension: {action_dim}")

    # Initialize policy network
    hidden_dim = 64
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_layer, action_range).to(device)

    # Configurations to evaluate
    configs = [
        ("buf1M_r2_a0.3_w0.03", "1000000", "2", "0.3", "0.03", 499),
        ("buf100k_r2_a0.3_w0.01", "100000", "2", "0.3", "0.01", 426),
        ("buf100k_r3_a0.3_w0.01", "100000", "3", "0.3", "0.01", 235),
        ("buf100k_r4_a0.3_w0.01", "100000", "4", "0.3", "0.01", 425),
        ("buf100k_r10_a2_w0.01", "100000", "10", "2", "0.01", 18),
    ]

    results = []
    num_eval_eps = 3  # Evaluate each model with 3 runs

    print(f"\nEvaluating {len(configs)} configurations (each with {num_eval_eps} runs)...")
    print("="*80)

    BASE_DIR = "/home/erzhu419/mine_code/LSTM-RL/model/sac_v2_bus_ensemble"

    for name, buffer, ratio, alpha, wreg, final_ep in configs:
        model_path = os.path.join(
            BASE_DIR,
            f"replay_buffer_size_{buffer}",
            f"critic_actor_ratio_{ratio}",
            f"maximum_alpha_{alpha}",
            f"weight_reg_{wreg} {final_ep}"
        )

        if os.path.exists(model_path):
            print(f"\n{name} (episode {final_ep})...")
            result = evaluate_model(model_path, env, policy_net, num_episodes=num_eval_eps)

            if result:
                results.append({
                    'name': name,
                    'episodes': final_ep + 1,
                    'buffer': buffer,
                    'ratio': ratio,
                    'alpha': alpha,
                    'wreg': wreg,
                    **result
                })
                print(f"  Mean reward: {result['mean']:,.0f} ± {result['std']:,.0f}")
                print(f"  Range: [{result['min']:,.0f}, {result['max']:,.0f}]")
        else:
            print(f"\n{name}: Model not found at {model_path}")

    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Config':<25} {'Episodes':<10} {'Mean Reward':<20} {'Std':<15}")
    print("-"*80)

    for r in sorted(results, key=lambda x: x['mean'], reverse=True):
        print(f"{r['name']:<25} {r['episodes']:<10} {r['mean']:>15,.0f} {r['std']:>15,.0f}")

    if results:
        best = max(results, key=lambda x: x['mean'])
        print(f"\nBest performing config: {best['name']}")
        print(f"  Mean reward: {best['mean']:,.0f} ± {best['std']:,.0f}")
        print(f"  Improvement over no-control (-980,000): {(best['mean'] + 980000) / 980000 * 100:.1f}%")

    # Save results
    np.save('may_final_eval_results.npy', {'results': results})
    print(f"\nResults saved to: may_final_eval_results.npy")
    print("="*80)


if __name__ == '__main__':
    main()
