#!/usr/bin/env python3
"""
Evaluate the 500-episode May 2024 model
Simple script focused on evaluating the most complete training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
import sys

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Add paths
sys.path.insert(0, '/home/erzhu419/mine_code/LSTM-RL')
from env.sim import env_bus
from bus_feature_utils import create_embedding_layer, build_bus_categorical_info

# ============================================================================
# Policy Network (from sac_ensemble_original.py)
# ============================================================================

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, embedding_layer, action_range=1.):
        super(PolicyNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)
        self.action_range = action_range
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Split features
        cat_features = state[:, :4].long()
        cont_features = state[:, 4:]

        # Embedding
        cat_embedded = self.embedding_layer(cat_features)
        x = torch.cat([cat_embedded, cont_features], dim=1)

        # Forward
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state, deterministic=True):
        state = torch.FloatTensor(state).to(device)
        mean, log_std = self.forward(state)

        if deterministic:
            action = self.action_range / 2 * torch.tanh(mean) + self.action_range / 2
        else:
            std = log_std.exp()
            normal = Normal(0, 1)
            z = normal.sample(mean.shape).to(device)
            action = self.action_range / 2 * torch.tanh(mean + std * z) + self.action_range / 2

        return action.detach().cpu().numpy()[0]


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_one_episode(env, policy_net):
    """Run one episode and return total reward"""
    env.reset()
    state_dict, reward_dict, _ = env.initialize_state(render=False)

    done = False
    episode_reward = 0
    action_dict = {key: None for key in range(env.max_agent_num)}

    step_count = 0
    while not done:
        for key in state_dict:
            if len(state_dict[key]) == 1:
                if action_dict[key] is None:
                    raw_state = np.array(state_dict[key][0])
                    action = policy_net.get_action(raw_state, deterministic=True)
                    action_dict[key] = action

            elif len(state_dict[key]) == 2:
                if state_dict[key][0][1] != state_dict[key][1][1]:
                    episode_reward += reward_dict[key]

                state_dict[key] = state_dict[key][1:]
                raw_state = np.array(state_dict[key][0])
                action_dict[key] = policy_net.get_action(raw_state, deterministic=True)

        state_dict, reward_dict, done = env.step(action_dict, render=False)
        step_count += 1

        if step_count % 100 == 0:
            print(f"  Step {step_count}, current reward: {episode_reward:,.0f}", end='\r')

    print(f"  Episode completed in {step_count} steps, reward: {episode_reward:,.0f}" + " "*20)
    return episode_reward


def main():
    print("="*80)
    print("Evaluating 500-Episode May 2024 Ensemble Model")
    print("="*80)

    # Model path
    model_path = "/home/erzhu419/mine_code/LSTM-RL/model/sac_v2_bus_ensemble/replay_buffer_size_1000000/critic_actor_ratio_2/maximum_alpha_0.3/weight_reg_0.03 499"

    print(f"\nModel: {os.path.basename(model_path)}")
    print(f"Config: buffer=1M, ratio=2, alpha=0.3, wreg=0.03, episode=500")

    # Check model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        return

    print(f"✓ Model file found")

    # Initialize environment
    print("\nInitializing environment...")
    env = env_bus('/home/erzhu419/mine_code/LSTM-RL/env', debug=False, route_sigma=1.5)
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]
    print(f"✓ Environment initialized")
    print(f"  Action dim: {action_dim}, Action range: {action_range}")

    # Build embedding layer with custom dimensions (from trained model)
    print("\nBuilding embedding layer...")
    cat_cols, cat_code_dict = build_bus_categorical_info(env)

    # Specify embedding dimensions to match trained model
    embedding_dims = {
        'bus_id': 12,
        'station_id': 11,
        'time_period': 7,
        'direction': 1
    }

    embedding_layer = create_embedding_layer('full', cat_code_dict, cat_cols, embedding_dims=embedding_dims)

    embedding_dim = embedding_layer.output_dim
    num_cont_features = 24
    state_dim = embedding_dim + num_cont_features

    print(f"✓ Embedding layer built")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Continuous features: {num_cont_features}")
    print(f"  Total state dim: {state_dim}")

    # Initialize policy
    print("\nInitializing policy network...")
    hidden_dim = 64
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_layer, action_range).to(device)
    print(f"✓ Policy network created (hidden_dim={hidden_dim})")

    # Load model
    print("\nLoading model weights...")
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        policy_net.load_state_dict(state_dict)
        policy_net.eval()
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate
    num_eval_episodes = 5
    print(f"\n{'='*80}")
    print(f"Running {num_eval_episodes} evaluation episodes...")
    print(f"{'='*80}\n")

    rewards = []
    for i in range(num_eval_episodes):
        print(f"Episode {i+1}/{num_eval_episodes}:")
        reward = evaluate_one_episode(env, policy_net)
        rewards.append(reward)
        print()

    # Results
    rewards = np.array(rewards)
    print("="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Mean reward:    {rewards.mean():>15,.0f}")
    print(f"Std deviation:  {rewards.std():>15,.0f}")
    print(f"Min reward:     {rewards.min():>15,.0f}")
    print(f"Max reward:     {rewards.max():>15,.0f}")
    print(f"\nAll rewards: {[f'{r:,.0f}' for r in rewards]}")

    # Compare to baseline
    no_control = -980000
    improvement = (rewards.mean() - no_control) / abs(no_control) * 100
    print(f"\nNo-control baseline: {no_control:,}")
    print(f"Improvement: {improvement:.1f}%")

    # Save
    np.save('may_500ep_eval_results.npy', {
        'rewards': rewards,
        'mean': rewards.mean(),
        'std': rewards.std(),
        'config': 'buf1M_r2_a0.3_w0.03_ep500'
    })
    print(f"\n✓ Results saved to: may_500ep_eval_results.npy")
    print("="*80)


if __name__ == '__main__':
    main()
