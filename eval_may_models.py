"""
Evaluate May 2024 Ensemble Models
Load models from May 1-2 training and evaluate their performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.sim import env_bus

# Simple argument configuration (no argparse to avoid conflict)
class Args:
    episodes_to_test = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    num_eval_episodes = 5
    deterministic = True

args = Args()

# Simple Policy Network (copied from sac_ensemble_original.py)
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range / 2 * torch.tanh(mean + std * z) + self.action_range / 2

        action = self.action_range / 2 * torch.tanh(mean).detach().cpu().numpy()[0] + self.action_range / 2 if deterministic else action.detach().cpu().numpy()[0]
        return action

# Model directory
MODEL_DIR = '/home/erzhu419/mine_code/LSTM-RL/model/sac_v2_bus_ensemble/replay_buffer_size_1000000/critic_actor_ratio_2/maximum_alpha_0.3/'
MODEL_PREFIX = 'weight_reg_0.03'

print("="*80)
print("Evaluating May 2024 Ensemble Models")
print("="*80)
print(f"Model directory: {MODEL_DIR}")
print(f"Configuration:")
print(f"  - replay_buffer_size: 1000000")
print(f"  - critic_actor_ratio: 2")
print(f"  - maximum_alpha: 0.3")
print(f"  - weight_reg: 0.03")
print(f"Episodes to test: {args.episodes_to_test}")
print("="*80)

# Check which models exist
available_models = []
for ep in args.episodes_to_test:
    model_path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX} {ep}")
    if os.path.exists(model_path):
        available_models.append(ep)
    else:
        print(f"Warning: Model for episode {ep} not found")

print(f"\nFound {len(available_models)} available models: {available_models}")

# Initialize environment
print("\nInitializing environment...")
path = os.path.abspath('env')
env = env_bus(path, debug=False, route_sigma=1.5)
action_dim = env.action_space.shape[0]
action_range = env.action_space.high[0]

# Initialize policy network
print("Initializing policy network...")
state_dim = env.state_dim
hidden_dim = 256
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)

# Results storage
results = {
    'episodes': [],
    'mean_rewards': [],
    'std_rewards': [],
    'all_rewards': []
}

# Evaluate each checkpoint
print("\n" + "="*80)
print("Starting evaluation...")
print("="*80)

for idx, ep in enumerate(available_models):
    print(f"\n[{idx+1}/{len(available_models)}] Evaluating episode {ep}...")
    model_path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX} {ep}")

    # Load model
    try:
        policy_net.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        policy_net.eval()
    except Exception as e:
        print(f"\nError loading model for episode {ep}: {e}")
        continue

    # Run evaluation episodes
    eval_rewards = []
    for eval_ep in range(args.num_eval_episodes):
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
                        a = policy_net.get_action(
                            raw_state, deterministic=args.deterministic
                        )
                        action_dict[key] = a

                elif len(state_dict[key]) == 2:
                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        episode_reward += reward_dict[key]

                    state_dict[key] = state_dict[key][1:]
                    raw_state = np.array(state_dict[key][0])
                    action_dict[key] = policy_net.get_action(
                        raw_state, deterministic=args.deterministic
                    )

            state_dict, reward_dict, done = env.step(action_dict, render=False)

        eval_rewards.append(episode_reward)

    # Store results
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)

    results['episodes'].append(ep)
    results['mean_rewards'].append(mean_reward)
    results['std_rewards'].append(std_reward)
    results['all_rewards'].append(eval_rewards)

    print(f"\nEpisode {ep}: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")

# Save results
print("\n" + "="*80)
print("Saving results...")
print("="*80)

np.save('may_ensemble_eval_results.npy', results)
print("Results saved to: may_ensemble_eval_results.npy")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Mean rewards with error bars
ax1.errorbar(results['episodes'], results['mean_rewards'],
             yerr=results['std_rewards'], marker='o', capsize=5, linewidth=2)
ax1.set_xlabel('Training Episode', fontsize=12)
ax1.set_ylabel('Mean Evaluation Reward', fontsize=12)
ax1.set_title('May 2024 Ensemble Model Performance\n(1M buffer, ratio=2, alpha=0.3, wreg=0.03)',
              fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=-0.98e6, color='r', linestyle='--', alpha=0.5, label='No Control Baseline')
ax1.legend()

# Plot 2: All individual runs
for i, ep in enumerate(results['episodes']):
    x_positions = [ep] * len(results['all_rewards'][i])
    ax2.scatter(x_positions, results['all_rewards'][i], alpha=0.5, s=30)

ax2.plot(results['episodes'], results['mean_rewards'], 'r-', linewidth=2, label='Mean')
ax2.set_xlabel('Training Episode', fontsize=12)
ax2.set_ylabel('Evaluation Reward', fontsize=12)
ax2.set_title('Individual Evaluation Runs', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('may_ensemble_evaluation.png', dpi=150, bbox_inches='tight')
print("Plot saved to: may_ensemble_evaluation.png")

# Print summary
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)
print(f"Best checkpoint: Episode {results['episodes'][np.argmax(results['mean_rewards'])]}")
print(f"Best mean reward: {max(results['mean_rewards']):.2f}")
print(f"Latest checkpoint: Episode {results['episodes'][-1]}")
print(f"Latest mean reward: {results['mean_rewards'][-1]:.2f}")
print(f"\nImprovement over no control (-0.98M): {(results['mean_rewards'][-1] + 0.98e6) / 0.98e6 * 100:.2f}%")
print("="*80)

plt.show()
