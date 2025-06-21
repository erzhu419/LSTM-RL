"""
DSAC (Distributional Soft Actor Critic) for Bus Environment
Optimized version with improved hyperparameters and training stability
"""

import psutil
import tracemalloc
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from normalization import Normalization, RewardScaling, RunningMeanStd

from IPython.display import clear_output
import matplotlib.pyplot as plt
from env.sim import env_bus
import os
import argparse
import numpy as np
import random
from copy import deepcopy
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser(description='Train or test DSAC neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--use_gradient_clip', type=bool, default=True, help="Trick 1:gradient clipping")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")
parser.add_argument("--training_freq", type=int, default=5, help="frequency of training the network")
parser.add_argument("--plot_freq", type=int, default=1, help="frequency of plotting the result")
parser.add_argument('--auto_entropy', type=bool, default=True, help='automatically updating alpha')
parser.add_argument("--maximum_alpha", type=float, default=0.3, help="max entropy weight")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--num_quantiles", type=int, default=10, help="number of quantiles (reduced for stability)")
parser.add_argument("--risk_type", type=str, default='CVaR', help="risk type: neutral, VaR, cvar, etc.")
parser.add_argument("--risk_param", type=float, default=0.0, help="risk parameter")
parser.add_argument("--critic_actor_ratio", type=int, default=2, help="ratio of critic updates to actor updates")
parser.add_argument("--tau_type", type=str, default='fix', help="quantile fraction type: fix, random")
args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity, last_episode_step=5000):
        self.capacity = capacity
        self.last_episode_step = last_episode_step
        self.buffer = {}
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position += 1

        if len(self.buffer) > self.capacity:
            keys_to_remove = list(self.buffer.keys())[:self.last_episode_step]
            for key in keys_to_remove:
                del self.buffer[key]

    def sample(self, batch_size):
        batch = random.sample(list(self.buffer.values()), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.stack(next_states)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class EmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict, cat_cols):
        super(EmbeddingLayer, self).__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = cat_cols

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(len(cat_code_dict[col]), min(50, len(cat_code_dict[col]) // 2))
            for col in cat_cols
        })

    def forward(self, cat_tensor):
        embedding_tensor_group = []
        for idx, col in enumerate(self.cat_cols):
            layer = self.embeddings[col]
            out = layer(cat_tensor[:, idx])
            embedding_tensor_group.append(out)

        embed_tensor = torch.cat(embedding_tensor_group, dim=1)
        return embed_tensor


class QuantileNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, num_quantiles=8, init_w=3e-3):
        super(QuantileNetwork, self).__init__()
        
        self.embedding_layer = embedding_layer
        self.num_quantiles = num_quantiles
        self.embedding_size = 64
        
        # Simplified network structure for better stability
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_quantiles)
        
        # Initialize weights
        self.fc4.weight.data.uniform_(-init_w, init_w)
        self.fc4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # Process categorical and numerical features
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]
        
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
        
        # State-action features
        x = torch.cat([state_with_embeddings, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        quantiles = self.fc4(x)  # (batch_size, num_quantiles)
        
        return quantiles


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.embedding_layer = embedding_layer
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
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))
        action = self.action_range/2 * action_0 + self.action_range/2

        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range/2 * torch.tanh(mean + std * z) + self.action_range/2

        action = self.action_range/2 * torch.tanh(mean).detach().cpu().numpy()[0] + self.action_range/2 if deterministic else action.detach().cpu().numpy()[0]
        return action


def quantile_huber_loss(quantile_pred, target, tau, kappa=1.0):
    """Simplified quantile Huber loss function"""
    quantile_pred = quantile_pred.unsqueeze(2)  # (batch, quantiles, 1)
    target = target.unsqueeze(1)  # (batch, 1, quantiles)
    tau = tau.unsqueeze(2)  # (batch, quantiles, 1)
    
    # u = target - quantile_pred  # (batch, quantiles, quantiles)
    # huber_loss = F.smooth_l1_loss(quantile_pred, target, reduction='none')
    # quantile_weight = torch.abs(tau - (u < 0).float())
    # loss = quantile_weight * huber_loss
    # return loss.mean()

    u = target - quantile_pred  # (batch, N, N)
    abs_u = torch.abs(u)
    huber_loss = torch.where(abs_u <= kappa, 0.5 * u ** 2, kappa * (abs_u - 0.5 * kappa))
    quantile_weight = torch.abs(tau - (u < 0).float())
    loss = quantile_weight * huber_loss / kappa
    return loss.mean()


def quantile_regression_loss(quantile_pred, target, tau, kappa=1.0):
    """
    Quantile regression loss for distributional RL
    
    Args:
        quantile_pred: (batch_size, N) - predicted quantiles
        target: (batch_size, N') - target quantiles  
        tau: (batch_size, N) - quantile fractions for predictions
        kappa: Huber loss threshold
    """
    batch_size = quantile_pred.shape[0]
    N = quantile_pred.shape[1]
    N_prime = target.shape[1]
    
    # Expand dimensions for pairwise comparison
    quantile_pred_expanded = quantile_pred.unsqueeze(-1)  # (batch_size, N, 1)
    target_expanded = target.unsqueeze(1)  # (batch_size, 1, N')
    tau_expanded = tau.unsqueeze(-1)  # (batch_size, N, 1)
    
    # Calculate pairwise differences
    u = target_expanded - quantile_pred_expanded  # (batch_size, N, N')
    
    # Huber loss - fix dimension mismatch
    # Ensure both tensors have the same shape for smooth_l1_loss
    pred_broadcasted = quantile_pred_expanded.expand(-1, -1, N_prime)  # (batch_size, N, N')
    target_broadcasted = target_expanded.expand(-1, N, -1)  # (batch_size, N, N')
    
    huber_loss = F.smooth_l1_loss(pred_broadcasted, target_broadcasted, reduction='none', beta=kappa)
    
    # Quantile loss weights
    quantile_weight = torch.abs(tau_expanded - (u < 0).float())
    
    # Combine losses
    loss = quantile_weight * huber_loss
    return loss.mean()

class DSAC_Trainer():
    def __init__(self, env, replay_buffer, hidden_dim, action_range, num_quantiles=8):
        # Categorical and numerical features
        cat_cols = ['bus_id', 'station_id', 'time_period','direction']
        cat_code_dict = {
            'bus_id': {i: i for i in range(env.max_agent_num)},
            'station_id': {i: i for i in range(round(len(env.stations) / 2))},
            'time_period': {i: i for i in range(env.timetables[-1].launch_time//3600 + 2)},
            'direction': {0: 0, 1: 1}
        }
        
        self.num_cat_features = len(cat_cols)
        self.num_cont_features = env.state_dim - self.num_cat_features
        self.num_quantiles = num_quantiles
        
        # Create embedding layer
        embedding_layer = EmbeddingLayer(cat_code_dict, cat_cols)
        embedding_dim = sum([min(50, len(cat_code_dict[col]) // 2) for col in cat_cols])
        state_dim = embedding_dim + self.num_cont_features

        self.replay_buffer = replay_buffer

        # Networks
        self.zf1 = QuantileNetwork(state_dim, action_dim, hidden_dim, embedding_layer, num_quantiles).to(device)
        self.zf2 = QuantileNetwork(state_dim, action_dim, hidden_dim, embedding_layer, num_quantiles).to(device)
        self.target_zf1 = deepcopy(self.zf1).to(device)
        self.target_zf2 = deepcopy(self.zf2).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_layer, action_range).to(device)
        
        # Initialize target networks
        for target_param, param in zip(self.target_zf1.parameters(), self.zf1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_zf2.parameters(), self.zf2.parameters()):
            target_param.data.copy_(param.data)

        # Alpha for entropy regularization
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        print('Quantile Networks (1,2): ', self.zf1)
        print('Policy Network: ', self.policy_net)

        # Optimizers with higher learning rates
        zf_lr = 1e-5  # Increased from 1e-5
        policy_lr = 1e-5  # Increased from 1e-5
        alpha_lr = 1e-5  # Increased from 1e-5

        self.zf1_optimizer = optim.Adam(self.zf1.parameters(), lr=zf_lr)
        self.zf2_optimizer = optim.Adam(self.zf2.parameters(), lr=zf_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Normalization
        initial_mean = [360., 360., 90.]
        initial_std = [165., 133., 45.]
        running_ms = RunningMeanStd(shape=(self.num_cont_features,), init_mean=initial_mean, init_std=initial_std)
        self.state_norm = Normalization(num_categorical=self.num_cat_features, num_numerical=self.num_cont_features, running_ms=running_ms)
        self.reward_scaling = RewardScaling(shape=1, gamma=0.99)

        # Fixed quantile fractions for stability
        if args.tau_type == 'fix':
            self.tau = torch.linspace(0.0, 1.0, num_quantiles + 2)[1:-1].to(device)  # Remove 0 and 1
        else:
            self.tau = None

    def get_tau(self, batch_size):
        """Generate quantile fractions"""
        if args.tau_type == 'fix':
            tau = self.tau.unsqueeze(0).expand(batch_size, -1)  # (batch_size, num_quantiles)
        else:
            tau = torch.rand(batch_size, self.num_quantiles).to(device)
            tau = tau.sort(dim=1)[0]  # Sort to ensure ascending order
        return tau

    def get_risk_weighted_q(self, quantile_values, risk_type='neutral', risk_param=0.0):
        """Apply risk measure to quantile values"""
        if risk_type == 'neutral':
            # Expected value (risk-neutral)
            return quantile_values.mean(dim=1, keepdim=True)
        elif risk_type == 'VaR':
            # Value at Risk
            idx = int(risk_param * self.num_quantiles)
            idx = max(0, min(idx, self.num_quantiles - 1))
            return quantile_values[:, idx:idx+1]
        elif risk_type == 'CVaR':
            # Conditional Value at Risk
            idx = int(risk_param * self.num_quantiles)
            idx = max(1, min(idx, self.num_quantiles))
            return quantile_values[:, :idx].mean(dim=1, keepdim=True)
        else:
            return quantile_values.mean(dim=1, keepdim=True)

    def update(self, batch_size, training_steps, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=5e-3):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        q_new_actions = np.array([0.])
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # Generate quantile fractions
        tau = self.get_tau(batch_size)

        # Get new actions and log probs
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        new_action, log_prob, _, _, _ = self.policy_net.evaluate(state)

        # Normalize rewards
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

        # Update alpha
        if auto_entropy:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = min(args.maximum_alpha, self.log_alpha.exp().item())
        else:
            self.alpha = 0.2  # Fixed alpha value

        # Target quantile values
        with torch.no_grad():
            target_z1_values = self.target_zf1(next_state, new_next_action)
            target_z2_values = self.target_zf2(next_state, new_next_action)
            target_z_values = torch.min(target_z1_values, target_z2_values) - self.alpha * next_log_prob
            z_target = reward + (1. - done) * gamma * target_z_values

        # Current quantile values
        z1_pred = self.zf1(state, action)
        z2_pred = self.zf2(state, action)

        # Quantile regression loss
        zf1_loss = quantile_huber_loss(z1_pred, z_target, tau)
        zf2_loss = quantile_huber_loss(z2_pred, z_target, tau)

        # Update quantile networks
        self.zf1_optimizer.zero_grad()
        zf1_loss.backward()
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.zf1.parameters(), max_norm=1.0)
        self.zf1_optimizer.step()

        self.zf2_optimizer.zero_grad()
        zf2_loss.backward()
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.zf2.parameters(), max_norm=1.0)
        self.zf2_optimizer.step()

        # Policy loss
        if training_steps % args.critic_actor_ratio == 0:
            z1_new_actions = self.zf1(state, new_action)
            z2_new_actions = self.zf2(state, new_action)
            
            # Apply risk measure
            q1_new_actions = self.get_risk_weighted_q(z1_new_actions, args.risk_type, args.risk_param)
            q2_new_actions = self.get_risk_weighted_q(z2_new_actions, args.risk_type, args.risk_param)
            q_new_actions = torch.min(q1_new_actions, q2_new_actions)

            policy_loss = (self.alpha * log_prob - q_new_actions).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            if args.use_gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.policy_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.target_zf1.parameters(), self.zf1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for target_param, param in zip(self.target_zf2.parameters(), self.zf2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        # Record metrics
        q_values.append(q_new_actions.mean().item())
        log_probs.append(-log_prob.mean().item())
        alpha_values.append(self.alpha)
        zf_losses.append((zf1_loss.item() + zf2_loss.item()) / 2)

        return q_new_actions.mean()

    def save_model(self, path):
        torch.save(self.zf1.state_dict(), path + '_z1')
        torch.save(self.zf2.state_dict(), path + '_z2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.zf1.load_state_dict(torch.load(path + '_z1', weights_only=True))
        self.zf2.load_state_dict(torch.load(path + '_z2', weights_only=True))
        self.policy_net.load_state_dict(torch.load(path + '_policy', weights_only=True))

        self.zf1.eval()
        self.zf2.eval()
        self.policy_net.eval()


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(rewards, label="Reward")
    plt.legend()
    plt.title(f"DSAC Training Reward (risk_type={args.risk_type}, risk_param={args.risk_param})")
    
    plt.subplot(2, 3, 2)
    plt.plot(q_values_episode, label="Q-Value")
    plt.legend()
    plt.title("Q-Value")
    
    plt.subplot(2, 3, 3)
    plt.plot(log_probs_episode, label="Log Prob")
    plt.legend()
    plt.title("Log Probability")
    
    plt.subplot(2, 3, 4)
    plt.plot(alpha_values_episode, label="Alpha")
    plt.legend()
    plt.title("Alpha (Entropy Coefficient)")
    
    plt.subplot(2, 3, 5)
    plt.plot(zf_losses_episode, label="ZF Loss")
    plt.legend()
    plt.title("Quantile Function Loss")
    
    plt.subplot(2, 3, 6)
    # Plot recent performance
    recent_rewards = rewards[-50:] if len(rewards) > 50 else rewards
    plt.plot(recent_rewards, label="Recent Reward")
    plt.legend()
    plt.title("Recent 50 Episodes")

    plt.tight_layout()

    if not os.path.exists('pic'):
        os.makedirs('pic')
    
    subdir_name = f'dsac_risk_{args.risk_type}_param_{args.risk_param}'
    subdir_path = os.path.join('pic', subdir_name)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

    plt.savefig(os.path.join(subdir_path, f'dsac_monitoring.png'))
    plt.close()


# Initialize environment and parameters
replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)

debug = False
render = False
path = os.getcwd() + '/env'
env = env_bus(path, debug=debug)
env.reset()

action_dim = env.action_space.shape[0]
action_range = env.action_space.high[0]

# Training parameters
step = 0
step_trained = 0
max_episodes = 500
frame_idx = 0
explore_steps = 0
update_itr = 1
DETERMINISTIC = False
hidden_dim = 256  # Increased hidden dimension

# Monitoring variables
rewards = []
q_values = []
log_probs = []
alpha_values = []
zf_losses = []

q_values_episode = []
log_probs_episode = []
alpha_values_episode = []
zf_losses_episode = []

model_path = './model/dsac_v2'
tracemalloc.start()

dsac_trainer = DSAC_Trainer(env, replay_buffer, hidden_dim=hidden_dim, action_range=action_range, num_quantiles=args.num_quantiles)

if __name__ == '__main__':
    if args.train:
        # Training loop
        for eps in range(max_episodes):
            if eps != 0:
                env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=render)

            done = False
            episode_steps = 0
            training_steps = 0
            action_dict = {key: None for key in list(range(env.max_agent_num))}
            
            total_rewards, v_loss = 0, 0
            episode_reward = 0

            while not done:
                for key in state_dict:
                    if len(state_dict[key]) == 1:
                        if action_dict[key] is None:
                            if args.use_state_norm:
                                state_input = dsac_trainer.state_norm(np.array(state_dict[key][0]))
                            else:
                                state_input = np.array(state_dict[key][0])
                            a = dsac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)
                            action_dict[key] = a

                            if key == 2 and debug:
                                print('From DSAC Algorithm, when no state, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', a, ', reward: ', reward_dict[key])
                                print()

                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            if args.use_state_norm:
                                state = dsac_trainer.state_norm(np.array(state_dict[key][0]))
                                next_state = dsac_trainer.state_norm(np.array(state_dict[key][1]))
                            else:
                                state = np.array(state_dict[key][0])
                                next_state = np.array(state_dict[key][1])
                            if args.use_reward_scaling:
                                reward = dsac_trainer.reward_scaling(reward_dict[key])
                            else:
                                reward = reward_dict[key]

                            replay_buffer.push(state, action_dict[key], reward, next_state, done)
                            if key == 2 and debug:
                                print('From DSAC Algorithm store, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', action_dict[key], ', reward: ', reward_dict[key])
                                print()

                            episode_steps += 1
                            step += 1
                            episode_reward += reward_dict[key]

                        state_dict[key] = state_dict[key][1:]
                        if args.use_state_norm:
                            state_input = dsac_trainer.state_norm(np.array(state_dict[key][0]))
                        else:
                            state_input = np.array(state_dict[key][0])

                        action_dict[key] = dsac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)
                        
                        if key == 2 and debug:
                            print('From DSAC Algorithm run, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', action_dict[key], ', reward: ', reward_dict[key])
                            print()

                state_dict, reward_dict, done = env.step(action_dict, debug=debug, render=render)
                
                if len(replay_buffer) > args.batch_size and len(replay_buffer) % args.training_freq == 0 and step_trained != step:
                    step_trained = step
                    for i in range(update_itr):
                        _ = dsac_trainer.update(args.batch_size, training_steps, reward_scale=10., auto_entropy=args.auto_entropy, target_entropy=-1. * action_dim)
                        training_steps += 1

                if done:
                    replay_buffer.last_episode_step = episode_steps
                    break

            # Record episode metrics
            rewards.append(episode_reward)
            if training_steps > 0:
                q_values_episode.append(np.mean(q_values[-training_steps:]) if q_values[-training_steps:] else 0)
                log_probs_episode.append(np.mean(log_probs[-training_steps:]) if log_probs[-training_steps:] else 0)
                alpha_values_episode.append(np.mean(alpha_values[-training_steps:]) if alpha_values[-training_steps:] else 0)
                zf_losses_episode.append(np.mean(zf_losses[-training_steps:]) if zf_losses[-training_steps:] else 0)
            else:
                q_values_episode.append(0)
                log_probs_episode.append(0)
                alpha_values_episode.append(0)
                zf_losses_episode.append(0)

            if eps % args.plot_freq == 0:
                plot(rewards)
                np.save('rewards_dsac', rewards)
                torch.save(dsac_trainer.policy_net.state_dict(), model_path)

            replay_buffer_usage = len(replay_buffer) / replay_buffer_size * 100

            print(
                f"Episode: {eps} | Episode Reward: {episode_reward:.2f} "
                f"| CPU Memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB | "
                f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | "
                f"Replay Buffer Usage: {replay_buffer_usage:.2f}% | "
                f"Alpha: {dsac_trainer.alpha:.4f} | "
                f"Avg Q-Value: {q_values_episode[-1]:.2f} | "
                f"ZF Loss: {zf_losses_episode[-1]:.4f}")
        
        torch.save(dsac_trainer.policy_net.state_dict(), model_path)

    if args.test:
        dsac_trainer.policy_net.load_state_dict(torch.load(model_path))
        for eps in range(10):
            done = False
            env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=render)
            episode_reward = 0
            action_dict = {key: None for key in list(range(env.max_agent_num))}

            while not done:
                for key in state_dict:
                    if len(state_dict[key]) == 1:
                        if action_dict[key] is None:
                            state_input = np.array(state_dict[key][0])
                            a = dsac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=True)  # Use deterministic for testing
                            action_dict[key] = a
                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            episode_reward += reward_dict[key]

                        state_dict[key] = state_dict[key][1:]
                        state_input = np.array(state_dict[key][0])
                        action_dict[key] = dsac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=True)

                state_dict, reward_dict, done = env.step(action_dict)
            print('Episode: ', eps, '| Episode Reward: ', episode_reward)