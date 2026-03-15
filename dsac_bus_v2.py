"""
DSAC (Distributional Soft Actor Critic) for Bus Environment
True distributional implementation with proper quantile regression
"""

import psutil
import tracemalloc
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
from copy import deepcopy
import json

from bus_feature_utils import create_embedding_layer, build_bus_categorical_info
from bus_replay_buffer import ReplayBuffer

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
parser.add_argument("--plot_freq", type=int, default=5, help="frequency of plotting the result")
parser.add_argument('--auto_entropy', type=bool, default=True, help='automatically updating alpha')
parser.add_argument("--maximum_alpha", type=float, default=0.3, help="max entropy weight")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--num_quantiles", type=int, default=32, help="number of quantiles")
parser.add_argument("--tau_type", type=str, default='iqn', help="quantile fraction type: fix, iqn, fqf")
parser.add_argument("--risk_type", type=str, default='CVaR', help="risk type: neutral, VaR, cvar, wang, cpw")
parser.add_argument("--critic_actor_ratio", type=int, default=2, help="ratio of critic updates to actor updates")
parser.add_argument("--risk_param", type=float, default=0.0, help="risk parameter")
parser.add_argument("--max_episodes", type=int, default=500, help="maximum number of training episodes")
parser.add_argument('--save_root', type=str, default='.', help='Base directory for saving models, logs, and figures')
parser.add_argument('--run_name', type=str, default='gpt_version', help='Optional identifier appended to save directories to avoid overwriting previous runs')
parser.add_argument('--env_path', type=str, default='env', help='Path to the environment configuration directory')
parser.add_argument('--embedding_mode', type=str, default='full', choices=['full', 'one_hot', 'none'], help='Categorical feature handling strategy')
parser.add_argument('--route_sigma', type=float, default=1.5, help='Sigma used for route speed sampling')
parser.add_argument('--eval_sigmas', type=float, nargs='*', default=None, help='List of sigma values for cross-evaluation after training')
parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension size for DSAC v2 networks')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for critic, policy, and alpha optimizers')
args = parser.parse_args()
args.embedding_mode = args.embedding_mode.lower()

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RUN_NAME = args.run_name.strip() if args.run_name else None
SAVE_ROOT = os.path.abspath(args.save_root)

sigma_token = f"sigma{args.route_sigma}".replace('.', 'p')
experiment_components = [SCRIPT_NAME, sigma_token, f"embed-{args.embedding_mode}", f"risk-{args.risk_type}", f"tau-{args.tau_type}"]
if RUN_NAME:
    experiment_components.append(RUN_NAME)
EXPERIMENT_ID = "_".join(experiment_components)

PIC_DIR = os.path.join(SAVE_ROOT, 'pic', EXPERIMENT_ID)
LOG_DIR = os.path.join(SAVE_ROOT, 'logs', EXPERIMENT_ID)
MODEL_DIR = os.path.join(SAVE_ROOT, 'model', EXPERIMENT_ID)

for directory in (PIC_DIR, LOG_DIR, MODEL_DIR):
    os.makedirs(directory, exist_ok=True)

with open(os.path.join(LOG_DIR, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)

MODEL_PREFIX = os.path.join(MODEL_DIR, 'dsac_bus_v2')


class QuantileNetwork(nn.Module):
    """Distributional Q-network that outputs quantile values"""
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, num_quantiles=32, embedding_dim=64, init_w=3e-3):
        super(QuantileNetwork, self).__init__()
        
        self.embedding_layer = embedding_layer
        self.num_quantiles = num_quantiles
        self.embedding_dim = embedding_dim
        
        # State-action feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Quantile embedding network (cosine embedding)
        self.quantile_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
        )
        
        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize final layer
        self.value_net[-1].weight.data.uniform_(-init_w, init_w)
        self.value_net[-1].bias.data.uniform_(-init_w, init_w)
        
        # Cosine embedding constants
        self.register_buffer('const_vec', torch.arange(1, embedding_dim + 1, dtype=torch.float32))

    def forward(self, state, action, tau):
        """
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
            tau: (batch_size, num_quantiles) - quantile fractions
        Returns:
            quantiles: (batch_size, num_quantiles) - quantile values
        """
        # Process categorical and numerical features
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]
        
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
        
        # Extract state-action features
        sa_features = torch.cat([state_with_embeddings, action], dim=1)
        sa_features = self.feature_net(sa_features)  # (batch_size, hidden_size)
        
        # Quantile embedding using cosine functions
        batch_size = tau.shape[0]
        n_tau = tau.shape[1]
        
        # Cosine embedding: cos(i * pi * tau) for i = 1, ..., embedding_dim
        tau_embed = torch.cos(tau.unsqueeze(-1) * self.const_vec.unsqueeze(0).unsqueeze(0) * np.pi)  # (batch_size, n_tau, embedding_dim)
        tau_features = self.quantile_net(tau_embed)  # (batch_size, n_tau, hidden_size)
        
        # Element-wise multiplication and sum
        sa_features_expanded = sa_features.unsqueeze(1).expand(-1, n_tau, -1)  # (batch_size, n_tau, hidden_size)
        combined_features = sa_features_expanded * tau_features  # (batch_size, n_tau, hidden_size)
        
        # Output quantile values
        quantiles = self.value_net(combined_features).squeeze(-1)  # (batch_size, n_tau)
        
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


def distortion_risk_measure(quantiles, tau, risk_type='neutral', risk_param=0.0):
    """
    Apply risk distortion to quantile values
    
    Args:
        quantiles: (batch_size, num_quantiles)
        tau: (batch_size, num_quantiles) - quantile fractions
        risk_type: type of risk measure
        risk_param: risk parameter
    """
    if risk_type == 'neutral':
        # Risk-neutral (expectation)
        weights = torch.ones_like(tau) / tau.shape[1]
        return (weights * quantiles).sum(dim=1, keepdim=True)
    
    elif risk_type == 'VaR':
        # Value at Risk
        target_quantile = risk_param
        # Find closest quantile
        _, indices = torch.min(torch.abs(tau - target_quantile), dim=1, keepdim=True)
        return torch.gather(quantiles, 1, indices)
    
    elif risk_type == 'CVaR' or risk_type == 'cvar':
        # Conditional Value at Risk
        alpha = risk_param
        mask = (tau <= alpha).float()
        weights = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)
        return (weights * quantiles).sum(dim=1, keepdim=True)
    
    elif risk_type == 'wang':
        # Wang transform: Φ(Φ^(-1)(τ) + β)
        from torch.distributions import Normal
        std_normal = Normal(0, 1)
        transformed_tau = std_normal.cdf(std_normal.icdf(torch.clamp(tau, 1e-6, 1-1e-6)) + risk_param)
        weights = torch.gradient(transformed_tau, dim=1)[0]
        weights = F.softmax(weights, dim=1)  # Normalize
        return (weights * quantiles).sum(dim=1, keepdim=True)
    
    elif risk_type == 'cpw':
        # Cumulative Prospect Theory weighting
        beta = risk_param if risk_param > 0 else 0.71
        transformed_tau = tau ** beta / (tau ** beta + (1 - tau) ** beta) ** (1 / beta)
        weights = torch.gradient(transformed_tau, dim=1)[0]
        weights = F.softmax(weights, dim=1)  # Normalize
        return (weights * quantiles).sum(dim=1, keepdim=True)
    
    else:
        # Default to risk-neutral
        weights = torch.ones_like(tau) / tau.shape[1]
        return (weights * quantiles).sum(dim=1, keepdim=True)


class DSAC_Trainer():
    def __init__(self, env, replay_buffer, hidden_dim, action_range, num_quantiles=32, embedding_mode='full'):
        cat_cols, cat_code_dict = build_bus_categorical_info(env)

        self.num_cat_features = len(cat_cols)
        self.num_cont_features = env.state_dim - self.num_cat_features
        self.num_quantiles = num_quantiles

        embedding_kwargs = {'layer_norm': True, 'dropout': 0.05} if embedding_mode == 'full' else {}
        embedding_template = create_embedding_layer(embedding_mode, cat_code_dict, cat_cols, **embedding_kwargs)
        state_dim = embedding_template.output_dim + self.num_cont_features

        self.replay_buffer = replay_buffer

        # Networks
        self.zf1 = QuantileNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone(), num_quantiles).to(device)
        self.zf2 = QuantileNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone(), num_quantiles).to(device)
        self.target_zf1 = deepcopy(self.zf1).to(device)
        self.target_zf2 = deepcopy(self.zf2).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone(), action_range).to(device)
        
        # Initialize target networks
        for target_param, param in zip(self.target_zf1.parameters(), self.zf1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_zf2.parameters(), self.zf2.parameters()):
            target_param.data.copy_(param.data)

        # Alpha for entropy regularization
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        self.alpha = args.maximum_alpha

        print('Quantile Networks (1,2): ', self.zf1)
        print('Policy Network: ', self.policy_net)

        zf_lr = policy_lr = alpha_lr = args.lr

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

    def get_tau(self, batch_size, num_quantiles):
        """Generate quantile fractions based on tau_type"""
        if args.tau_type == 'fix':
            # Fixed uniform quantiles
            tau = torch.linspace(0.0, 1.0, num_quantiles + 2)[1:-1]  # Remove 0 and 1
            tau = tau.unsqueeze(0).expand(batch_size, -1).to(device)
        elif args.tau_type == 'iqn':
            # Random uniform sampling (IQN style)
            tau = torch.rand(batch_size, num_quantiles).to(device)
            tau = tau.sort(dim=1)[0]  # Sort to maintain order
        else:
            # Default to IQN
            tau = torch.rand(batch_size, num_quantiles).to(device)
            tau = tau.sort(dim=1)[0]
        
        return tau

    def update(self, batch_size, training_steps, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=5e-3):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        q_new_actions = np.array([0.])
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # Generate quantile fractions
        tau = self.get_tau(batch_size, self.num_quantiles)
        tau_next = self.get_tau(batch_size, self.num_quantiles)

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
            self.alpha = 0.2

        # Target quantile values
        with torch.no_grad():
            target_z1_values = self.target_zf1(next_state, new_next_action, tau_next)
            target_z2_values = self.target_zf2(next_state, new_next_action, tau_next)
            target_z_values = torch.min(target_z1_values, target_z2_values) - self.alpha * next_log_prob
            z_target = reward + (1. - done) * gamma * target_z_values

        # Current quantile values
        z1_pred = self.zf1(state, action, tau)
        z2_pred = self.zf2(state, action, tau)

        # Quantile regression loss
        zf1_loss = quantile_regression_loss(z1_pred, z_target, tau)
        zf2_loss = quantile_regression_loss(z2_pred, z_target, tau)

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

        # Policy loss with risk measure
        if training_steps % args.critic_actor_ratio ==0:
            new_tau = self.get_tau(batch_size, self.num_quantiles)
            z1_new_actions = self.zf1(state, new_action, new_tau)
            z2_new_actions = self.zf2(state, new_action, new_tau)
            
            # Apply risk measure to get Q-values
            q1_new_actions = distortion_risk_measure(z1_new_actions, new_tau, args.risk_type, args.risk_param)
            q2_new_actions = distortion_risk_measure(z2_new_actions, new_tau, args.risk_type, args.risk_param)
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


def evaluate_policy(dsac_trainer, env, num_eval_episodes=5, deterministic=True):
    eval_rewards = []

    for _ in range(num_eval_episodes):
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
                        if args.use_state_norm:
                            state_input = dsac_trainer.state_norm(raw_state, update=False)
                        else:
                            state_input = raw_state
                        a = dsac_trainer.policy_net.get_action(
                            torch.from_numpy(state_input).float(), deterministic=deterministic
                        )
                        action_dict[key] = a
                elif len(state_dict[key]) == 2:
                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        episode_reward += reward_dict[key]

                    state_dict[key] = state_dict[key][1:]
                    raw_state = np.array(state_dict[key][0])
                    if args.use_state_norm:
                        state_input = dsac_trainer.state_norm(raw_state, update=False)
                    else:
                        state_input = raw_state
                    action_dict[key] = dsac_trainer.policy_net.get_action(
                        torch.from_numpy(state_input).float(), deterministic=deterministic
                    )

            state_dict, reward_dict, done = env.step(action_dict, render=False)

        eval_rewards.append(episode_reward)

    mean_reward = np.mean(eval_rewards)
    reward_std = np.std(eval_rewards)

    return mean_reward, reward_std


def plot(rewards):
    pass


# Initialize environment and parameters
replay_buffer_size = int(1e6)
replay_buffer = ReplayBuffer(replay_buffer_size)

debug = False
render = False
path = os.path.abspath(args.env_path)
env = env_bus(path, debug=debug, route_sigma=args.route_sigma)
env.reset()

action_dim = env.action_space.shape[0]
action_range = env.action_space.high[0]

# Training parameters
step = 0
step_trained = 0
frame_idx = 0
explore_steps = 0
update_itr = 1
DETERMINISTIC = False
hidden_dim = args.hidden_dim

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

eval_episodes = []
eval_mean_rewards = []
eval_reward_stds = []

tracemalloc.start()

dsac_trainer = DSAC_Trainer(
    env,
    replay_buffer,
    hidden_dim=hidden_dim,
    action_range=action_range,
    num_quantiles=args.num_quantiles,
    embedding_mode=args.embedding_mode,
)

if __name__ == '__main__':
    if args.train:
        # Training loop
        for eps in range(args.max_episodes):
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
                recent_q = q_values[-training_steps:] if training_steps <= len(q_values) else q_values
                recent_log_prob = log_probs[-training_steps:] if training_steps <= len(log_probs) else log_probs
                recent_alpha = alpha_values[-training_steps:] if training_steps <= len(alpha_values) else alpha_values
                recent_zf = zf_losses[-training_steps:] if training_steps <= len(zf_losses) else zf_losses

                q_values_episode.append(float(np.mean(recent_q)) if recent_q else 0.0)
                log_probs_episode.append(float(np.mean(recent_log_prob)) if recent_log_prob else 0.0)
                alpha_values_episode.append(float(np.mean(recent_alpha)) if recent_alpha else 0.0)
                zf_losses_episode.append(float(np.mean(recent_zf)) if recent_zf else 0.0)
            else:
                q_values_episode.append(0.0)
                log_probs_episode.append(0.0)
                alpha_values_episode.append(0.0)
                zf_losses_episode.append(0.0)

            if eps % args.plot_freq == 0:
                plot(rewards)
                np.save(os.path.join(LOG_DIR, 'rewards.npy'), np.array(rewards, dtype=np.float32))
                np.save(os.path.join(LOG_DIR, 'q_values_episode.npy'), np.array(q_values_episode, dtype=np.float32))
                np.save(os.path.join(LOG_DIR, 'log_probs_episode.npy'), np.array(log_probs_episode, dtype=np.float32))
                np.save(os.path.join(LOG_DIR, 'alpha_values_episode.npy'), np.array(alpha_values_episode, dtype=np.float32))
                np.save(os.path.join(LOG_DIR, 'zf_losses_episode.npy'), np.array(zf_losses_episode, dtype=np.float32))

                mean_reward, reward_std = evaluate_policy(dsac_trainer, env, num_eval_episodes=10, deterministic=True)
                eval_episodes.append(eps)
                eval_mean_rewards.append(mean_reward)
                eval_reward_stds.append(reward_std)
                np.save(os.path.join(LOG_DIR, 'eval_episodes.npy'), np.array(eval_episodes, dtype=np.int32))
                np.save(os.path.join(LOG_DIR, 'eval_mean_rewards.npy'), np.array(eval_mean_rewards, dtype=np.float32))
                np.save(os.path.join(LOG_DIR, 'eval_reward_stds.npy'), np.array(eval_reward_stds, dtype=np.float32))

                model_name = f"{MODEL_PREFIX}_episode_{eps}"
                dsac_trainer.save_model(model_name)
                dsac_trainer.save_model(os.path.join(LOG_DIR, f'{SCRIPT_NAME}_episode_{eps}'))

            replay_buffer_usage = len(replay_buffer) / replay_buffer_size * 100

            print(
                f"[DSAC-V2 | risk={args.risk_type}, param={args.risk_param}, max_alpha={args.maximum_alpha}, tau={args.tau_type}] Episode: {eps} | Episode Reward: {episode_reward:.2f} "
                f"| CPU Memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB | "
                f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | "
                f"Replay Buffer Usage: {replay_buffer_usage:.2f}% | "
                f"Alpha: {dsac_trainer.alpha:.4f} | "
                f"Avg Q-Value: {q_values_episode[-1]:.2f} | "
                f"ZF Loss: {zf_losses_episode[-1]:.4f}")
        
        dsac_trainer.save_model(MODEL_PREFIX)

        plot(rewards)
        np.save(os.path.join(LOG_DIR, 'rewards.npy'), np.array(rewards, dtype=np.float32))
        np.save(os.path.join(LOG_DIR, 'q_values_episode.npy'), np.array(q_values_episode, dtype=np.float32))
        np.save(os.path.join(LOG_DIR, 'log_probs_episode.npy'), np.array(log_probs_episode, dtype=np.float32))
        np.save(os.path.join(LOG_DIR, 'alpha_values_episode.npy'), np.array(alpha_values_episode, dtype=np.float32))
        np.save(os.path.join(LOG_DIR, 'zf_losses_episode.npy'), np.array(zf_losses_episode, dtype=np.float32))

        mean_reward, reward_std = evaluate_policy(dsac_trainer, env, num_eval_episodes=15, deterministic=True)
        print(f"最终评估结果: 平均奖励 = {mean_reward:.2f}, 标准差 = {reward_std:.2f}")
        final_eval_episode = args.max_episodes - 1
        eval_episodes.append(final_eval_episode)
        eval_mean_rewards.append(mean_reward)
        eval_reward_stds.append(reward_std)
        np.save(os.path.join(LOG_DIR, 'eval_episodes.npy'), np.array(eval_episodes, dtype=np.int32))
        np.save(os.path.join(LOG_DIR, 'eval_mean_rewards.npy'), np.array(eval_mean_rewards, dtype=np.float32))
        np.save(os.path.join(LOG_DIR, 'eval_reward_stds.npy'), np.array(eval_reward_stds, dtype=np.float32))

        if args.eval_sigmas:
            sigma_results = []
            for sigma in args.eval_sigmas:
                eval_env = env_bus(path, debug=debug, route_sigma=sigma)
                eval_env.reset()
                sigma_mean, sigma_std = evaluate_policy(dsac_trainer, eval_env, num_eval_episodes=10, deterministic=True)
                sigma_results.append((sigma, sigma_mean, sigma_std))
            np.save(os.path.join(LOG_DIR, 'eval_cross_sigma.npy'), np.array(sigma_results, dtype=np.float32))

    if args.test:
        dsac_trainer.load_model(MODEL_PREFIX)
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
