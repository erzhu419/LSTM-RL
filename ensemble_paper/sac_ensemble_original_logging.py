'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''

import psutil, tracemalloc
import gym
import copy
from tqdm import tqdm
import gc
import time
import torch, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from normalization import Normalization, RewardScaling, RunningMeanStd

from IPython.display import clear_output
import matplotlib.pyplot as plt
from env_original.sim import env_bus
import os
import argparse
import json
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

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--max_episodes', type=int, default=500, help='number of episodes to train')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--use_gradient_clip', type=bool, default=True, help="Trick 1:gradient clipping")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")
parser.add_argument("--training_freq", type=int, default=5, help="frequency of training the network")
parser.add_argument("--plot_freq", type=int, default=1, help="frequency of plotting the result")
parser.add_argument('--weight_reg', type=float, default=0.01, help='weight of regularization')
parser.add_argument('--auto_entropy', type=bool, default=True, help='automatically updating alpha')
parser.add_argument("--maximum_alpha", type=float, default=0.6, help="max entropy weight")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
#TODO 可以看到这里把beta相关的三个参数降低之后，收敛性好很多，继续调参
parser.add_argument("--beta_bc", type=float, default=0.001, help="weight of behavior cloning loss")
# beta这个参数在源代码中是负数(我开始也奇怪为什么下面代码关于ood_std是+,原来是因为这里是负数)
parser.add_argument("--beta", type=float, default=-2, help="weight of variance")
parser.add_argument("--beta_ood", type=float, default=0.01, help="weight of OOD loss")
parser.add_argument('--critic_actor_ratio', type=int, default=2, help="ratio of critic and actor training")
parser.add_argument('--replay_buffer_size', type=int, default=int(1e6), help="buffer size")
parser.add_argument('--route_sigma', type=float, default=1.5, help='Sigma used for route speed sampling')
parser.add_argument('--eval_sigmas', type=float, nargs='*', default=None, help='List of sigma values for cross-evaluation after training')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
parser.add_argument('--save_root', type=str, default='test_sign_before_regnorm', help='Base directory for saving models')
parser.add_argument('--run_name', type=str, default=None, help='Specific run name (if provided, overrides auto-generated parameter path)')
parser.add_argument("--ensemble_size", type=int, default=10, help="Number of models in the ensemble")

args = parser.parse_args()

ROUTE_SIGMA_TOKEN = f"route_sigma_{str(args.route_sigma).replace('.', 'p')}"


class ReplayBuffer:
    def __init__(self, capacity, last_episode_step=5000):
        self.capacity = capacity
        self.last_episode_step = last_episode_step
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Add new data to buffer, overwriting old if full"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample batch_size elements in O(1)"""
        # Optimized sampling: generates random indices instead of copying list
        batch_indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in batch_indices]
        
        # Unzip and convert to numpy
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return np.stack(states), np.stack(actions), np.array(rewards, dtype=np.float32), \
               np.stack(next_states), np.array(dones, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


class EmbeddingLayer(nn.Module):
    def __init__(self, cat_code_dict, cat_cols):
        super(EmbeddingLayer, self).__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = cat_cols

        # Create embedding layers for categorical variables
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

        # Concatenate all embeddings
        embed_tensor = torch.cat(embedding_tensor_group, dim=1)
        return embed_tensor


class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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


class VectorizedCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_critics, embedding_layer):
        super().__init__()
        self.embedding_layer = embedding_layer # EmbeddingLayer initialization
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )

        self.num_critics = num_critics

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(self.num_critics, dim=0)
        q_values = self.critic(state_action).squeeze(-1)
        return q_values


# Replace original SoftQNetwork with vectorized version
class SoftQNetwork(VectorizedCritic):
    def __init__(self, state_dim, action_dim, hidden_dim, embedding_layer, ensemble_size=10):
        # compute input dim after embedding

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_critics=ensemble_size,
            embedding_layer=embedding_layer
        )

        self.ensemble_size = ensemble_size

    def forward(self, state, action):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)
        return super().forward(state_with_embeddings, action)


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

        mean = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()  # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range / 2 * action_0 + self.action_range / 2  # bounded action
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape).to(device)
        action = self.action_range / 2 * torch.tanh(mean + std * z) + self.action_range / 2

        action = self.action_range / 2 * torch.tanh(mean).detach().cpu().numpy()[0] + self.action_range / 2 if deterministic else action.detach().cpu().numpy()[0]
        return action


class SAC_Trainer():
    def __init__(self, env, replay_buffer, hidden_dim, action_range, ensemble_size=10):
        # 以下是类别特征和数值特征
        cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']
        cat_code_dict = {
            'bus_id': {i: i for i in range(env.max_agent_num)},  # 最大车辆数，预设值
            'station_id': {i: i for i in range(round(len(env.stations) / 2))},  # station_id，有几个站就有几个类别
            'time_period': {i: i for i in range(env.timetables[-1].launch_time // 3600 + 2)},  # time period,以每小时区分，+2是因为让车运行完
            'direction': {0: 0, 1: 1}  # direction 二分类
        }
        # 数值特征的数量
        self.num_cat_features = len(cat_cols)
        self.num_cont_features = env.state_dim - self.num_cat_features  # 包括 forward_headway, backward_headway 和最后一个 feature
        # 创建嵌入层
        embedding_layer = EmbeddingLayer(cat_code_dict, cat_cols)
        # SAC 网络的输入维度
        embedding_dim = sum([min(50, len(cat_code_dict[col]) // 2) for col in cat_cols])  # 总嵌入维度
        state_dim = embedding_dim + self.num_cont_features  # 状态维度 = 嵌入维度 + 数值特征维度

        self.replay_buffer = replay_buffer

        self.soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_layer, ensemble_size=ensemble_size).to(device)
        self.target_soft_q_net = deepcopy(self.soft_q_net)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_layer, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network: ', self.soft_q_net)
        print('Policy Network: ', self.policy_net)

        self.soft_q_criterion = nn.MSELoss()

        soft_q_lr = 1e-5
        policy_lr = 1e-5
        alpha_lr = 1e-5

        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # 初始化RunningMeanStd
        initial_mean = [360., 360., 90.]
        initial_std = [165., 133., 45.]

        running_ms = RunningMeanStd(shape=(self.num_cont_features,), init_mean=initial_mean, init_std=initial_std)

        self.state_norm = Normalization(num_categorical=self.num_cat_features, num_numerical=self.num_cont_features, running_ms=running_ms)
        self.reward_scaling = RewardScaling(shape=1, gamma=0.99)

    # Q loss computation
    def compute_q_loss(self, state, action, reward, next_state, done, new_next_action, next_log_prob, reg_norm, gamma):
        predicted_q_value = self.soft_q_net(state, action)  # shape: [ensemble_size, batch, 1]
        # with torch.no_grad():
        target_q_next = self.target_soft_q_net(next_state, new_next_action)  # shape: [ensemble_size, batch, 1]
        next_log_prob = next_log_prob.unsqueeze(0).repeat(self.soft_q_net.num_critics, 1)  # Expand and repeat for ensemble_size
        reg_norm = reg_norm.unsqueeze(-1).repeat(1, args.batch_size)  # Adjust shape to match target_q_next
        target_q_next = target_q_next - self.alpha * next_log_prob + args.weight_reg * reg_norm  # shape: [ensemble_size, batch, 1]
        target_q_value = reward + (1 - done) * gamma * target_q_next.unsqueeze(-1)

        ood_loss = predicted_q_value.std(0).mean()
        q_value_loss = self.soft_q_criterion(predicted_q_value, target_q_value.squeeze(-1).detach())
        loss = q_value_loss + args.beta_ood * ood_loss
        return loss, predicted_q_value, ood_loss

    # Policy loss computation
    def compute_policy_loss(self, state, action, new_action, log_prob, reg_norm):

        reg_norm = reg_norm.unsqueeze(-1).repeat(1, args.batch_size)  # Adjust shape to match target_q_next

        q_values_dist = self.soft_q_net(state, new_action) + args.weight_reg * reg_norm - self.alpha * log_prob

        q_mean = q_values_dist.mean(dim=0)
        q_std = q_values_dist.std(dim=0)
        q_loss = -(q_mean + args.beta * q_std).mean()

        bc_loss = F.mse_loss(new_action, action)
        # smooth_loss = self.get_policy_smooth_loss(state)

        loss = args.beta_bc * bc_loss + q_loss

        return loss, q_loss, q_std

    # Smooth loss regularization (based on LCB get_policy_loss style)
    # def get_policy_smooth_loss(self, state, noise_std=0.2):
    #     obs_repeat = state.unsqueeze(0).repeat(self.soft_q_net.ensemble_size, 1, 1)  # [ensemble, batch, state_dim]
    #     obs_flat = obs_repeat.view(-1, state.shape[1])
    #     pi_action, _, _, _ = self.policy_net(obs_flat)
    #     pi_action = pi_action.view(self.soft_q_net.ensemble_size, -1, pi_action.shape[-1])
    #
    #     noise = noise_std * torch.randn_like(pi_action)
    #     noisy_action = torch.clamp(pi_action + noise, -1.0, 1.0)
    #
    #     smooth_loss = F.mse_loss(pi_action, noisy_action)
    #     return smooth_loss

    # Alpha loss computation (entropy regularization)
    def compute_alpha_loss(self, log_prob, target_entropy):
        alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
        return alpha_loss

    # Regularization term computation
    def compute_reg_norm(self, model):
        weight_norm, bias_norm = [], []
        for name, param in model.named_parameters():
            if 'critic' in name:  # Only include parameters from the critic
                if 'weight' in name:
                    weight_norm.append(torch.norm(param, p=1, dim=[1, 2]))  # Keep the first dimension (10,)
                elif 'bias' in name:
                    bias_norm.append(torch.norm(param, p=1, dim=[1, 2]))  # Keep the first dimension (10,)
        reg_norm = torch.sum(torch.stack(weight_norm), dim=0) + torch.sum(torch.stack(bias_norm[:-1]), dim=0)  # Final shape [10,]
        return reg_norm

    def update(self, batch_size, training_steps, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        global q_values, reg_norms1, reg_norms2, log_probs, alpha_values, ood_losses, q_stds
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        
        # Avoid division by zero in reward scaling
        reward_std = reward.std(dim=0)
        reward_std = torch.where(reward_std < 1e-6, torch.tensor(1.0, device=device), reward_std)
        reward = reward_scale * (reward - reward.mean(dim=0)) / reward_std

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy:
            alpha_loss = self.compute_alpha_loss(log_prob, target_entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=False)
            self.alpha_optimizer.step()
            self.alpha = min(args.maximum_alpha, self.log_alpha.exp().item())
        else:
            self.alpha = 1.
            alpha_loss = 0

        reg_norm = self.compute_reg_norm(self.target_soft_q_net)

        q_value_loss, predicted_q_value, ood_loss = self.compute_q_loss(state, action, reward, next_state, done, new_next_action, next_log_prob, reg_norm, gamma)
        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward(retain_graph=False)
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.soft_q_net.parameters(), max_norm=1.0)
        self.soft_q_optimizer.step()

        if training_steps % args.critic_actor_ratio == 0:
            policy_loss, predicted_new_q_value, q_std= self.compute_policy_loss(state, action, new_action, log_prob, reg_norm)
            q_stds.append(q_std.mean().item())

            self.policy_optimizer.zero_grad()

            policy_loss.backward(retain_graph=False)
            self.policy_optimizer.step()

        for target_param, param in zip(self.target_soft_q_net.parameters(), self.soft_q_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        # --- LOGGING INJECTION START ---
        # Record one Q-value (mean of predicted_q_value)
        q_values.append(predicted_q_value.mean().item())
        # Assuming reg_norm is a tensor of shape [num_critics]
        # If reg_norm is a single value, use .item()
        if reg_norm.numel() > 1:
             reg_norms1.append(args.weight_reg * reg_norm[0].item())
             reg_norms2.append(args.weight_reg * reg_norm[1].item())
        else:
             reg_norms1.append(args.weight_reg * reg_norm.item())
             reg_norms2.append(args.weight_reg * reg_norm.item())
             
        log_probs.append(-log_prob.mean().item())
        alpha_values.append(self.alpha)
        ood_losses.append(ood_loss.item())
        # --- LOGGING INJECTION END ---

        return predicted_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net.state_dict(), path + '_q')
        torch.save(self.policy_net.state_dict(), path + '_policy')
        torch.save(self.state_norm, path + '_norm')

    def load_model(self, path):
        self.soft_q_net.load_state_dict(torch.load(path + '_q', weights_only=True))
        self.policy_net.load_state_dict(torch.load(path + '_policy', weights_only=True))

        self.soft_q_net.eval()
        self.policy_net.eval()


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward")
    plt.legend()
    plt.title(f"Training Reward (weight_reg={args.weight_reg}, auto_entropy={args.auto_entropy}, reward_scaling={args.use_reward_scaling}, maximum_alpha={args.maximum_alpha})")
    plt.subplot(1, 2, 2)

    # Use the new episode-level logging lists
    plt.plot(q_values_episode, label="Q-Value Mean")
    plt.plot(reg_norms1_episode, label="Reg Norm 1")
    plt.plot(reg_norms2_episode, label="Reg Norm 2")
    plt.plot(log_probs_episode, label="Log Prob")
    plt.plot(alpha_values_episode, label="Alpha")
    plt.plot(ood_losses_episode, label="OOD Loss")
    plt.plot(q_stds_episode, label="Q Std")

    plt.legend()
    plt.title(f"Q-Value & V-Value and log_prob & regularization Monitoring (weight_reg={args.weight_reg})")

    os.makedirs(pic_path, exist_ok=True)
    plt.savefig(os.path.join(pic_path, 'sac_monitoring.png'))
    plt.close()


def evaluate_policy(trainer, environment, num_eval_episodes=10, deterministic=True):
    episode_rewards = []
    for _ in range(num_eval_episodes):
        done = False
        environment.reset()
        state_dict, reward_dict, _ = environment.initialize_state(render=False)
        action_dict = {key: None for key in list(range(environment.max_agent_num))}
        episode_reward = 0

        while not done:
            for key in state_dict:
                agent_states = state_dict[key]
                if len(agent_states) == 1:
                    if action_dict[key] is None:
                        state_input = np.array(agent_states[0])
                        action = trainer.policy_net.get_action(
                            torch.from_numpy(state_input).float(),
                            deterministic=deterministic
                        )
                        action_dict[key] = action
                elif len(agent_states) == 2:
                    if agent_states[0][1] != agent_states[1][1]:
                        episode_reward += reward_dict[key]

                    state_dict[key] = agent_states[1:]
                    state_input = np.array(state_dict[key][0])
                    action_dict[key] = trainer.policy_net.get_action(
                        torch.from_numpy(state_input).float(),
                        deterministic=deterministic
                    )

            state_dict, reward_dict, done = environment.step(action_dict)

        episode_rewards.append(episode_reward)

    if not episode_rewards:
        return 0.0, 0.0

    rewards_array = np.array(episode_rewards, dtype=np.float32)
    return float(rewards_array.mean()), float(rewards_array.std(ddof=0))

replay_buffer = ReplayBuffer(args.replay_buffer_size)

debug = False
render = False
path = os.getcwd() + '/env_original'
env = env_bus(path, debug=debug, route_sigma=args.route_sigma)
env.reset()

action_dim = env.action_space.shape[0]
action_range = env.action_space.high[0]

# hyperparameters for RL training

step = 0
step_trained = 0
frame_idx = 0
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY = True
DETERMINISTIC = False
hidden_dim = args.hidden_dim

# --- LOGGING INITIALIZATION for GLOBAL SCOPE ---
rewards = []  # 记录奖励
q_values = []  # 记录 Q 值变化 (per step)
reg_norms1 = [] # 记录正则化项1 (per step)
reg_norms2 = [] # 记录正则化项2 (per step)
log_probs = []  # 记录 log_prob (per step)
alpha_values = []  # 记录 alpha 值 (per step)
ood_losses = [] # OOD losses (per step)
q_stds = [] # Q value standard deviations (per step)

q_values_episode = []  # 记录每个 episode 的 Q 值
reg_norms1_episode = []  # 记录每个 episode 的正则化项1
reg_norms2_episode = []  # 记录每个 episode 的正则化项2
log_probs_episode = []  # 记录每个 episode 的 log_prob
alpha_values_episode = []  # 记录每个 episode 的 alpha 值
ood_losses_episode = [] # OOD losses (per episode)
q_stds_episode = [] # Q value standard deviations (per episode)

eval_episodes = []
eval_mean_rewards = []
eval_reward_stds = []
# ---------------------------------------------

# Define param_str for consistent path naming
param_str = (
    f"{ROUTE_SIGMA_TOKEN}/"
    f"replay_buffer_size_{args.replay_buffer_size}/"
    f"critic_actor_ratio_{args.critic_actor_ratio}/"
    f"maximum_alpha_{args.maximum_alpha}/"
    f"weight_reg_{args.weight_reg}"
)

if args.run_name:
    model_path = os.path.join(args.save_root, 'model', args.run_name)
    logs_path = os.path.join(args.save_root, 'logs', args.run_name)
    pic_path = os.path.join(args.save_root, 'pic', args.run_name)
else:
    model_path = os.path.join(args.save_root, 'model', param_str)
    logs_path = os.path.join(args.save_root, 'logs', param_str)
    pic_path = os.path.join(args.save_root, 'pic', param_str)

os.makedirs(model_path, exist_ok=True)
os.makedirs(logs_path, exist_ok=True)
os.makedirs(pic_path, exist_ok=True)

# tracemalloc.start()

sac_trainer = SAC_Trainer(
    env,
    replay_buffer,
    hidden_dim=hidden_dim,
    action_range=action_range,
    ensemble_size=args.ensemble_size
)

if __name__ == '__main__':
    if args.train:
        # training loop
        for eps in range(args.max_episodes):
            episode_start_time = time.time()
            if eps != 0:
                env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=render)

            done = False
            episode_steps = 0
            training_steps = 0  # 记录已经训练了多少次
            action_dict = {key: None for key in list(range(env.max_agent_num))}
            action_dict_zero = {key: 0 for key in list(range(env.max_agent_num))}  # 全0的action，用于查看reward的上限
            action_dict_twenty = {key: 20 for key in list(range(env.max_agent_num))}  # 全20的action，用于查看reward的上限

            prob_dict = {key: None for key in list(range(env.max_agent_num))}
            v_dict = {key: None for key in list(range(env.max_agent_num))}
            total_rewards, v_loss = 0, 0

            episode_reward = 0
            
            # Track lengths before episode starts - REMOVED for optimization
            # len_q_values_before = len(q_values)
            # len_q_stds_before = len(q_stds)
            
            while not done:
                for key in state_dict:
                    if len(state_dict[key]) == 1:
                        if action_dict[key] is None:
                            if args.use_state_norm:
                                state_input = sac_trainer.state_norm(np.array(state_dict[key][0]))
                            else:
                                state_input = np.array(state_dict[key][0])
                            a = sac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)
                            action_dict[key] = a

                            if key == 2 and debug:
                                print('From Algorithm, when no state, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', a, ', reward: ', reward_dict[key])
                                print()

                    elif len(state_dict[key]) == 2:

                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            # print(state_dict[key][0], action_dict[key], reward_dict[key], state_dict[key][1], prob_dict[key], v_dict[key], done)

                            if args.use_state_norm:
                                state = sac_trainer.state_norm(np.array(state_dict[key][0]))
                                next_state = sac_trainer.state_norm(np.array(state_dict[key][1]))
                            else:
                                state = np.array(state_dict[key][0])
                                next_state = np.array(state_dict[key][1])
                            if args.use_reward_scaling:
                                reward = sac_trainer.reward_scaling(reward_dict[key])
                            else:
                                reward = reward_dict[key]

                            replay_buffer.push(state, action_dict[key], reward, next_state, done)
                            if key == 2 and debug:
                                print('From Algorithm store, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', action_dict[key], ', reward: ', reward_dict[key],
                                      'value is: ', v_dict[key])
                                print()

                            episode_steps += 1
                            step += 1
                            episode_reward += reward_dict[key]
                            # if reward_dict[key] == 1.0:
                            #     print('Bus id: ',key,' , station id is: ' , state_dict[key][1][1],' ,current time is: ', env.current_time)
                        state_dict[key] = state_dict[key][1:]
                        if args.use_state_norm:
                            state_input = sac_trainer.state_norm(np.array(state_dict[key][0]))
                        else:
                            state_input = np.array(state_dict[key][0])

                        action_dict[key] = sac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)
                        # print(action_dict[key])
                        # print info like before
                        if key == 2 and debug:
                            print('From Algorithm run, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', action_dict[key], ', reward: ', reward_dict[key], ' ,value is: ',
                                  v_dict[key])
                            print()

                state_dict, reward_dict, done = env.step(action_dict, debug=debug, render=render)
                if len(replay_buffer) > args.batch_size and len(replay_buffer) % args.training_freq == 0 and step_trained != step:
                    step_trained = step
                    for i in range(update_itr):
                        _ = sac_trainer.update(args.batch_size, training_steps, reward_scale=10., auto_entropy=args.auto_entropy, target_entropy=-1. * action_dim)
                        training_steps += 1

                if done:
                    replay_buffer.last_episode_step = episode_steps
                    break
            
            # --- LOGGING AGGREGATION ---
            # --- LOGGING AGGREGATION ---
            def safe_mean(arr):
                if len(arr) == 0:
                    return 0.0
                return np.mean(arr)

            rewards.append(episode_reward)
            # Use tracked lengths to slice exactly what was added this episode
            # MEMORY FIX: We now clear lists at end of episode, so we just take the whole list
            num_added_q = len(q_values)
            num_added_stds = len(q_stds)
            
            if num_added_q > 0:
                q_values_episode.append(safe_mean(q_values))
                reg_norms1_episode.append(safe_mean(reg_norms1))
                reg_norms2_episode.append(safe_mean(reg_norms2))
                log_probs_episode.append(safe_mean(log_probs))
                alpha_values_episode.append(safe_mean(alpha_values))
                ood_losses_episode.append(safe_mean(ood_losses))
            else:
                q_values_episode.append(0)
                reg_norms1_episode.append(0)
                reg_norms2_episode.append(0)
                log_probs_episode.append(0)
                alpha_values_episode.append(0)
                ood_losses_episode.append(0)
                
            if num_added_stds > 0:
                q_stds_episode.append(safe_mean(q_stds))
            else:
                q_stds_episode.append(0)
            
            # --- CRITICAL MEMORY FIX: Clear lists after aggregation ---
            q_values.clear()
            reg_norms1.clear()
            reg_norms2.clear()
            log_probs.clear()
            alpha_values.clear()
            ood_losses.clear()
            q_stds.clear()
            # ---------------------------

            if eps % args.plot_freq == 0:  # plot and model saving interval
                plot(rewards)
                
                # --- SAVE LOGS ---
                np.save(os.path.join(logs_path, 'rewards.npy'), rewards)
                np.save(os.path.join(logs_path, 'q_values_episode.npy'), q_values_episode)
                np.save(os.path.join(logs_path, 'reg_norms1_episode.npy'), reg_norms1_episode)
                np.save(os.path.join(logs_path, 'reg_norms2_episode.npy'), reg_norms2_episode)
                np.save(os.path.join(logs_path, 'log_probs_episode.npy'), log_probs_episode)
                np.save(os.path.join(logs_path, 'alpha_values_episode.npy'), alpha_values_episode)
                np.save(os.path.join(logs_path, 'ood_losses_episode.npy'), ood_losses_episode)
                np.save(os.path.join(logs_path, 'q_stds_episode.npy'), q_stds_episode)
                # -----------------
                
                sac_trainer.save_model(os.path.join(model_path, f"checkpoint_episode_{eps}"))
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()  # Force garbage collection to clean up potential fragmentation
                # snapshot = tracemalloc.take_snapshot()
                # for stat in snapshot.statistics('lineno')[:10]:
                #     print(stat)  # 显示内存占用最大的10行
            replay_buffer_usage = len(replay_buffer) / args.replay_buffer_size * 100
            episode_duration = time.time() - episode_start_time
            
            print(
                f"Episode: {eps} | Episode Reward: {episode_reward} | Duration: {episode_duration:.2f}s "
                f"| CPU Memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB | "
                f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | "
                f"Replay Buffer Usage: {replay_buffer_usage:.2f}%")
        sac_trainer.save_model(os.path.join(model_path, "final"))

        if args.eval_sigmas:
            cross_sigma_results = []
            for eval_sigma in args.eval_sigmas:
                eval_env = env_bus(path, debug=debug, route_sigma=eval_sigma)
                mean_reward, reward_std = evaluate_policy(
                    sac_trainer,
                    eval_env,
                    num_eval_episodes=10,
                    deterministic=True
                )
                print(
                    f"Cross-evaluation | train_sigma={args.route_sigma} -> eval_sigma={eval_sigma} | "
                    f"mean_reward={mean_reward:.2f}, std={reward_std:.2f}"
                )
                cross_sigma_results.append({
                    "train_sigma": float(args.route_sigma),
                    "eval_sigma": float(eval_sigma),
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "weight_reg": float(args.weight_reg),
                    "max_alpha": float(args.maximum_alpha),
                    "critic_actor_ratio": int(args.critic_actor_ratio),
                    "replay_buffer_size": int(args.replay_buffer_size)
                })

            output_path = os.path.join(model_path, 'cross_sigma_eval.json')
            with open(output_path, 'w') as f:
                json.dump(cross_sigma_results, f, indent=2)

    if args.test:
        sac_trainer.load_model(os.path.join(model_path, "final"))
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
                            a = sac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)
                            action_dict[key] = a
                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            episode_reward += reward_dict[key]

                        state_dict[key] = state_dict[key][1:]

                        state_input = np.array(state_dict[key][0])

                        action_dict[key] = sac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)

                state_dict, reward_dict, done = env.step(action_dict)
                # env.render()
            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
