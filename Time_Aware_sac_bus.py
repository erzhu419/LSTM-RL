'''
Complete Time-Aware SAC for Bus Control
解决公交系统时变特性问题的完整实现
'''

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
import math

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description='Train Time-Aware SAC for bus control.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--use_gradient_clip', type=bool, default=True, help="Gradient clipping")
parser.add_argument("--use_state_norm", type=bool, default=False, help="State normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Reward scaling")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--training_freq", type=int, default=5, help="Training frequency")
parser.add_argument("--plot_freq", type=int, default=1, help="Plotting frequency")
parser.add_argument('--auto_entropy', type=bool, default=True, help='Auto entropy tuning')
parser.add_argument("--maximum_alpha", type=float, default=0.3, help="Max entropy weight")
parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
parser.add_argument("--tau", type=float, default=0.005, help="Soft update rate")
# Time-aware specific parameters
parser.add_argument("--use_time_features", type=bool, default=True, help="Use time-aware features")
parser.add_argument("--time_embed_dim", type=int, default=16, help="Time embedding dimension")
parser.add_argument("--adaptive_noise", type=bool, default=True, help="Adaptive exploration noise")
args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity, last_episode_step=5000):
        self.capacity = capacity
        self.last_episode_step = last_episode_step
        self.buffer = {}
        self.position = 0

    def push(self, state, action, reward, next_state, done, time_info=None):
        self.buffer[self.position] = (state, action, reward, next_state, done, time_info)
        self.position += 1

        if len(self.buffer) > self.capacity:
            keys_to_remove = list(self.buffer.keys())[:self.last_episode_step]
            for key in keys_to_remove:
                del self.buffer[key]

    def sample(self, batch_size):
        batch = random.sample(list(self.buffer.values()), batch_size)
        if len(batch[0]) == 6:  # 包含时间信息
            states, actions, rewards, next_states, dones, time_infos = zip(*batch)
            time_infos = np.stack(time_infos) if time_infos[0] is not None else None
        else:
            states, actions, rewards, next_states, dones = zip(*batch)
            time_infos = None

        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.stack(next_states)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones, time_infos

    def __len__(self):
        return len(self.buffer)


class TimeEncoder(nn.Module):
    """时间编码器，将时间信息转换为高维特征"""

    def __init__(self, time_embed_dim=16):
        super(TimeEncoder, self).__init__()
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(4, time_embed_dim),  # hour, minute, day_of_week, is_peak
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )

    def forward(self, time_features):
        return self.time_mlp(time_features)

    def extract_time_features(self, current_time_seconds):
        """从当前时间(秒)提取时间特征"""
        hours = (current_time_seconds // 3600) % 24
        minutes = (current_time_seconds % 3600) // 60
        day_of_week = 0  # 简化处理

        # 判断是否为高峰期
        is_peak = 1.0 if (7 <= hours <= 9) or (17 <= hours <= 19) else 0.0

        # 归一化时间特征
        hour_norm = hours / 24.0
        minute_norm = minutes / 60.0

        return np.array([hour_norm, minute_norm, day_of_week, is_peak], dtype=np.float32)


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


class TimeAwareQNetwork(nn.Module):
    """时间感知的Q网络"""

    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, time_encoder=None, init_w=3e-3):
        super(TimeAwareQNetwork, self).__init__()

        self.embedding_layer = embedding_layer
        self.time_encoder = time_encoder
        self.use_time = time_encoder is not None

        # 计算输入维度
        input_dim = num_inputs + num_actions
        if self.use_time:
            input_dim += time_encoder.time_embed_dim

        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action, time_features=None):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]

        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)

        # 组合状态、动作和时间特征
        x = torch.cat([state_with_embeddings, action], 1)

        if self.use_time and time_features is not None:
            time_embed = self.time_encoder(time_features)
            x = torch.cat([x, time_embed], dim=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class TimeAwarePolicyNetwork(nn.Module):
    """时间感知的策略网络"""

    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, time_encoder=None,
                 action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(TimeAwarePolicyNetwork, self).__init__()

        self.embedding_layer = embedding_layer
        self.time_encoder = time_encoder
        self.use_time = time_encoder is not None
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 计算输入维度
        input_dim = num_inputs
        if self.use_time:
            input_dim += time_encoder.time_embed_dim

        self.linear1 = nn.Linear(input_dim, hidden_size)
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

    def forward(self, state, time_features=None):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]

        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)

        # 添加时间特征
        if self.use_time and time_features is not None:
            time_embed = self.time_encoder(time_features)
            state_with_embeddings = torch.cat([state_with_embeddings, time_embed], dim=1)

        x = F.relu(self.linear1(state_with_embeddings))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, time_features=None, epsilon=1e-6):
        mean, log_std = self.forward(state, time_features)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.to(device))
        action = self.action_range / 2 * action_0 + self.action_range / 2

        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, time_features=None, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if time_features is not None and isinstance(time_features, np.ndarray):
            time_features = torch.FloatTensor(time_features).unsqueeze(0).to(device)

        mean, log_std = self.forward(state, time_features)
        std = log_std.exp()

        if deterministic:
            action = self.action_range / 2 * torch.tanh(mean) + self.action_range / 2
        else:
            normal = Normal(0, 1)
            z = normal.sample(mean.shape).to(device)
            action = self.action_range / 2 * torch.tanh(mean + std * z) + self.action_range / 2

        return action.detach().cpu().numpy()[0]


class TimeAwareSAC_Trainer:
    def __init__(self, env, replay_buffer, hidden_dim, action_range):
        # Categorical and numerical features
        cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']
        cat_code_dict = {
            'bus_id': {i: i for i in range(env.max_agent_num)},
            'station_id': {i: i for i in range(round(len(env.stations) / 2))},
            'time_period': {i: i for i in range(env.timetables[-1].launch_time // 3600 + 2)},
            'direction': {0: 0, 1: 1}
        }

        self.num_cat_features = len(cat_cols)
        self.num_cont_features = env.state_dim - self.num_cat_features

        # Create embedding layer and time encoder
        embedding_layer = EmbeddingLayer(cat_code_dict, cat_cols)
        self.time_encoder = TimeEncoder(args.time_embed_dim) if args.use_time_features else None

        # Calculate input dimensions
        embedding_dim = sum([min(50, len(cat_code_dict[col]) // 2) for col in cat_cols])
        state_dim = embedding_dim + self.num_cont_features

        self.replay_buffer = replay_buffer

        # Create networks with time awareness
        self.q_net1 = TimeAwareQNetwork(state_dim, action_dim, hidden_dim, embedding_layer, self.time_encoder).to(device)
        self.q_net2 = TimeAwareQNetwork(state_dim, action_dim, hidden_dim, embedding_layer, self.time_encoder).to(device)
        self.target_q_net1 = TimeAwareQNetwork(state_dim, action_dim, hidden_dim, embedding_layer, self.time_encoder).to(device)
        self.target_q_net2 = TimeAwareQNetwork(state_dim, action_dim, hidden_dim, embedding_layer, self.time_encoder).to(device)

        # Policy network
        self.policy_net = TimeAwarePolicyNetwork(state_dim, action_dim, hidden_dim, embedding_layer, self.time_encoder, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        print('Time-Aware Q Networks: ', self.q_net1)
        print('Time-Aware Policy Network: ', self.policy_net)

        # Initialize target networks
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers
        q_lr = 1e-4
        policy_lr = 1e-4
        alpha_lr = 3e-4

        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # 自适应探索噪声
        self.exploration_noise = 0.1
        self.noise_decay = 0.9999

        # Initialize normalization
        initial_mean = [360., 360., 90.]
        initial_std = [165., 133., 45.]
        running_ms = RunningMeanStd(shape=(self.num_cont_features,), init_mean=initial_mean, init_std=initial_std)
        self.state_norm = Normalization(num_categorical=self.num_cat_features, num_numerical=self.num_cont_features, running_ms=running_ms)
        self.reward_scaling = RewardScaling(shape=1, gamma=0.99)

    def get_adaptive_noise(self, current_time):
        """根据时间调整探索噪声"""
        if not args.adaptive_noise:
            return self.exploration_noise

        # 高峰期减少探索，平峰期增加探索
        if self.time_encoder:
            time_features = self.time_encoder.extract_time_features(current_time)
            is_peak = time_features[3]  # is_peak feature
            noise_scale = 0.5 if is_peak > 0.5 else 1.2  # 高峰期噪声减半，平峰期增加20%
            return self.exploration_noise * noise_scale
        return self.exploration_noise

    def update(self, batch_size, training_steps, current_time, reward_scale=10., auto_entropy=True, target_entropy=-1, gamma=0.99, soft_tau=1e-2):
        sample_result = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, time_infos = sample_result

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(device)

        # 处理时间特征
        time_features = None
        next_time_features = None
        if time_infos is not None and args.use_time_features:
            time_features = torch.FloatTensor(time_infos).to(device)
            next_time_features = time_features.clone()  # 使用clone避免in-place操作

        # 标准化奖励
        reward_mean = rewards.mean(dim=0, keepdim=True)
        reward_std = rewards.std(dim=0, keepdim=True) + 1e-6
        rewards_normalized = reward_scale * (rewards - reward_mean) / reward_std

        # 当前Q值
        q1_values = self.q_net1(states, actions, time_features)
        q2_values = self.q_net2(states, actions, time_features)

        # 计算目标Q值
        with torch.no_grad():
            next_actions, next_log_probs, _, _, _ = self.policy_net.evaluate(next_states, next_time_features)
            target_q1 = self.target_q_net1(next_states, next_actions, next_time_features)
            target_q2 = self.target_q_net2(next_states, next_actions, next_time_features)
            target_q_min = torch.min(target_q1, target_q2)
            target_q = rewards_normalized + (1 - dones) * gamma * (target_q_min - self.alpha * next_log_probs)

        # Q网络损失 - 分别计算避免共享计算图
        q1_loss = F.mse_loss(q1_values, target_q.detach())
        q2_loss = F.mse_loss(q2_values, target_q.detach())

        # 更新第一个Q网络
        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), max_norm=1.0)
        self.q_optimizer1.step()

        # 更新第二个Q网络
        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), max_norm=1.0)
        self.q_optimizer2.step()

        # 更新策略网络
        policy_loss = None
        if training_steps % 2 == 0:
            # 重新计算，避免使用之前的计算图
            new_actions, log_probs, _, _, _ = self.policy_net.evaluate(states, time_features)
            q1_new = self.q_net1(states, new_actions, time_features)
            q2_new = self.q_net2(states, new_actions, time_features)
            q_new_min = torch.min(q1_new, q2_new)

            policy_loss = (self.alpha * log_probs - q_new_min).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            if args.use_gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.policy_optimizer.step()

            # 更新温度参数
            if auto_entropy:
                # 重新采样避免梯度冲突
                with torch.no_grad():
                    _, log_probs_detached, _, _, _ = self.policy_net.evaluate(states, time_features)
                alpha_loss = -(self.log_alpha * (log_probs_detached + target_entropy)).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # 安全地更新alpha值
                new_alpha = self.log_alpha.exp().item()
                self.alpha = min(args.maximum_alpha, max(new_alpha, 1e-8))

        # 软更新目标网络
        with torch.no_grad():
            for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.mul_(1.0 - soft_tau).add_(param.data, alpha=soft_tau)
            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.mul_(1.0 - soft_tau).add_(param.data, alpha=soft_tau)

        # 更新探索噪声
        if args.adaptive_noise:
            self.exploration_noise = max(self.exploration_noise * self.noise_decay, 0.01)

        # 返回当前Q值用于监控
        with torch.no_grad():
            current_q = torch.min(q1_values, q2_values).mean().item()

        return current_q

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @alpha.setter
    def alpha(self, value):
        self.log_alpha.data.fill_(math.log(max(value, 1e-8)))

    def save_model(self, path):
        torch.save({
            'q_net1': self.q_net1.state_dict(),
            'q_net2': self.q_net2.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'target_q_net1': self.target_q_net1.state_dict(),
            'target_q_net2': self.target_q_net2.state_dict(),
            'log_alpha': self.log_alpha,
            'time_encoder': self.time_encoder.state_dict() if self.time_encoder else None,
            'exploration_noise': self.exploration_noise
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.q_net1.load_state_dict(checkpoint['q_net1'])
        self.q_net2.load_state_dict(checkpoint['q_net2'])
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_q_net1.load_state_dict(checkpoint['target_q_net1'])
        self.target_q_net2.load_state_dict(checkpoint['target_q_net2'])
        self.log_alpha = checkpoint['log_alpha']
        if self.time_encoder and checkpoint.get('time_encoder'):
            self.time_encoder.load_state_dict(checkpoint['time_encoder'])
        if 'exploration_noise' in checkpoint:
            self.exploration_noise = checkpoint['exploration_noise']

        self.q_net1.eval()
        self.q_net2.eval()
        self.policy_net.eval()


def plot(rewards, q_values_episode, alpha_values_episode, noise_values_episode):
    clear_output(True)
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    plt.plot(rewards, label="Episode Reward", color='blue')
    plt.legend()
    plt.title("Time-Aware SAC Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 2)
    plt.plot(q_values_episode, label="Q-Value", color='green')
    plt.legend()
    plt.title("Q-Value Evolution")
    plt.xlabel("Episode")
    plt.ylabel("Q-Value")

    plt.subplot(2, 2, 3)
    plt.plot(alpha_values_episode, label="Alpha", color='red')
    plt.legend()
    plt.title("Temperature Parameter")
    plt.xlabel("Episode")
    plt.ylabel("Alpha")

    plt.subplot(2, 2, 4)
    plt.plot(noise_values_episode, label="Exploration Noise", color='orange')
    plt.legend()
    plt.title("Exploration Noise Decay")
    plt.xlabel("Episode")
    plt.ylabel("Noise Level")

    plt.tight_layout()

    if not os.path.exists('pic'):
        os.makedirs('pic')

    subdir_name = f'time_aware_sac_time_features_{args.use_time_features}_adaptive_noise_{args.adaptive_noise}'
    subdir_path = os.path.join('pic', subdir_name)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

    plt.savefig(os.path.join(subdir_path, f'time_aware_sac_monitoring.png'), dpi=300)
    plt.close()


# Initialize environment and training components
replay_buffer_size = int(1e6)
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
max_episodes = 50
update_itr = 1
AUTO_ENTROPY = True
DETERMINISTIC = False
hidden_dim = 256

# Monitoring variables
rewards = []
q_values = []
alpha_values = []
noise_values = []
q_values_episode = []
alpha_values_episode = []
noise_values_episode = []

model_path = './model/time_aware_sac'
if not os.path.exists('./model'):
    os.makedirs('./model')

tracemalloc.start()

# Initialize Time-Aware SAC trainer
sac_trainer = TimeAwareSAC_Trainer(env, replay_buffer, hidden_dim=hidden_dim, action_range=action_range)

if __name__ == '__main__':
    if args.train:
        print("=" * 60)
        print("Starting Time-Aware SAC training...")
        print(f"Use time features: {args.use_time_features}")
        print(f"Adaptive noise: {args.adaptive_noise}")
        print(f"Time embedding dimension: {args.time_embed_dim}")
        print(f"Max episodes: {max_episodes}")
        print("=" * 60)

        # Training loop
        for eps in range(max_episodes):
            if eps != 0:
                env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=render)

            done = False
            episode_steps = 0
            training_steps = 0
            action_dict = {key: None for key in list(range(env.max_agent_num))}
            episode_reward = 0

            while not done:
                current_time = env.current_time

                for key in state_dict:
                    time_features = None
                    if args.use_time_features and sac_trainer.time_encoder:
                        time_features = sac_trainer.time_encoder.extract_time_features(current_time)

                    if len(state_dict[key]) == 1:
                        if action_dict[key] is None:
                            if args.use_state_norm:
                                state_input = sac_trainer.state_norm(np.array(state_dict[key][0]))
                            else:
                                state_input = np.array(state_dict[key][0])

                            # 获取自适应噪声
                            adaptive_noise = sac_trainer.get_adaptive_noise(current_time)

                            a = sac_trainer.policy_net.get_action(state_input, time_features, deterministic=DETERMINISTIC)

                            # 添加自适应探索噪声
                            if not DETERMINISTIC:
                                a += np.random.normal(0, adaptive_noise, size=a.shape)
                                a = np.clip(a, 0, action_range)

                            action_dict[key] = a

                            if key == 2 and debug:
                                time_str = f"{int(current_time // 3600):02d}:{int((current_time % 3600) // 60):02d}"
                                print(f'Time-Aware SAC: Bus {key}, Station {state_dict[key][0][1]}, Time {time_str}, Action {a:.3f}, Noise {adaptive_noise:.4f}')

                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:
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

                            # 存储时包含时间特征
                            replay_buffer.push(state, action_dict[key], reward, next_state, done, time_features)

                            episode_steps += 1
                            step += 1
                            episode_reward += reward_dict[key]

                            if key == 2 and debug:
                                time_str = f"{int(current_time // 3600):02d}:{int((current_time % 3600) // 60):02d}"
                                print(f'Time-Aware SAC store: Bus {key}, Station {state_dict[key][0][1]}, Time {time_str}, Action {action_dict[key]:.3f}, Reward {reward:.3f}')

                        state_dict[key] = state_dict[key][1:]
                        if args.use_state_norm:
                            state_input = sac_trainer.state_norm(np.array(state_dict[key][0]))
                        else:
                            state_input = np.array(state_dict[key][0])

                        # 获取自适应噪声
                        adaptive_noise = sac_trainer.get_adaptive_noise(current_time)
                        a = sac_trainer.policy_net.get_action(state_input, time_features, deterministic=DETERMINISTIC)

                        # 添加自适应探索噪声
                        if not DETERMINISTIC:
                            a += np.random.normal(0, adaptive_noise, size=a.shape)
                            a = np.clip(a, 0, action_range)

                        action_dict[key] = a

                        if key == 2 and debug:
                            time_str = f"{int(current_time // 3600):02d}:{int((current_time % 3600) // 60):02d}"
                            print(f'Time-Aware SAC run: Bus {key}, Station {state_dict[key][0][1]}, Time {time_str}, Action {a:.3f}')

                state_dict, reward_dict, done = env.step(action_dict, debug=debug, render=render)

                # Training step
                if len(replay_buffer) > args.batch_size and len(replay_buffer) % args.training_freq == 0 and step_trained != step:
                    step_trained = step
                    for i in range(update_itr):
                        q_value = sac_trainer.update(args.batch_size, training_steps, current_time, reward_scale=10.,
                                                     auto_entropy=args.auto_entropy, target_entropy=-1. * action_dim)
                        q_values.append(q_value)
                        alpha_values.append(sac_trainer.alpha)
                        noise_values.append(sac_trainer.exploration_noise)
                        training_steps += 1

                if done:
                    replay_buffer.last_episode_step = episode_steps
                    break

            # Record episode metrics
            rewards.append(episode_reward)
            if training_steps > 0:
                q_values_episode.append(np.mean(q_values[-min(training_steps, 100):]) if q_values else 0)
                alpha_values_episode.append(np.mean(alpha_values[-min(training_steps, 100):]) if alpha_values else sac_trainer.alpha)
                noise_values_episode.append(np.mean(noise_values[-min(training_steps, 100):]) if noise_values else sac_trainer.exploration_noise)
            else:
                q_values_episode.append(0)
                alpha_values_episode.append(sac_trainer.alpha)
                noise_values_episode.append(sac_trainer.exploration_noise)

            # Plot and save
            if eps % args.plot_freq == 0:
                plot(rewards, q_values_episode, alpha_values_episode, noise_values_episode)
                np.save('rewards_time_aware_sac', rewards)
                sac_trainer.save_model(model_path + '_checkpoint.pth')

            replay_buffer_usage = len(replay_buffer) / replay_buffer_size * 100
            current_noise = sac_trainer.get_adaptive_noise(env.current_time)

            # 计算时间相关统计
            current_hour = (env.current_time // 3600) % 24
            is_peak = (7 <= current_hour <= 9) or (17 <= current_hour <= 19)

            print(
                f"Episode: {eps:3d} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Steps: {episode_steps:4d} | "
                f"Q-Val: {q_values_episode[-1]:7.3f} | "
                f"Alpha: {alpha_values_episode[-1]:6.4f} | "
                f"Noise: {current_noise:6.4f} | "
                f"Hour: {int(current_hour):2d} {'(Peak)' if is_peak else '     '} | "
                f"Buffer: {replay_buffer_usage:5.1f}% | "
                f"GPU: {torch.cuda.memory_allocated() / 1024 ** 2:6.1f}MB"
            )

            # 每10个episode打印更详细的统计
            if eps % 10 == 0 and eps > 0:
                recent_rewards = rewards[-10:]
                print(f"  Last 10 episodes: Mean={np.mean(recent_rewards):.2f}, "
                      f"Std={np.std(recent_rewards):.2f}, "
                      f"Min={np.min(recent_rewards):.2f}, "
                      f"Max={np.max(recent_rewards):.2f}")

        # Final save
        sac_trainer.save_model(model_path + '_final.pth')

        # Save training data
        np.savez(model_path + '_training_data.npz',
                 rewards=rewards,
                 q_values=q_values_episode,
                 alpha_values=alpha_values_episode,
                 noise_values=noise_values_episode)

        print("=" * 60)
        print("Time-Aware SAC training completed successfully!")
        print(f"Final episode reward: {rewards[-1]:.2f}")
        print(f"Best episode reward: {max(rewards):.2f}")
        print(f"Average last 10 episodes: {np.mean(rewards[-10:]):.2f}")
        print("=" * 60)

    if args.test:
        print("=" * 60)
        print("Loading Time-Aware SAC model for testing...")

        try:
            sac_trainer.load_model(model_path + '_final.pth')
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Final model not found, trying checkpoint...")
            sac_trainer.load_model(model_path + '_checkpoint.pth')
            print("Checkpoint model loaded!")

        test_rewards = []

        for eps in range(10):
            done = False
            env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=render)
            episode_reward = 0
            action_dict = {key: None for key in list(range(env.max_agent_num))}
            episode_steps = 0

            while not done:
                current_time = env.current_time

                for key in state_dict:
                    time_features = None
                    if args.use_time_features and sac_trainer.time_encoder:
                        time_features = sac_trainer.time_encoder.extract_time_features(current_time)

                    if len(state_dict[key]) == 1:
                        if action_dict[key] is None:
                            state_input = np.array(state_dict[key][0])
                            a = sac_trainer.policy_net.get_action(state_input, time_features, deterministic=True)
                            action_dict[key] = a

                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            episode_reward += reward_dict[key]
                            episode_steps += 1

                        state_dict[key] = state_dict[key][1:]
                        state_input = np.array(state_dict[key][0])
                        action_dict[key] = sac_trainer.policy_net.get_action(state_input, time_features, deterministic=True)

                state_dict, reward_dict, done = env.step(action_dict)

            test_rewards.append(episode_reward)
            current_hour = (env.current_time // 3600) % 24
            time_str = f"{int(current_hour):02d}:{int((env.current_time % 3600) // 60):02d}"

            print(f'Test Episode {eps:2d}: Reward={episode_reward:8.2f}, Steps={episode_steps:4d}, End Time={time_str}')

        print("=" * 60)
        print("Testing completed!")
        print(f"Test episodes mean reward: {np.mean(test_rewards):.2f}")
        print(f"Test episodes std reward: {np.std(test_rewards):.2f}")
        print(f"Test episodes best reward: {max(test_rewards):.2f}")
        print("=" * 60)