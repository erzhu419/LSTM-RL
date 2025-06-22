'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''
import random

import gym
import numpy as np
from normalization import Normalization, RewardScaling, RunningMeanStd

import torch
import psutil
torch.multiprocessing.set_start_method('forkserver', force=True)  # critical for make multiprocessing work
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from IPython.display import clear_output
import matplotlib.pyplot as plt
from env.sim import env_bus

import argparse
import os

import torch.multiprocessing as mp
from multiprocessing import Process
from multiprocessing.managers import BaseManager
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--use_gradient_clip', type=bool, default=True, help="Trick 1:gradient clipping")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")
parser.add_argument("--training_freq", type=int, default=5, help="frequency of training the network")
parser.add_argument("--plot_freq", type=int, default=1, help="frequency of plotting the result")
parser.add_argument('--weight_reg', type=float, default=0.1, help='weight of regularization')
parser.add_argument('--auto_entropy', type=bool, default=True, help='automatically updating alpha')
parser.add_argument("--maximum_alpha", type=float, default=0.3, help="max entropy weight")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity, last_episode_step=5000):
        self.capacity = capacity
        self.last_episode_step = last_episode_step  # 预估每个 episode 的 step 数
        self.buffer = {}
        self.position = 0  # 用作 dict 的 key

    def push(self, state, action, reward, next_state, done):
        """添加新数据"""
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position += 1

        # 当 buffer 过大时，删除最早的 episode 数据
        if len(self.buffer) > self.capacity:
            keys_to_remove = list(self.buffer.keys())[:self.last_episode_step]  # 找到最早的 N 条数据
            for key in keys_to_remove:
                del self.buffer[key]  # 直接删除，提高性能

    def sample(self, batch_size):
        """随机采样 batch_size 大小的数据，确保数据格式正确"""
        batch = random.sample(list(self.buffer.values()), batch_size)  # 直接从 dict 的值采样
        states, actions, rewards, next_states, dones = zip(*batch)

        # 确保维度正确，防止 PyTorch 计算时出现广播错误
        states = np.stack(states)                      # (batch_size, state_dim)
        actions = np.stack(actions)                    # (batch_size, action_dim) 或 (batch_size,)
        rewards = np.array(rewards, dtype=np.float32)  # (batch_size,)
        next_states = np.stack(next_states)            # (batch_size, state_dim)
        dones = np.array(dones, dtype=np.float32)      # (batch_size,)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

# # 用列表作为 replaybuffer 的数据结构
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.last_episode_step = 5000  # 估算每个 episode 的 step 数
#         self.buffer = []
#
#     def push(self, state, action, reward, next_state, done):
#         """添加新数据，当 buffer 超出容量时，直接丢弃最早的一批数据"""
#         self.buffer.append((state, action, reward, next_state, done))
#
#         # 当 buffer 过大时，批量删除最早的 episode 数据
#         if len(self.buffer) > self.capacity:
#             self.buffer = self.buffer[self.last_episode_step:]  # 直接删除最早的 `last_episode_step` 条数据
#
#     def sample(self, batch_size):
#         """随机采样 batch_size 大小的数据，确保数据格式正确"""
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#
#         # 确保维度正确，防止 PyTorch 计算时出现广播错误
#         states = np.stack(states)  # (batch_size, state_dim)
#         actions = np.stack(actions)  # (batch_size, action_dim) 或 (batch_size,)
#         rewards = np.array(rewards, dtype=np.float32)  # (batch_size,)
#         next_states = np.stack(next_states)  # (batch_size, state_dim)
#         dones = np.array(dones, dtype=np.float32)  # (batch_size,)
#
#         return states, actions, rewards, next_states, dones
#
#     def __len__(self):
#         return len(self.buffer)

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


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, embedding_layer, activation=F.relu, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.activation = activation

    def forward(self, state):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]  # Assuming first columns are categorical
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]  # The rest are numerical

        # cat_tensor = torch.clamp(cat_tensor, min=0, max=max(self.embedding_layer.cat_code_dict.values()))
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)  # Concatenate embedding and numerical features

        x = self.activation(self.linear1(state_with_embeddings))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, activation=F.relu, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.embedding_layer = embedding_layer
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.activation = activation

    def forward(self, state, action):
        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]  # Assuming first columns are categorical
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]  # The rest are numerical

        # cat_tensor = torch.clamp(cat_tensor, min=0, max=max(self.embedding_layer.cat_code_dict.values()))
        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)  # Concatenate embedding and numerical features
        x = torch.cat([state_with_embeddings, action], 1)

        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, embedding_layer, action_range, activation=F.relu, init_w=3e-3, log_std_min=-20, log_std_max=2):
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
        self.activation = activation


    def forward(self, state):

        cat_tensor = state[:, :len(self.embedding_layer.cat_cols)]
        num_tensor = state[:, len(self.embedding_layer.cat_cols):]

        embedding = self.embedding_layer(cat_tensor.long())
        state_with_embeddings = torch.cat([embedding, num_tensor], dim=1)

        x = self.activation(self.linear1(state_with_embeddings))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))

        mean    = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        deterministic evaluation provides better performance according to the original paper;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow

        normal = Normal(0, 1)
        z      = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range/2 *action_0 + self.action_range/2
        ''' stochastic evaluation '''
        log_prob = Normal(mean, std).log_prob(mean + std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        ''' deterministic evaluation '''
        # log_prob = Normal(mean, std).log_prob(mean) - torch.log(1. - torch.tanh(mean).pow(2) + epsilon) -  np.log(self.action_range)
        '''
         both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
         the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
         needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
         '''
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std


    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(device)
        action = self.action_range/2 * torch.tanh(mean + std*z) + self.action_range/2
        action = self.action_range/2 * torch.tanh(mean).detach().cpu().numpy()[0] + self.action_range/2 if deterministic else action.detach().cpu().numpy()[0]

        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return (self.action_range*a).numpy()


class SAC_Trainer():

    def __init__(self, env, replay_buffer, hidden_dim, action_range):

        self.rewards = []  # 记录奖励
        self.q_values = []  # 记录 Q 值变化
        self.v_values = []  # 记录 V 值变化
        self.reg_norms = []  # 记录正则化项
        self.log_probs = []  # 记录 log_prob
        self.alpha_values = []  # 记录 alpha 值

        self.q_values_episode = []  # 记录每个 episode 的 Q 值
        self.v_values_episode = []  # 记录每个 episode 的 V 值
        self.reg_norms_episode = []  # 记录每个 episode 的正则化项
        self.log_probs_episode = []  # 记录每个 episode 的 log_prob
        self.alpha_values_episode = []  # 记录每个 episode 的 alpha 值

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
        self.embedding_layer = EmbeddingLayer(cat_code_dict, cat_cols)
        # SAC 网络的输入维度
        embedding_dim = sum([min(50, len(cat_code_dict[col]) // 2) for col in cat_cols])  # 总嵌入维度
        state_dim = embedding_dim + self.num_cont_features  # 状态维度 = 嵌入维度 + 数值特征维度
        self.replay_buffer = replay_buffer

        self.value_net = ValueNetwork(state_dim, hidden_dim, self.embedding_layer, activation=F.relu).to(device)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim, self.embedding_layer, activation=F.relu).to(device)
        self.soft_q_net1 = SoftQNetwork(state_dim, env.action_space.shape[0], hidden_dim, self.embedding_layer, activation=F.relu).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, env.action_space.shape[0], hidden_dim, self.embedding_layer, activation=F.relu).to(device)
        self.policy_net = PolicyNetwork(state_dim, env.action_space.shape[0], hidden_dim, self.embedding_layer, action_range, activation=F.relu).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        print('(Target) Value Network: ', self.value_net)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        value_lr = 1e-5
        soft_q_lr = 1e-5
        policy_lr = 1e-5
        alpha_lr = 1e-5

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def update(self, batch_size, training_steps, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        reward = reward_scale * (reward - reward.mean(dim=0)) / reward.std(dim=0)  # normalize with batch mean and std
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=False)
            self.alpha_optimizer.step()
            # 限制 alpha 的范围
            alpha = min(args.maximum_alpha, self.log_alpha.exp().item())
        else:
            alpha = 1.
            alpha_loss = 0
        # 计算 reg_norm
        reg_norm, weight_norm, bias_norm = 0, [], []
        for layer in self.value_net.children():
            if isinstance(layer, nn.Linear):
                weight_norm.append(torch.norm(layer.state_dict()['weight']) ** 2)
                bias_norm.append(torch.norm(layer.state_dict()['bias']) ** 2)
        # for layer in soft_q_net2.children():
        #     if isinstance(layer, nn.Linear):
        #         weight_norm.append(torch.norm(layer.state_dict()['weight']) ** 2)
        #         bias_norm.append(torch.norm(layer.state_dict()['bias']) ** 2)

        reg_norm = torch.sqrt(torch.sum(torch.stack(weight_norm)) + torch.sum(torch.stack(bias_norm[0:-1])))
        # Training Q Function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward(retain_graph=False)
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), max_norm=1.0)  # Q 网络梯度裁剪
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward(retain_graph=False)
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), max_norm=1.0)  # Q 网络梯度裁剪
        self.soft_q_optimizer2.step()

        # Training Value Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        target_value_func = predicted_new_q_value - alpha * log_prob - args.weight_reg * reg_norm  # for stochastic training, it equals to expectation over action
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=False)
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)  # V 网络梯度裁剪
        self.value_optimizer.step()

        if training_steps % 2 == 0:
            #   Training Policy Function
            ''' implementation 1 '''
            policy_loss = (alpha * log_prob - predicted_new_q_value).mean() + args.weight_reg * reg_norm
            ''' implementation 2 '''
            # policy_loss = (alpha * log_prob - soft_q_net1(state, new_action)).mean() - args.weight_reg * reg_norm  # Openai Spinning Up implementation
            ''' implementation 3 '''
            # policy_loss = (alpha * log_prob - (predicted_new_q_value - predicted_value.detach())).mean() - args.weight_reg * reg_norm # max Advantage instead of Q to prevent the Q-value drifted high

            ''' implementation 4 '''  # version of github/higgsfield
            # log_prob_target=predicted_new_q_value - predicted_value
            # policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean() - args.weight_reg * reg_norm
            # mean_lambda=1e-3
            # std_lambda=1e-3
            # mean_loss = mean_lambda * mean.pow(2).mean()
            # std_loss = std_lambda * log_std.pow(2).mean()
            # policy_loss += mean_loss + std_loss - args.weight_reg * reg_norm

            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=False)
            if args.use_gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)  # Policy 网络梯度裁剪
            self.policy_optimizer.step()

        # print('value_loss: ', value_loss)
        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        # 记录 Q 值和 V 值用于绘图
        self.q_values.append(predicted_new_q_value.mean().item())
        self.v_values.append(predicted_value.mean().item())
        self.reg_norms.append(args.weight_reg * reg_norm.item())
        self.log_probs.append(-log_prob.mean().item())
        self.alpha_values.append(alpha)

        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '_q1')
        torch.save(self.soft_q_net2.state_dict(), path + '_q2')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path + '_q2'))
        self.policy_net.load_state_dict(torch.load(path + '_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


def worker(id, sac_trainer, rewards_queue, replay_buffer, max_episodes, update_itr, DETERMINISTIC, model_path, replay_buffer_size):
    '''
    the function for sampling with multi-processing
    '''

    print(sac_trainer, replay_buffer)  # sac_tainer are not the same, but all networks and optimizers in it are the same; replay  buffer is the same one.
    debug = False
    render = False
    path = os.getcwd() + '/env'
    env = env_bus(path, debug=debug)
    env.reset()
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]
    initial_mean = [360., 360., 90.]
    initial_std = [165., 133., 45.]
    step = 0
    step_trained = 0
    running_ms = RunningMeanStd(shape=(sac_trainer.num_cont_features,), init_mean=initial_mean, init_std=initial_std)

    state_norm = Normalization(num_categorical=sac_trainer.num_cat_features, num_numerical=sac_trainer.num_cont_features, running_ms=running_ms)
    reward_scaling = RewardScaling(shape=1, gamma=0.99)
    rewards = []
    # training loop
    for eps in range(max_episodes):
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

        while not done:
            for key in state_dict:
                if len(state_dict[key]) == 1:
                    if action_dict[key] is None:
                        if args.use_state_norm:
                            state_input = state_norm(np.array(state_dict[key][0]))
                        else:
                            state_input = np.array(state_dict[key][0])

                        action_dict[key] = sac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)

                        if key == 2 and debug:
                            print('From Algorithm, when no state, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', action_dict[key], ', reward: ', reward_dict[key])
                            print()

                elif len(state_dict[key]) == 2:

                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        # print(state_dict[key][0], action_dict[key], reward_dict[key], state_dict[key][1], prob_dict[key], v_dict[key], done)

                        if args.use_state_norm:
                            state = state_norm(np.array(state_dict[key][0]))
                            next_state = state_norm(np.array(state_dict[key][1]))
                        else:
                            state = np.array(state_dict[key][0])
                            next_state = np.array(state_dict[key][1])
                        if args.use_reward_scaling:
                            reward = reward_scaling(reward_dict[key])
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
                        state_input = state_norm(np.array(state_dict[key][0]))
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
            # 对某一个worker输出env.step的信息
            # if id == 0:
            #     if env.current_time % 3600 == 0:
            #         print('Env.step, current time is: ', env.current_time)
            if replay_buffer.get_length() > args.batch_size and replay_buffer.get_length() % args.training_freq == 0 and step_trained != step:
                step_trained = step
                for i in range(update_itr):
                    _ = sac_trainer.update(args.batch_size, training_steps, reward_scale=10., auto_entropy=args.auto_entropy, target_entropy=-1. * action_dim)

                    training_steps += 1

            if done: # TODO 这里也许有问题，因为一个线程累计的，可能不代表其他线程累计的
                replay_buffer.last_episode_step = episode_steps
                break
        # 计算每个 episode 的平均 Q 值和 V 值
        rewards.append(episode_reward)
        sac_trainer.q_values_episode.append(np.mean(sac_trainer.q_values[-training_steps:]))
        sac_trainer.v_values_episode.append(np.mean(sac_trainer.v_values[-training_steps:]))
        sac_trainer.reg_norms_episode.append(np.mean(sac_trainer.reg_norms[-training_steps:]))
        sac_trainer.log_probs_episode.append(np.mean(sac_trainer.log_probs[-training_steps:]))
        sac_trainer.alpha_values_episode.append(np.mean(sac_trainer.alpha_values[-training_steps:]))

        if eps % args.plot_freq == 0:  # plot and model saving interval
            plot(rewards)
            np.save('rewards', rewards)
            torch.save(sac_trainer.policy_net.state_dict(), model_path)
            # snapshot = tracemalloc.take_snapshot()
            # for stat in snapshot.statistics('lineno')[:10]:
            #     print(stat)  # 显示内存占用最大的10行
        replay_buffer_usage = replay_buffer.get_length() / replay_buffer_size * 100

        print(
            f"Episode: {eps} | Episode Reward: {episode_reward} | CPU Memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB | GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | Replay Buffer Usage: {replay_buffer_usage:.2f}%")

        rewards_queue.put(episode_reward)

    sac_trainer.save_model(model_path)


def ShareParameters(adamoptim):
    ''' share parameters of Adamoptimizers for multiprocessing '''
    for group in adamoptim.param_groups:
        for p in group['params']:
            state = adamoptim.state[p]
            # initialize: have to initialize here, or else cannot find
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

            # share in memory
            state['exp_avg'].share_memory_()
            state['exp_avg_sq'].share_memory_()


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.plot(rewards)
    plt.savefig('sac_v2_multi.png')
    # plt.show()
    plt.clf()


if __name__ == '__main__':

    replay_buffer_size = 1e8
    # replay_buffer = ReplayBuffer(replay_buffer_size)

    # the replay buffer is a class, have to use torch manager to make it a proxy for sharing across processes
    BaseManager.register('ReplayBuffer', ReplayBuffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(replay_buffer_size)  # share the replay buffer through manager

    # choose env
    debug = False
    path = os.getcwd() + '/env'
    env = env_bus(path, debug=debug)

    # hyper-parameters for RL training, no need for sharing across processes
    max_episodes = 1000
    batch_size = 8192
    update_itr = 1
    DETERMINISTIC = False
    hidden_dim = 64
    model_path = './model/sac_v2_multi'

    sac_trainer = SAC_Trainer(env, replay_buffer, hidden_dim=hidden_dim, action_range=env.action_space.high[0])

    if args.train:

        # share the global parameters in multiprocessing
        sac_trainer.embedding_layer.share_memory()
        sac_trainer.soft_q_net1.share_memory()
        sac_trainer.soft_q_net2.share_memory()
        sac_trainer.value_net.share_memory()
        sac_trainer.target_value_net.share_memory()
        sac_trainer.policy_net.share_memory()  # model
        sac_trainer.log_alpha.share_memory_()  # variable
        ShareParameters(sac_trainer.value_optimizer)
        ShareParameters(sac_trainer.soft_q_optimizer1)
        ShareParameters(sac_trainer.soft_q_optimizer2)
        ShareParameters(sac_trainer.policy_optimizer)
        ShareParameters(sac_trainer.alpha_optimizer)

        rewards_queue = mp.Queue()  # used for get rewards from all processes and plot the curve

        num_workers = 8  # or: mp.cpu_count()
        processes = []
        rewards = []

        for i in range(num_workers):
            process = Process(target=worker, args=(i, sac_trainer, rewards_queue, replay_buffer, max_episodes, update_itr, DETERMINISTIC, model_path, replay_buffer_size))  # the args contain shared and not shared
            process.daemon = True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]
        while True:  # keep getting the episode reward from the queue
            r = rewards_queue.get()
            if r is not None:
                rewards.append(r)
            else:
                break

            if len(rewards) % 20 == 0 and len(rewards) > 0:
                plot(rewards)

        [p.join() for p in processes]  # finished at the same time

        sac_trainer.save_model(model_path)

    if args.test:
        sac_trainer.policy_net.load_state_dict(torch.load(model_path))
        for eps in range(10):
            done = False
            env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=False)
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
