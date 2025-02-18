'''
Soft Actor-Critic version 1
using state value function: 1 V net, 1 target V net, 2 Q net, 1 policy net
paper: https://arxiv.org/pdf/1801.01290.pdf
'''


import math
import random

import gym
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from normalization import Normalization, RewardScaling, RunningMeanStd

from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
from reacher import Reacher
from env.sim import env_bus
import os
import argparse
import time


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
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")
parser.add_argument("--training_freq", type=int, default=5, help="frequency of training the network")

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

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


def update(batch_size, reward_scale, auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2):

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    # print('sample:', state, action,  reward, done)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value    = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

    reward = reward_scale*(reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std
    # Updating alpha wrt entropy
    # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
    if auto_entropy is True:
        alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
        # print('alpha loss: ',alpha_loss)
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()
        alpha = log_alpha.exp()
    else:
        alpha = 1.
        alpha_loss = 0
    # 计算 reg_norm
    reg_norm, weight_norm, bias_norm = 0, [], []
    for layer in value_net.children():
        if isinstance(layer, nn.Linear):
            weight_norm.append(torch.norm(layer.state_dict()['weight']) ** 2)
            bias_norm.append(torch.norm(layer.state_dict()['bias']) ** 2)
    # for layer in soft_q_net2.children():
    #     if isinstance(layer, nn.Linear):
    #         weight_norm.append(torch.norm(layer.state_dict()['weight']) ** 2)
    #         bias_norm.append(torch.norm(layer.state_dict()['bias']) ** 2)

    reg_norm = torch.sqrt(torch.sum(torch.stack(weight_norm)) + torch.sum(torch.stack(bias_norm[0:-1])))
    # Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value # if done==1, only reward
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())


    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()

# Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - alpha * log_prob # for stochastic training, it equals to expectation over action
    value_loss = value_criterion(predicted_value, target_value_func.detach()) + weight_reg * reg_norm

    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

# Training Policy Function
    ''' implementation 1 '''
    policy_loss = (alpha * log_prob - predicted_new_q_value).mean() - weight_reg * reg_norm
    ''' implementation 2 '''
    # policy_loss = (alpha * log_prob - soft_q_net1(state, new_action)).mean() - weight_reg * reg_norm  # Openai Spinning Up implementation
    ''' implementation 3 '''
    # policy_loss = (alpha * log_prob - (predicted_new_q_value - predicted_value.detach())).mean() - weight_reg * reg_norm # max Advantage instead of Q to prevent the Q-value drifted high

    ''' implementation 4 '''  # version of github/higgsfield
    # log_prob_target=predicted_new_q_value - predicted_value
    # policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean() - weight_reg * reg_norm
    # mean_lambda=1e-3
    # std_lambda=1e-3
    # mean_loss = mean_lambda * mean.pow(2).mean()
    # std_loss = std_lambda * log_std.pow(2).mean()
    # policy_loss += mean_loss + std_loss - weight_reg * reg_norm


    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    # print('value_loss: ', value_loss)
    # print('q loss: ', q_value_loss1, q_value_loss2)
    # print('policy loss: ', policy_loss )


# Soft update the target value net
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(  # copy data value into target parameters
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    return predicted_new_q_value.mean()


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac.png')
    # plt.show()


DETERMINISTIC=False
weight_reg = 0.1

debug = False
render = False
path = os.getcwd() + '/env'
env = env_bus(path, debug=debug)
env.reset()

action_dim = env.action_space.shape[0]
action_range = env.action_space.high[0]

hidden_dim = 64
cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']
cat_code_dict = {
    'bus_id': {i: i for i in range(env.max_agent_num)},  # 最大车辆数，预设值
    'station_id': {i: i for i in range(round(len(env.stations) / 2))},  # station_id，有几个站就有几个类别
    'time_period': {i: i for i in range(env.timetables[-1].launch_time // 3600 + 2)},  # time period,以每小时区分，+2是因为让车运行完
    'direction': {0: 0, 1: 1}  # direction 二分类
}
# 数值特征的数量
num_cat_features = len(cat_cols)
num_cont_features = env.state_dim - num_cat_features  # 包括 forward_headway, backward_headway 和最后一个 feature
# 创建嵌入层
embedding_layer = EmbeddingLayer(cat_code_dict, cat_cols)
# SAC 网络的输入维度
embedding_dim = sum([min(50, len(cat_code_dict[col]) // 2) for col in cat_cols])  # 总嵌入维度
state_dim = embedding_dim + num_cont_features  # 状态维度 = 嵌入维度 + 数值特征维度


value_net = ValueNetwork(state_dim, hidden_dim, embedding_layer, activation=F.relu).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim, embedding_layer, activation=F.relu).to(device)

soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_layer, activation=F.relu).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_layer, activation=F.relu).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_layer, action_range, activation=F.relu).to(device)
log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

print('(Target) Value Network: ', value_net)
print('Soft Q Network (1,2): ', soft_q_net1)
print('Policy Network: ', policy_net)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

value_lr  = 1e-5
soft_q_lr = 1e-5
policy_lr = 1e-5
alpha_lr = 1e-5

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
alpha_optimizer = optim.Adam([log_alpha], lr=alpha_lr)

replay_buffer_size = int(1e6)
replay_buffer = ReplayBuffer(replay_buffer_size)

# 初始化RunningMeanStd
initial_mean = [360., 360., 90.]
initial_std = [165., 133., 45.]

running_ms = RunningMeanStd(shape=(num_cont_features,), init_mean=initial_mean, init_std=initial_std)

state_norm = Normalization(num_categorical=num_cat_features, num_numerical=num_cont_features, running_ms=running_ms)
reward_scaling = RewardScaling(shape=1, gamma=0.99)

# hyper-parameters
step = 0
step_trained = 0
max_episodes = 1000
frame_idx = 0
batch_size = 2048
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY = True
DETERMINISTIC = False
rewards = []
model_path = './model/sac_v2'


if __name__ == '__main__':
    if args.train:
        # training loop
        for eps in range(max_episodes):
            if eps != 0:
                env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=render)

            done = False
            episode_steps = 0
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
                                state_input = state_norm(copy.deepcopy(np.array(state_dict[key][0])))
                            else:
                                state_input = np.array(state_dict[key][0])
                            a = policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)

                            action_dict[key] = a

                            if key == 2 and debug:
                                print('From Algorithm, when no state, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', a, ', reward: ', reward_dict[key])
                                print()

                    elif len(state_dict[key]) == 2:

                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            # print(state_dict[key][0], action_dict[key], reward_dict[key], state_dict[key][1], prob_dict[key], v_dict[key], done)

                            if args.use_state_norm:
                                state = state_norm(copy.deepcopy(np.array(state_dict[key][0])))
                                next_state = state_norm(copy.deepcopy(np.array(state_dict[key][1])))
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
                            state_input = state_norm(copy.deepcopy(np.array(state_dict[key][0])))
                        else:
                            state_input = np.array(state_dict[key][0])

                        action_dict[key]= policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)
                        # print(action_dict[key])
                        # print info like before
                        if key == 2 and debug:
                            print('From Algorithm run, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', action_dict[key], ', reward: ', reward_dict[key], ' ,value is: ',
                                  v_dict[key])
                            print()

                state_dict, reward_dict, done = env.step(action_dict, debug=debug, render=render)
                if len(replay_buffer) > batch_size and len(replay_buffer) % args.training_freq == 0 and step_trained != step:
                    step_trained = step
                    for i in range(update_itr):
                        _ = update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1. * action_dim)

                if done:
                    break

            if eps % 20 == 0 and eps > 0:  # plot and model saving interval
                plot(rewards)
                np.save('rewards', rewards)
                torch.save(policy_net.state_dict(), model_path)
            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            rewards.append(episode_reward)
        torch.save(policy_net.state_dict(), model_path)

    if args.test:
        policy_net.load_state_dict(torch.load(model_path))
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
                            a = policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)
                            action_dict[key] = a
                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            episode_reward += reward_dict[key]

                        state_dict[key] = state_dict[key][1:]

                        state_input = np.array(state_dict[key][0])

                        action_dict[key] = policy_net.get_action(torch.from_numpy(state_input).float(), deterministic=DETERMINISTIC)

                state_dict, reward_dict, done = env.step(action_dict)
                # env.render()
            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
