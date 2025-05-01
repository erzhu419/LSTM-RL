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
import torch, math
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
parser.add_argument('--weight_reg', type=float, default=0.03, help='weight of regularization')
parser.add_argument('--auto_entropy', type=bool, default=True, help='automatically updating alpha')
parser.add_argument("--maximum_alpha", type=float, default=0.3, help="max entropy weight")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
#TODO 可以看到这里把beta相关的三个参数降低之后，收敛性好很多，继续调参
parser.add_argument("--beta_bc", type=float, default=0.001, help="weight of behavior cloning loss")
# beta这个参数在源代码中是负数(我开始也奇怪为什么下面代码关于ood_std是+,原来是因为这里是负数)
parser.add_argument("--beta", type=float, default=-2, help="weight of variance")
parser.add_argument("--beta_ood", type=float, default=0.01, help="weight of OOD loss")
parser.add_argument('--critic_actor_ratio', type=int, default=2, help="ratio of critic and actor training")
parser.add_argument('--replay_buffer_size', type=int, default=int(1e6), help="buffer size")
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
        states = np.stack(states)  # (batch_size, state_dim)
        actions = np.stack(actions)  # (batch_size, action_dim) 或 (batch_size,)
        rewards = np.array(rewards, dtype=np.float32)  # (batch_size,)
        next_states = np.stack(next_states)  # (batch_size, state_dim)
        dones = np.array(dones, dtype=np.float32)  # (batch_size,)

        return states, actions, rewards, next_states, dones

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
    def __init__(self, env, replay_buffer, hidden_dim, action_range):
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

        self.soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_layer).to(device)
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
        global q_values
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)
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
        # 把q_value分开花
        if len(q_values) == 0:
            q_values = predicted_q_value.mean(1).cpu().detach().numpy().reshape(-1,1)
        else:
            q_values = np.concatenate((q_values, predicted_q_value.mean(1).cpu().detach().numpy().reshape(-1,1)), axis=1)
        reg_norms.append(args.weight_reg * reg_norm.mean().item())
        log_probs.append(-log_prob.mean().item())
        alpha_values.append(self.alpha)
        ood_losses.append(ood_loss.item())

        return predicted_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net.state_dict(), path + '_q')
        torch.save(self.policy_net.state_dict(), path + '_policy')

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

    for i in range(q_values_episode.shape[0]):
        # 给Q值做了缩放，方便画图
        plt.plot(q_values_episode[i]/50, label=f"Q-Value {i + 1}", color=f"C{i % 10}")
    plt.plot(reg_norms_episode, label="Regularization Term")
    plt.plot(log_probs_episode, label="Log Prob")
    plt.plot(alpha_values_episode, label="Alpha")
    plt.plot(ood_losses_episode, label="OOD Loss")
    plt.plot(q_stds_episode, label="Q Std")

    plt.legend()
    plt.title(f"Q-Value & V-Value and log_prob & regularization Monitoring (weight_reg={args.weight_reg})")

    if not os.path.exists('pic'):
        os.makedirs('pic')
    # Create subdirectory based on parameters except weight_reg
    subdir_name = f"replay_buffer_size_{args.replay_buffer_size}/critic_actor_ratio_{args.critic_actor_ratio}/maximum_alpha_{args.maximum_alpha}/weight_reg_{args.weight_reg}"
    subdir_path = os.path.join('pic', subdir_name)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

    # Save the plot in the subdirectory
    plt.savefig(os.path.join(subdir_path, f'sac_monitoring_weight_reg_{args.weight_reg}.png'))
    plt.close()

replay_buffer = ReplayBuffer(args.replay_buffer_size)

debug = False
render = False
path = os.getcwd() + '/env'
env = env_bus(path, debug=debug)
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
hidden_dim = 64

rewards = []  # 记录奖励
q_values = np.array([], dtype=np.float32)  # 记录 Q 值变化
reg_norms = []  # 记录正则化项1
log_probs = []  # 记录 log_prob
alpha_values = []  # 记录 alpha 值
ood_losses = []
q_stds = []  # 记录 Q 值的标准差

q_values_episode = np.array([],dtype=np.float32)  # 记录每个 episode 的 Q 值
reg_norms_episode = []  # 记录每个 episode 的正则化项1
log_probs_episode = []  # 记录每个 episode 的 log_prob
alpha_values_episode = []  # 记录每个 episode 的 alpha 值
ood_losses_episode = []
q_stds_episode = []  # 记录每个 episode 的 Q 值的标准差

model_path = f"./model/sac_v2_bus_ensemble/replay_buffer_size_{args.replay_buffer_size}/critic_actor_ratio_{args.critic_actor_ratio}/maximum_alpha_{args.maximum_alpha}/weight_reg_{args.weight_reg}"
os.makedirs(model_path, exist_ok=True)
tracemalloc.start()

sac_trainer = SAC_Trainer(env, replay_buffer, hidden_dim=hidden_dim, action_range=action_range)

if __name__ == '__main__':
    if args.train:
        # training loop
        for eps in range(args.max_episodes):
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
            # 计算每个 episode 的平均 Q 值
            rewards.append(episode_reward)
            # q_values_episode.append(np.mean(q_values[:,-training_steps:],axis=1))
            if len(q_values_episode) == 0:
                q_values_episode = np.mean(q_values[:,-training_steps:],axis=1).reshape(-1,1)
            else:
                q_values_episode = np.concatenate((q_values_episode, np.mean(q_values[:,-training_steps:], axis=1).reshape(-1,1)), axis=1)

            reg_norms_episode.append(np.mean(reg_norms[-training_steps:]))
            log_probs_episode.append(np.mean(log_probs[-training_steps:]))
            alpha_values_episode.append(np.mean(alpha_values[-training_steps:]))
            ood_losses_episode.append(np.mean(ood_losses[-training_steps:]))
            q_stds_episode.append(np.mean(q_stds[-training_steps:])) if len(q_stds) > 0 else None

            if eps % args.plot_freq == 0:  # plot and model saving interval
                plot(rewards)
                np.save('rewards', rewards)
                torch.save(sac_trainer.policy_net.state_dict(), model_path + ' ' + str(eps))
                # snapshot = tracemalloc.take_snapshot()
                # for stat in snapshot.statistics('lineno')[:10]:
                #     print(stat)  # 显示内存占用最大的10行
            replay_buffer_usage = len(replay_buffer) / args.replay_buffer_size * 100

            print(
                f"Episode: {eps} | Episode Reward: {episode_reward} | CPU Memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB | GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | Replay Buffer Usage: {replay_buffer_usage:.2f}%")
        torch.save(sac_trainer.policy_net.state_dict(), model_path)

    if args.test:
        sac_trainer.policy_net.load_state_dict(torch.load(model_path))
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