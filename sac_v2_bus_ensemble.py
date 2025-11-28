'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''

import psutil, tracemalloc
import torch, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
import json
import argparse
import numpy as np
import random
from copy import deepcopy

from normalization import Normalization, RewardScaling, RunningMeanStd
from bus_feature_utils import create_embedding_layer, build_bus_categorical_info
from bus_replay_buffer import ReplayBuffer

from IPython.display import clear_output
import matplotlib.pyplot as plt
from env.sim import env_bus
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
parser.add_argument("--plot_freq", type=int, default=5, help="frequency of plotting the result")
parser.add_argument('--weight_reg', type=float, default=0.03, help='weight of regularization')
parser.add_argument('--auto_entropy', type=bool, default=True, help='automatically updating alpha')
parser.add_argument("--maximum_alpha", type=float, default=0.3, help="max entropy weight")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--ensemble_size", type=int, default=10, help="Number of critics in the ensemble")
parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension size for networks")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for actor, critic, and alpha optimizers")
#TODO 可以看到这里把beta相关的三个参数降低之后，收敛性好很多，继续调参
parser.add_argument("--beta_bc", type=float, default=0.001, help="weight of behavior cloning loss")
# beta这个参数在源代码中是负数(我开始也奇怪为什么下面代码关于ood_std是+,原来是因为这里是负数)
parser.add_argument("--beta", type=float, default=-2, help="weight of variance")
parser.add_argument("--beta_ood", type=float, default=0.01, help="weight of OOD loss")
parser.add_argument('--critic_actor_ratio', type=int, default=2, help="ratio of critic and actor training")
parser.add_argument('--replay_buffer_size', type=int, default=int(1e6), help="buffer size")
parser.add_argument('--save_root', type=str, default='.', help='Base directory for saving models, logs, and figures')
parser.add_argument('--run_name', type=str, default='gpt_version', help='Optional identifier appended to save directories')
parser.add_argument('--env_path', type=str, default='env', help='Path to the environment configuration directory')
parser.add_argument('--embedding_mode', type=str, default='full', choices=['full', 'one_hot', 'none'], help='Categorical feature handling strategy')
parser.add_argument('--route_sigma', type=float, default=1.5, help='Sigma used for route speed sampling')
parser.add_argument('--eval_sigmas', type=float, nargs='*', default=None, help='List of sigma values for cross-evaluation after training')
args = parser.parse_args()
args.embedding_mode = args.embedding_mode.lower()

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RUN_NAME = args.run_name.strip() if args.run_name else None
SAVE_ROOT = os.path.abspath(args.save_root)

sigma_token = f"sigma{args.route_sigma}".replace('.', 'p')
weight_token = f"wreg{str(args.weight_reg).replace('.', 'p')}"
experiment_components = [SCRIPT_NAME, sigma_token, f"embed-{args.embedding_mode}", weight_token]
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

MODEL_PREFIX = os.path.join(MODEL_DIR, 'sac_v2_bus_ensemble')


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
    def __init__(self, state_dim, action_dim, hidden_dim, embedding_layer, ensemble_size=5):
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
    def __init__(self, env, replay_buffer, hidden_dim, action_range, embedding_mode='full', ensemble_size=5):
        cat_cols, cat_code_dict = build_bus_categorical_info(env)
        self.num_cat_features = len(cat_cols)
        self.num_cont_features = env.state_dim - self.num_cat_features
        self.embedding_mode = embedding_mode
        embedding_kwargs = {'layer_norm': True, 'dropout': 0.05} if embedding_mode == 'full' else {}
        embedding_template = create_embedding_layer(embedding_mode, cat_code_dict, cat_cols, **embedding_kwargs)
        state_dim = embedding_template.output_dim + self.num_cont_features

        self.replay_buffer = replay_buffer
        self.ensemble_size = ensemble_size

        self.soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone(), ensemble_size=ensemble_size).to(device)
        self.target_soft_q_net = deepcopy(self.soft_q_net).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone(), action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        self.alpha = args.maximum_alpha
        print('Soft Q Network: ', self.soft_q_net)
        print('Policy Network: ', self.policy_net)

        self.soft_q_criterion = nn.MSELoss()

        soft_q_lr = policy_lr = alpha_lr = args.lr

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
        batch_size = reward.size(0)
        reg_norm = reg_norm.unsqueeze(-1).repeat(1, batch_size)  # Adjust shape to match target_q_next
        target_q_next = target_q_next - self.alpha * next_log_prob - args.weight_reg * reg_norm  # shape: [ensemble_size, batch, 1]
        target_q_value = reward + (1 - done) * gamma * target_q_next.unsqueeze(-1)

        ood_loss = predicted_q_value.std(0).mean()
        q_value_loss = self.soft_q_criterion(predicted_q_value, target_q_value.squeeze(-1).detach())
        loss = q_value_loss + args.beta_ood * ood_loss
        return loss, predicted_q_value, ood_loss

    # Policy loss computation
    def compute_policy_loss(self, state, action, new_action, log_prob, reg_norm):

        batch_size = action.size(0)
        reg_norm = reg_norm.unsqueeze(-1).repeat(1, batch_size)  # Adjust shape to match target_q_next

        q_values_dist = self.soft_q_net(state, new_action) - args.weight_reg * reg_norm - self.alpha * log_prob

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
        global q_values, reg_norms, log_probs, alpha_values, ood_losses, q_stds

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)
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

        q_value_loss, predicted_q_value, ood_loss = self.compute_q_loss(
            state, action, reward, next_state, done, new_next_action, next_log_prob, reg_norm, gamma
        )
        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward(retain_graph=False)
        if args.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.soft_q_net.parameters(), max_norm=1.0)
        self.soft_q_optimizer.step()

        q_std_value = None
        if training_steps % args.critic_actor_ratio == 0:
            policy_loss, predicted_new_q_value, q_std = self.compute_policy_loss(
                state, action, new_action, log_prob, reg_norm
            )
            q_std_value = q_std.mean().item()

            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=False)
            self.policy_optimizer.step()

        for target_param, param in zip(self.target_soft_q_net.parameters(), self.soft_q_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

        ensemble_means = predicted_q_value.mean(1).detach().cpu().numpy()
        for idx, value in enumerate(ensemble_means):
            q_values[idx].append(float(value))
        reg_norms.append(args.weight_reg * reg_norm.mean().item())
        log_probs.append(-log_prob.mean().item())
        alpha_values.append(self.alpha)
        ood_losses.append(ood_loss.item())
        if q_std_value is not None:
            q_stds.append(q_std_value)

        return float(ensemble_means.mean())

    def save_model(self, path):
        torch.save(self.soft_q_net.state_dict(), path + '_q')
        torch.save(self.policy_net.state_dict(), path + '_policy')

    def load_model(self, path):
        self.soft_q_net.load_state_dict(torch.load(path + '_q', weights_only=True))
        self.policy_net.load_state_dict(torch.load(path + '_policy', weights_only=True))

        self.soft_q_net.eval()
        self.policy_net.eval()


def evaluate_policy(sac_trainer, env, num_eval_episodes=5, deterministic=True):
    eval_rewards = []

    for eval_ep in range(num_eval_episodes):
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
                            state_input = sac_trainer.state_norm(raw_state, update=False)
                        else:
                            state_input = raw_state
                        a = sac_trainer.policy_net.get_action(
                            torch.from_numpy(state_input).float(), deterministic=deterministic
                        )
                        action_dict[key] = a

                elif len(state_dict[key]) == 2:
                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        episode_reward += reward_dict[key]

                    state_dict[key] = state_dict[key][1:]
                    raw_state = np.array(state_dict[key][0])
                    if args.use_state_norm:
                        state_input = sac_trainer.state_norm(raw_state, update=False)
                    else:
                        state_input = raw_state
                    action_dict[key] = sac_trainer.policy_net.get_action(
                        torch.from_numpy(state_input).float(), deterministic=deterministic
                    )

            state_dict, reward_dict, done = env.step(action_dict, render=False)

        eval_rewards.append(episode_reward)

    mean_reward = np.mean(eval_rewards)
    reward_std = np.std(eval_rewards)

    return mean_reward, reward_std


def plot(rewards):
    pass

replay_buffer = ReplayBuffer(args.replay_buffer_size)

debug = False
render = False
path = os.path.abspath(args.env_path)
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

rewards = []  # 记录奖励
q_values = []  # Will be initialised after trainer creation
reg_norms = []  # 记录正则化项1
log_probs = []  # 记录 log_prob
alpha_values = []  # 记录 alpha 值
ood_losses = []
q_stds = []  # 记录 Q 值的标准差

q_values_episode = []  # 记录每个 episode 的 Q 值
reg_norms_episode = []  # 记录每个 episode 的正则化项1
log_probs_episode = []  # 记录每个 episode 的 log_prob
alpha_values_episode = []  # 记录每个 episode 的 alpha 值
ood_losses_episode = []
q_stds_episode = []  # 记录每个 episode 的 Q 值的标准差

eval_episodes = []
eval_mean_rewards = []
eval_reward_stds = []

tracemalloc.start()

sac_trainer = SAC_Trainer(
    env,
    replay_buffer,
    hidden_dim=hidden_dim,
    action_range=action_range,
    embedding_mode=args.embedding_mode,
    ensemble_size=args.ensemble_size,
)

ensemble_size = sac_trainer.soft_q_net.num_critics
q_values = [[] for _ in range(ensemble_size)]

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
            rewards.append(episode_reward)

            if training_steps > 0:
                ensemble_episode_means = []
                for idx in range(len(q_values)):
                    history = q_values[idx][-training_steps:] if training_steps <= len(q_values[idx]) else q_values[idx]
                    if history:
                        ensemble_episode_means.append(float(np.mean(history)))
                    else:
                        ensemble_episode_means.append(float('nan'))
                q_values_episode.append(np.array(ensemble_episode_means, dtype=np.float32))

                reg_norms_episode.append(float(np.mean(reg_norms[-training_steps:])) if reg_norms else 0.0)
                log_probs_episode.append(float(np.mean(log_probs[-training_steps:])) if log_probs else 0.0)
                alpha_values_episode.append(float(np.mean(alpha_values[-training_steps:])) if alpha_values else 0.0)
                ood_losses_episode.append(float(np.mean(ood_losses[-training_steps:])) if ood_losses else 0.0)
                if q_stds:
                    q_stds_episode.append(float(np.mean(q_stds[-training_steps:])))
                else:
                    q_stds_episode.append(None)
            else:
                q_values_episode.append(np.zeros(len(q_values), dtype=np.float32))
                reg_norms_episode.append(0.0)
                log_probs_episode.append(0.0)
                alpha_values_episode.append(0.0)
                ood_losses_episode.append(0.0)
                q_stds_episode.append(None)

            if eps % args.plot_freq == 0:  # plot and model saving interval
                plot(rewards)

                np.save(os.path.join(LOG_DIR, 'rewards.npy'), np.array(rewards, dtype=np.float32))
                if q_values_episode:
                    np.save(
                        os.path.join(LOG_DIR, 'q_values_episode.npy'),
                        np.stack(q_values_episode, axis=1)
                    )
                np.save(os.path.join(LOG_DIR, 'reg_norms_episode.npy'), np.array(reg_norms_episode, dtype=np.float32))
                np.save(os.path.join(LOG_DIR, 'log_probs_episode.npy'), np.array(log_probs_episode, dtype=np.float32))
                np.save(os.path.join(LOG_DIR, 'alpha_values_episode.npy'), np.array(alpha_values_episode, dtype=np.float32))
                np.save(os.path.join(LOG_DIR, 'ood_losses_episode.npy'), np.array(ood_losses_episode, dtype=np.float32))
                np.save(
                    os.path.join(LOG_DIR, 'q_stds_episode.npy'),
                    np.array([np.nan if v is None else v for v in q_stds_episode], dtype=np.float32)
                )

                mean_reward, reward_std = evaluate_policy(sac_trainer, env, num_eval_episodes=10, deterministic=True)
                eval_episodes.append(eps)
                eval_mean_rewards.append(mean_reward)
                eval_reward_stds.append(reward_std)
                np.save(os.path.join(LOG_DIR, 'eval_episodes.npy'), np.array(eval_episodes, dtype=np.int32))
                np.save(os.path.join(LOG_DIR, 'eval_mean_rewards.npy'), np.array(eval_mean_rewards, dtype=np.float32))
                np.save(os.path.join(LOG_DIR, 'eval_reward_stds.npy'), np.array(eval_reward_stds, dtype=np.float32))

                model_name = f"{MODEL_PREFIX}_episode_{eps}"
                sac_trainer.save_model(model_name)
                sac_trainer.save_model(os.path.join(LOG_DIR, f'{SCRIPT_NAME}_episode_{eps}'))
                # snapshot = tracemalloc.take_snapshot()
                # for stat in snapshot.statistics('lineno')[:10]:
                #     print(stat)  # 显示内存占用最大的10行
            replay_buffer_usage = len(replay_buffer) / args.replay_buffer_size * 100

            print(
                f"[SAC-ENSEMBLE | wreg={args.weight_reg}, max_alpha={args.maximum_alpha}, ensemble_size={args.ensemble_size}] Episode: {eps} | Episode Reward: {episode_reward} | CPU Memory: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB | GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | Replay Buffer Usage: {replay_buffer_usage:.2f}%")
        sac_trainer.save_model(MODEL_PREFIX)

        # Ensure final metrics and diagnostics are saved
        plot(rewards)
        np.save(os.path.join(LOG_DIR, 'rewards.npy'), np.array(rewards, dtype=np.float32))
        if q_values_episode:
            np.save(
                os.path.join(LOG_DIR, 'q_values_episode.npy'),
                np.stack(q_values_episode, axis=1)
            )
        np.save(os.path.join(LOG_DIR, 'reg_norms_episode.npy'), np.array(reg_norms_episode, dtype=np.float32))
        np.save(os.path.join(LOG_DIR, 'log_probs_episode.npy'), np.array(log_probs_episode, dtype=np.float32))
        np.save(os.path.join(LOG_DIR, 'alpha_values_episode.npy'), np.array(alpha_values_episode, dtype=np.float32))
        np.save(os.path.join(LOG_DIR, 'ood_losses_episode.npy'), np.array(ood_losses_episode, dtype=np.float32))
        np.save(
            os.path.join(LOG_DIR, 'q_stds_episode.npy'),
            np.array([np.nan if v is None else v for v in q_stds_episode], dtype=np.float32)
        )

        mean_reward, reward_std = evaluate_policy(sac_trainer, env, num_eval_episodes=15, deterministic=True)
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
                sigma_mean, sigma_std = evaluate_policy(sac_trainer, eval_env, num_eval_episodes=10, deterministic=True)
                sigma_results.append((sigma, sigma_mean, sigma_std))
            np.save(os.path.join(LOG_DIR, 'eval_cross_sigma.npy'), np.array(sigma_results, dtype=np.float32))

    if args.test:
        sac_trainer.load_model(MODEL_PREFIX)
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
