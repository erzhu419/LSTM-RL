###
# Similar as ppo_gae_continous.py, but change the update function
# to follow the stablebaseline PPO2 (https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/ppo2/ppo2.html#PPO2) and cleanrl (https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
# it track value of state during sample collection and thus save computation.
###

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from env.sim import env_bus
import os

from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


# Hyperparameters
learning_rate = 1e-4
gamma = 0.99
lmbda = 0.95
eps_clip = 0.1
batch_size = 1280
K_epoch = 10
T_horizon = 10000

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

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


class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1.):
        super(PPO, self).__init__()
        self.data = []
        self.action_range = action_range

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear6 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_param = nn.Parameter(torch.zeros(num_actions, requires_grad=True))

        self.v_linear = nn.Linear(hidden_size, 1)

        orthogonal_init(self.linear1)
        orthogonal_init(self.linear2)
        orthogonal_init(self.linear3)
        orthogonal_init(self.linear4)
        orthogonal_init(self.linear5)
        orthogonal_init(self.linear6)
        orthogonal_init(self.mean_linear, gain=0.01)
        orthogonal_init(self.log_std_linear, gain=0.01)
        orthogonal_init(self.v_linear)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    def pi(self, x):

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x1 = F.relu(self.linear3(x)) 
        x2 = F.relu(self.linear4(x).detach()) # std learning not BP to the feature

        mean = F.tanh(self.mean_linear(x1))
        # log_std = self.log_std_linear(x2)
        log_std = torch.clamp(self.log_std_linear(x2), min=-10, max=10)

        # log_std = self.log_std_param.expand_as(mean)

        return mean, log_std

    def v(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))

        v = self.v_linear(x)
        return v

    def get_action(self, x):
        mean, log_std = self.pi(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(-1)
        prob = log_prob.exp()

        ## The following way of generating action seems not correct.
        ## All dimensions of action depends on the same hidden variable z.
        ## In some envs like Ant-v2, it may let the agent not fall easity due to the correlation of actions.
        ## But this does not in general holds true, and may cause numerical problem (nan) in update.
        # normal = Normal(0, 1)
        # z      = normal.sample()
        # action = mean + std*z
        # log_prob = Normal(mean, std).log_prob(action)
        # log_prob = log_prob.sum(dim=-1, keepdim=True)  # reduce dim
        # prob = log_prob.exp()

        action = self.action_range * action /2  # scale the action
        action += self.action_range
        action = torch.clamp(action, 0, self.action_range)

        value = self.v(x).detach().numpy()
        return action.detach().numpy(), prob, value

    def get_log_prob(self, mean, log_std, action):
        # 逆映射 action 回原始范围
        action = (action - self.action_range) * 2 / self.action_range

        # 使用 Normal 分布
        normal = Normal(mean, log_std.exp())

        # 处理被裁剪的动作
        log_prob = normal.log_prob(action)
        # 裁剪范围之外的动作概率
        log_prob = torch.where(
            (action < -1) | (action > 1),
            torch.full_like(log_prob, -float('inf')),  # 溢出概率设置为负无穷
            log_prob
        )
        return log_prob.sum(dim=-1, keepdim=True)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, value_lst, done_lst = [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, v, done = transition

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            value_lst.append([v])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.tensor(s_lst, dtype=torch.float)
        a = torch.tensor(np.array(a_lst))
        r = torch.tensor(r_lst, dtype=torch.float)
        s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
        v = torch.tensor(np.array(value_lst))
        done_mask = torch.tensor(done_lst, dtype=torch.float)
        prob_a = torch.tensor(prob_a_lst)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, v

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, v = self.make_batch()
        done_mask_ = torch.flip(done_mask, dims=(0,))
        with torch.no_grad():
            advantage = torch.zeros_like(r)
            lastgaelam = 0
            for t in reversed(range(s.shape[0] - 1)):
                if done_mask[t + 1]:
                    nextvalues = self.v(s[t + 1])
                else:
                    nextvalues = v[t + 1]
                delta = r[t] + gamma * nextvalues * done_mask_[t + 1] - v[t]
                advantage[t] = lastgaelam = delta + gamma * lmbda * lastgaelam * done_mask_[t + 1]

            if not np.isnan(advantage.std()):
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            td_target = advantage + self.v(s)

        for i in range(K_epoch):
            mean, log_std = self.pi(s)
            log_pi_a = self.get_log_prob(mean, log_std, a)
            # pi = self.pi(s, softmax_dim=1)
            # pi_a = pi.gather(1,a)
            ratio = torch.exp(log_pi_a - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

            total_rewards = -torch.min(surr1, surr2)
            v_loss = F.smooth_l1_loss(self.v(s), td_target.detach())

            loss = total_rewards + v_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        return total_rewards.mean().detach().numpy(), v_loss.mean().detach().numpy()

def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(10,5))
    plt.plot(rewards)
    plt.savefig('ppo_gae_continous2_bus.png')
    # plt.show()
    plt.clf()
    plt.close()
    
def main():
    debug = False
    render = False
    # env = gym.make('HalfCheetah-v2')
    path = os.getcwd() + '/env'
    env = env_bus(path, debug=debug)

    state_dim = env.state_dim
    action_dim = env.action_space.shape[0]
    hidden_dim = 32
    model = PPO(state_dim, action_dim, hidden_dim, action_range=env.action_space.high[0])
    score = 0.0
    print_interval = 4
    step = 0
    step_trained = 0

    for n_epi in range(1,100000):
        env.reset()
        state_dict, reward_dict, _ = env.initialize_state(render=render)    
        done = False
        episode_steps = 0
        action_dict = {key: None for key in list(range(env.max_agent_num))}
        action_dict_zero = {key: 0 for key in list(range(env.max_agent_num))} # 全0的action，用于查看reward的上限
        action_dict_twenty = {key: 20 for key in list(range(env.max_agent_num))} # 全20的action，用于查看reward的上限

        prob_dict = {key: None for key in list(range(env.max_agent_num))}
        v_dict = {key: None for key in list(range(env.max_agent_num))}
        total_rewards, v_loss = 0, 0
        # while not done:
        while not done:

            for key in state_dict:
                if len(state_dict[key]) == 1:
                    if action_dict[key] is None:
                        state_input = np.expand_dims(state_dict[key][0][1:], axis=0)
                        a, prob, v = model.get_action(torch.from_numpy(state_input).float())

                        action_dict[key], prob_dict[key], v_dict[key] = a, prob, v

                        if key == 2 and debug:
                            print('From Algorithm, when no state, Bus id: ',key,' , station id is: ' , state_dict[key][0][1],' ,current time is: ', env.current_time, ' ,action is: ', a, ', reward: ', reward_dict[key], 'value is: ', v)  
                            print()

                elif len(state_dict[key]) == 2:
                    # if state_dict[key][0][1] == state_dict[key][1][1]:
                    #
                    #     state_dict[key] = state_dict[key][1:]
                    #     state_input = np.expand_dims(state_dict[key][0][1:], axis=0)
                    #     action_dict[key], prob_dict[key], v_dict[key] = model.get_action(torch.from_numpy(state_input).float())
                    #
                    #     if key == 2 and debug:
                    #         print('From Algo, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', action_dict[key], ', reward: ', reward_dict[key])
                    #         print()

                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        # print(state_dict[key][0], action_dict[key], reward_dict[key], state_dict[key][1], prob_dict[key], v_dict[key], done)
                        model.put_data((state_dict[key][0][1:], action_dict[key], reward_dict[key], state_dict[key][1][1:], prob_dict[key], v_dict[key], done))
                        if key == 2 and debug:
                            print('From Algorithm store, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', action_dict[key], ', reward: ', reward_dict[key], 'value is: ', v_dict[key])
                            print()

                        episode_steps += 1
                        step += 1
                        score += reward_dict[key]
                        # if reward_dict[key] == 1.0:
                        #     print('Bus id: ',key,' , station id is: ' , state_dict[key][1][1],' ,current time is: ', env.current_time)
                    state_dict[key] = state_dict[key][1:]

                    state_input = np.expand_dims(state_dict[key][0][1:], axis=0)

                    action_dict[key],prob_dict[key], v_dict[key]  = model.get_action(torch.from_numpy(state_input).float())
                    # print info like before
                    if key == 2 and debug:
                        print('From Algorithm run, Bus id: ', key, ' , station id is: ', state_dict[key][0][1], ' ,current time is: ', env.current_time, ' ,action is: ', action_dict[key], ', reward: ', reward_dict[key], ' ,value is: ', v_dict[key])
                        print()

            state_dict, reward_dict, done = env.step(action_dict, debug=debug, render=render)

            if (step + 1) % batch_size == 0 and step_trained != step:
                step_trained = step
                # print("training!")
                total_rewards, v_loss = model.train_net()

            if done:
                break
        if n_epi % print_interval == 0:

            output_dir = os.path.join(env.path, 'pic')
            os.makedirs(output_dir, exist_ok=True)
            # env.visualizer.plot(n_epi)

            plot(score)
            print("# of episode :{}, avg score : {:.1f}, episode steps : {}, total_rewards : {:.4f}, value loss : {:.4f}".format(n_epi, score / print_interval, episode_steps, total_rewards, v_loss))
            score = 0.0

if __name__ == '__main__':
    main()