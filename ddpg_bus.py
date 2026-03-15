"""
Deep Deterministic Policy Gradient (DDPG) for the bus holding control environment
Supports categorical feature handling variants (full embedding, one-hot, none)
and configurable route speed variance (sigma).
"""

import tracemalloc
import json
import os
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env.sim import env_bus
from normalization import Normalization, RewardScaling, RunningMeanStd
from bus_feature_utils import create_embedding_layer, build_bus_categorical_info
from bus_replay_buffer import ReplayBuffer

GPU = True
device_idx = 0
if GPU:
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


def _discover_latest_checkpoint(directory: Path, script_name: str):
    if not directory.exists():
        return None
    prefix = f"{script_name}_episode_"
    max_ep = None
    for file in directory.iterdir():
        name = file.name
        if name.startswith(prefix) and name.endswith('_actor'):
            try:
                ep = int(name[len(prefix):-len('_actor')])
                if max_ep is None or ep > max_ep:
                    max_ep = ep
            except ValueError:
                continue
    return max_ep


def _load_history(list_path: Path):
    if list_path.exists():
        return list(np.load(list_path))
    return []


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


parser = argparse.ArgumentParser(description='DDPG baseline for bus holding control.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--use_gradient_clip', type=bool, default=True, help="Enable gradient clipping for stability")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Apply running state normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Apply reward scaling")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--training_freq", type=int, default=10, help="Frequency of gradient updates")
parser.add_argument("--plot_freq", type=int, default=1, help="Frequency of plotting/logging")
parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for updates")
parser.add_argument("--max_episodes", type=int, default=500, help="Maximum training episodes")
parser.add_argument("--actor_lr", type=float, default=1e-4, help="Actor learning rate")
parser.add_argument("--critic_lr", type=float, default=1e-4, help="Critic learning rate")
parser.add_argument("--tau", type=float, default=5e-3, help="Soft update coefficient for target networks")
parser.add_argument("--exploration_noise", type=float, default=0.1, help="Gaussian noise scale for exploration actions")
parser.add_argument('--save_root', type=str, default='.', help='Base directory for saving outputs')
parser.add_argument('--run_name', type=str, default='ddpg_baseline', help='Identifier appended to save directories')
parser.add_argument('--env_path', type=str, default='env', help='Path to environment configuration directory')
parser.add_argument('--embedding_mode', type=str, default='full', choices=['full', 'one_hot', 'none'],
                    help='Categorical feature handling mode')
parser.add_argument('--route_sigma', type=float, default=1.5, help='Sigma used for route speed sampling')
parser.add_argument('--eval_sigmas', type=float, nargs='*', default=None, help='List of sigma values for cross-evaluation after training')
parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint in the current run directory')
args = parser.parse_args()

args.embedding_mode = args.embedding_mode.lower()

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RUN_NAME = args.run_name.strip() if args.run_name else None
SAVE_ROOT = os.path.abspath(args.save_root)

sigma_token = f"sigma{args.route_sigma}".replace('.', 'p')
experiment_components = [SCRIPT_NAME, sigma_token, f"embed-{args.embedding_mode}"]
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

MODEL_PREFIX = os.path.join(MODEL_DIR, 'ddpg_bus')


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, embedding_layer, action_range, init_w=3e-3):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.num_categorical = len(self.embedding_layer.cat_cols)
        self.action_range = action_range

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim)

        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)

    def _project_state(self, state):
        cat_tensor = state[:, :self.num_categorical]
        num_tensor = state[:, self.num_categorical:]
        embedding = self.embedding_layer(cat_tensor.long())
        return torch.cat([embedding, num_tensor], dim=1)

    def forward(self, state):
        x = self._project_state(state)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        raw_action = torch.tanh(self.output_layer(x))
        return (raw_action + 1.0) * (self.action_range / 2.0)

    def sample_action(self, state, noise_scale=0.0):
        action = self.forward(state)
        if noise_scale > 0:
            noise = torch.normal(mean=0.0, std=noise_scale, size=action.shape, device=action.device)
            action = action + noise
        return action.clamp(0.0, self.action_range)

    def select_action(self, state, noise_scale=0.0, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        if deterministic:
            action = self.forward(state_tensor)
        else:
            action = self.sample_action(state_tensor, noise_scale=noise_scale)
        return action.detach().cpu().numpy()[0]


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, embedding_layer, init_w=3e-3):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.num_categorical = len(self.embedding_layer.cat_cols)

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)

    def _project_state(self, state):
        cat_tensor = state[:, :self.num_categorical]
        num_tensor = state[:, self.num_categorical:]
        embedding = self.embedding_layer(cat_tensor.long())
        return torch.cat([embedding, num_tensor], dim=1)

    def forward(self, state, action):
        state_features = self._project_state(state)
        x = torch.cat([state_features, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.output_layer(x)


class DDPGTrainer:
    def __init__(self, env, replay_buffer, hidden_dim, action_range, action_dim, embedding_mode,
                 actor_lr=1e-4, critic_lr=1e-4, gamma=0.99, tau=5e-3):
        cat_cols, cat_code_dict = build_bus_categorical_info(env)
        self.num_cat_features = len(cat_cols)
        self.num_cont_features = env.state_dim - self.num_cat_features

        embedding_kwargs = {'layer_norm': True, 'dropout': 0.05} if embedding_mode == 'full' else {}
        embedding_template = create_embedding_layer(embedding_mode, cat_code_dict, cat_cols, **embedding_kwargs)
        state_dim = embedding_template.output_dim + self.num_cont_features

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau

        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone(), action_range).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone(), action_range).to(device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone()).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim, embedding_template.clone()).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.criterion = nn.MSELoss()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        initial_mean = [360., 360., 90.]
        initial_std = [165., 133., 45.]
        running_ms = RunningMeanStd(shape=(self.num_cont_features,), init_mean=initial_mean, init_std=initial_std)
        self.state_norm = Normalization(num_categorical=self.num_cat_features, num_numerical=self.num_cont_features,
                                        running_ms=running_ms)
        self.reward_scaling = RewardScaling(shape=1, gamma=gamma)

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def select_action(self, state, noise_scale=0.0, deterministic=False):
        return self.actor.select_action(state, noise_scale=noise_scale, deterministic=deterministic)

    def update(self, batch_size, reward_scale=1.0, use_gradient_clip=True):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward_scale * reward + (1 - done) * self.gamma * target_q

        current_q = self.critic(state, action)
        critic_loss = self.criterion(current_q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q.mean().item()
        }

    def save_model(self, path_prefix):
        torch.save(self.actor.state_dict(), path_prefix + '_actor')
        torch.save(self.critic.state_dict(), path_prefix + '_critic')
        torch.save(self.actor_target.state_dict(), path_prefix + '_actor_target')
        torch.save(self.critic_target.state_dict(), path_prefix + '_critic_target')

    def load_model(self, path_prefix):
        actor_path = path_prefix + '_actor'
        critic_path = path_prefix + '_critic'
        actor_target_path = path_prefix + '_actor_target'
        critic_target_path = path_prefix + '_critic_target'

        print(f"Loading actor weights from {actor_path}")
        self.actor.load_state_dict(_safe_torch_load(actor_path))
        print(f"Loading critic weights from {critic_path}")
        self.critic.load_state_dict(_safe_torch_load(critic_path))
        if os.path.exists(actor_target_path):
            print(f"Loading actor target weights from {actor_target_path}")
            self.actor_target.load_state_dict(_safe_torch_load(actor_target_path))
        else:
            self.actor_target.load_state_dict(self.actor.state_dict())
        if os.path.exists(critic_target_path):
            print(f"Loading critic target weights from {critic_target_path}")
            self.critic_target.load_state_dict(_safe_torch_load(critic_target_path))
        else:
            self.critic_target.load_state_dict(self.critic.state_dict())

        # Keep target nets in eval mode parity
        self.actor_target.eval()
        self.critic_target.eval()
        self.actor.eval()
        self.critic.eval()
        self.actor.train()
        self.critic.train()


def evaluate_policy(trainer, env, num_eval_episodes=5):
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
                            state_input = trainer.state_norm(raw_state, update=False)
                        else:
                            state_input = raw_state
                        action_dict[key] = trainer.select_action(state_input, deterministic=True)
                elif len(state_dict[key]) == 2:
                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        episode_reward += reward_dict[key]

                    state_dict[key] = state_dict[key][1:]
                    raw_state = np.array(state_dict[key][0])
                    if args.use_state_norm:
                        state_input = trainer.state_norm(raw_state, update=False)
                    else:
                        state_input = raw_state
                    action_dict[key] = trainer.select_action(state_input, deterministic=True)

            state_dict, reward_dict, done = env.step(action_dict, render=False)

        eval_rewards.append(episode_reward)

    mean_reward = np.mean(eval_rewards)
    reward_std = np.std(eval_rewards)
    return mean_reward, reward_std


def plot(rewards, critic_losses, actor_losses):
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    clear_output(True)
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward")
    plt.legend()
    plt.title("Training Reward")

    plt.subplot(1, 2, 2)
    plt.plot(critic_losses, label="Critic Loss")
    plt.plot(actor_losses, label="Actor Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.savefig(os.path.join(PIC_DIR, f'ddpg_training_{len(rewards)}.png'))
    plt.close()


replay_buffer_size = int(1e6)
replay_buffer = ReplayBuffer(replay_buffer_size)

debug = False
render = False
path = os.path.abspath(args.env_path)
env = env_bus(path, debug=debug, route_sigma=args.route_sigma)
env.reset()

action_dim = env.action_space.shape[0]
action_range = env.action_space.high[0]

hidden_dim = 32
trainer = DDPGTrainer(
    env,
    replay_buffer,
    hidden_dim=hidden_dim,
    action_range=action_range,
    action_dim=action_dim,
    embedding_mode=args.embedding_mode,
    actor_lr=args.actor_lr,
    critic_lr=args.critic_lr,
    gamma=args.gamma,
    tau=args.tau,
)

step = 0
step_trained = 0
update_itr = 1

rewards = []
critic_losses = []
actor_losses = []
q_values = []

eval_episodes = []
eval_mean_rewards = []
eval_reward_stds = []

model_path = MODEL_PREFIX
tracemalloc.start()

start_episode = 0
if args.resume:
    model_dir_path = Path(MODEL_DIR)
    latest_episode = _discover_latest_checkpoint(model_dir_path, SCRIPT_NAME)
    if latest_episode is not None:
        checkpoint_prefix = str(model_dir_path / f"{SCRIPT_NAME}_episode_{latest_episode}")
        print(f"Resuming from checkpoint: {checkpoint_prefix}")
        trainer.load_model(checkpoint_prefix)
        start_episode = latest_episode + 1

        rewards = _load_history(Path(LOG_DIR) / 'rewards.npy')
        critic_losses = _load_history(Path(LOG_DIR) / 'critic_losses.npy')
        actor_losses = _load_history(Path(LOG_DIR) / 'actor_losses.npy')
        q_values = _load_history(Path(LOG_DIR) / 'q_values.npy')
        eval_episodes = _load_history(Path(LOG_DIR) / 'eval_episodes.npy')
        eval_mean_rewards = _load_history(Path(LOG_DIR) / 'eval_mean_rewards.npy')
        eval_reward_stds = _load_history(Path(LOG_DIR) / 'eval_reward_stds.npy')

        # Ensure numeric lists (np.load returns ndarray)
        rewards = list(rewards)
        critic_losses = list(critic_losses)
        actor_losses = list(actor_losses)
        q_values = list(q_values)
        eval_episodes = list(eval_episodes)
        eval_mean_rewards = list(eval_mean_rewards)
        eval_reward_stds = list(eval_reward_stds)

        if len(rewards) > start_episode:
            rewards = rewards[:start_episode]

        if len(eval_episodes) > 0 and eval_episodes[-1] >= start_episode:
            indices = [idx for idx, ep in enumerate(eval_episodes) if ep >= start_episode]
            if indices:
                cutoff = indices[0]
                eval_episodes = eval_episodes[:cutoff]
                eval_mean_rewards = eval_mean_rewards[:cutoff]
                eval_reward_stds = eval_reward_stds[:cutoff]

        step = len(q_values)
        step_trained = step
    else:
        print("Resume requested but no checkpoint found; starting fresh.")

if __name__ == '__main__':
    if args.train:
        if start_episode >= args.max_episodes:
            print(f"All {args.max_episodes} episodes already completed. Nothing to do.")
        for eps in range(start_episode, args.max_episodes):
            if eps != 0:
                env.reset()
            state_dict, reward_dict, _ = env.initialize_state(render=render)

            done = False
            episode_steps = 0
            training_steps = 0
            action_dict = {key: None for key in list(range(env.max_agent_num))}
            episode_reward = 0

            while not done:
                for key in state_dict:
                    if len(state_dict[key]) == 1:
                        if action_dict[key] is None:
                            raw_state = np.array(state_dict[key][0])
                            if args.use_state_norm:
                                state_input = trainer.state_norm(raw_state)
                            else:
                                state_input = raw_state
                            action_dict[key] = trainer.select_action(
                                state_input,
                                noise_scale=args.exploration_noise,
                                deterministic=False
                            )
                    elif len(state_dict[key]) == 2:
                        if state_dict[key][0][1] != state_dict[key][1][1]:
                            if args.use_state_norm:
                                state = trainer.state_norm(np.array(state_dict[key][0]))
                                next_state = trainer.state_norm(np.array(state_dict[key][1]))
                            else:
                                state = np.array(state_dict[key][0])
                                next_state = np.array(state_dict[key][1])

                            if args.use_reward_scaling:
                                reward = trainer.reward_scaling(reward_dict[key])
                            else:
                                reward = reward_dict[key]

                            replay_buffer.push(state, action_dict[key], reward, next_state, done)
                            episode_reward += reward_dict[key]
                            episode_steps += 1
                            step += 1

                        state_dict[key] = state_dict[key][1:]
                        raw_state = np.array(state_dict[key][0])
                        if args.use_state_norm:
                            state_input = trainer.state_norm(raw_state)
                        else:
                            state_input = raw_state
                        action_dict[key] = trainer.select_action(
                            state_input,
                            noise_scale=args.exploration_noise,
                            deterministic=False
                        )

                state_dict, reward_dict, done = env.step(action_dict, debug=debug, render=render)

                if len(replay_buffer) > args.batch_size and len(replay_buffer) % args.training_freq == 0 and step_trained != step:
                    step_trained = step
                    for _ in range(update_itr):
                        update_stats = trainer.update(
                            args.batch_size,
                            reward_scale=1.0,
                            use_gradient_clip=args.use_gradient_clip
                        )
                        training_steps += 1
                        critic_losses.append(update_stats['critic_loss'])
                        actor_losses.append(update_stats['actor_loss'])
                        q_values.append(update_stats['q_value'])

                if done:
                    replay_buffer.last_episode_step = episode_steps
                    break

            rewards.append(episode_reward)

            if eps % args.plot_freq == 0:
                plot(rewards, critic_losses, actor_losses)
                np.save(os.path.join(LOG_DIR, 'rewards.npy'), rewards)
                np.save(os.path.join(LOG_DIR, 'critic_losses.npy'), critic_losses)
                np.save(os.path.join(LOG_DIR, 'actor_losses.npy'), actor_losses)
                np.save(os.path.join(LOG_DIR, 'q_values.npy'), q_values)

                mean_reward, reward_std = evaluate_policy(trainer, env, num_eval_episodes=15)
                eval_episodes.append(eps)
                eval_mean_rewards.append(mean_reward)
                eval_reward_stds.append(reward_std)
                np.save(os.path.join(LOG_DIR, 'eval_episodes.npy'), eval_episodes)
                np.save(os.path.join(LOG_DIR, 'eval_mean_rewards.npy'), eval_mean_rewards)
                np.save(os.path.join(LOG_DIR, 'eval_reward_stds.npy'), eval_reward_stds)

                trainer.save_model(os.path.join(LOG_DIR, f'ddpg_episode_{eps}'))
                trainer.save_model(f"{model_path}_episode_{eps}")

    if args.eval_sigmas:
        cross_eval_results = []
        for eval_sigma in args.eval_sigmas:
            eval_env = env_bus(path, debug=debug, route_sigma=eval_sigma)
            eval_env.reset()
            mean_reward, reward_std = evaluate_policy(trainer, eval_env, num_eval_episodes=15)
            cross_eval_results.append({
                "train_sigma": args.route_sigma,
                "eval_sigma": eval_sigma,
                "mean_reward": float(mean_reward),
                "reward_std": float(reward_std),
                "embedding_mode": args.embedding_mode,
                "algorithm": SCRIPT_NAME
            })

        with open(os.path.join(LOG_DIR, 'cross_sigma_eval.json'), 'w') as f:
            json.dump(cross_eval_results, f, indent=2)
