import os
import argparse
import torch
import numpy as np
import pandas as pd
from sac_v2_bus import SAC_Trainer, ReplayBuffer
from sac_v2_bus import evaluate_policy  # for reference
from env.sim import env_bus
import matplotlib.pyplot as plt


def record_event(all_events, bus_id, act, state_dict, station_map, current_time, ep):
    # record holding actions before stepping the environment

    if act is not None and act[0] > 0 and bus_id in state_dict:
        station_id = state_dict[bus_id][0][1]
        station_name = station_map.get(station_id, str(station_id))
        direction = state_dict[bus_id][0][3]
        all_events.append({
            'run': ep + 1,
            'bus_id': bus_id,
            'station': station_name,
            'time': current_time,
            'duration': int(act[0]),
            'direction': direction,
        })


def evaluate_policy_with_holding(sac_trainer, env, num_eval_episodes=5, deterministic=True):
    """Evaluate a trained policy and record holding events."""
    eval_rewards = []
    all_events = []

    # mapping from station_id to station_name for quick lookup
    station_map = {s.station_id: s.station_name for s in env.stations}

    for ep in range(num_eval_episodes):
        env.reset()
        state_dict, reward_dict, _ = env.initialize_state(render=False)
        done = False
        episode_reward = 0
        action_dict = {key: None for key in range(env.max_agent_num)}

        while not done:
            for key in list(state_dict.keys()):
                if len(state_dict[key]) == 1:
                    if action_dict[key] is None:
                        state_input = np.array(state_dict[key][0])
                        a = sac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(),deterministic=deterministic)
                        action_dict[key] = a
                        record_event(all_events, key, a, state_dict, station_map, env.current_time, ep)


                elif len(state_dict[key]) == 2:
                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        episode_reward += reward_dict[key]
                    state_dict[key] = state_dict[key][1:]
                    state_input = np.array(state_dict[key][0])
                    action_dict[key] = sac_trainer.policy_net.get_action(torch.from_numpy(state_input).float(),deterministic=deterministic)
                    a = action_dict[key]
                    record_event(all_events, key, a, state_dict, station_map, env.current_time, ep)

            state_dict, reward_dict, done = env.step(action_dict, render=False)

        eval_rewards.append(episode_reward)

    mean_reward = np.mean(eval_rewards)
    reward_std = np.std(eval_rewards)
    return mean_reward, reward_std, all_events


def plot_holding_events(events, min_time=None, max_time=None, exp='0'):
    if not events:
        return
    exp = str(exp)
    path = os.getcwd()
    if min_time is None:
        min_time = min(e['time'] for e in events)
    if max_time is None:
        max_time = max(e['time'] for e in events)

    plt.figure(figsize=(96, 24), dpi=300)
    x1 = np.linspace(min_time, max_time, num=500)
    station_names = ['Terminal up'] + [f'X{i:02d}' for i in range(1, 21)] + ['Terminal down']
    for j in range(len(station_names)):
        y1 = [j * 500] * len(x1)
        plt.plot(x1, y1, color="red", linewidth=0.3, linestyle='-')

    station_y = {name: i * 500 for i, name in enumerate(station_names)}
    colors = {1: 'blue', 0: 'green', True: 'blue', False: 'green'}
    for event in events:
        if event['station'] in station_y and event['duration'] > 40:
            plt.scatter(event['time'], station_y[event['station']],
                        color=colors.get(event['direction'], 'black'),
                        s=max(event['duration'], 1)*3)

    plt.xticks(fontsize=16)
    plt.yticks(ticks=[j * 500 for j in range(len(station_names))],
               labels=station_names, fontsize=16)
    plt.xlabel('time', fontsize=20)
    plt.ylabel('station', fontsize=20)
    plt.title('holding events', fontsize=20)
    plt.xlim(min_time, max_time)
    plt.savefig(os.path.join(path, 'env/pic', f'exp {exp}, holding events.jpg'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_runs', type=int, default=1, help='number of evaluation episodes')

    parser.add_argument('--policy_path', type=str, default='/home/erzhu419/mine_code/LSTM-RL/model/sac_v2_bus_penalty_reward/sac_v2_bus_penalty_reward_episode_352_policy')
    args = parser.parse_args()

    env_path = os.path.join(os.getcwd(), 'env')
    env = env_bus(env_path, debug=False)
    env.reset()

    replay_buffer = ReplayBuffer(1)
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]
    sac_trainer = SAC_Trainer(env, replay_buffer, hidden_dim=32, action_range=action_range)

    sac_trainer.policy_net.load_state_dict(torch.load(args.policy_path, map_location=torch.device('cuda:0'), weights_only=True))

    sac_trainer.policy_net.eval()

    mean_reward, reward_std, events = evaluate_policy_with_holding(
        sac_trainer, env, num_eval_episodes=args.eval_runs, deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} +/- {reward_std:.2f}")

    df = pd.DataFrame(events)
    df = df[df['duration'] > 50]
    os.makedirs('pic', exist_ok=True)
    df.to_csv(os.path.join('env/pic', 'holding_records.csv'), index=False)
    plot_holding_events(events, exp=str(args.eval_runs))