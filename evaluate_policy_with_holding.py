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
    all_bunching_events = []

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

            state_dict, reward_dict, done = env.step(action_dict, render=False, debug=True)

        eval_rewards.append(episode_reward)
        
        # 提取bunching事件
        bunching_events = env.visualizer.extract_bunching_events()
        all_bunching_events.extend(bunching_events)

    mean_reward = np.mean(eval_rewards)
    reward_std = np.std(eval_rewards)
    return mean_reward, reward_std, all_events, all_bunching_events


def plot_holding_events(events, min_time=None, max_time=None, exp='0', policy_name=None):
    if not events:
        return
    exp = str(exp)
    path = os.getcwd()
    
    # 构建保存路径，如果有策略名称则添加到路径中
    save_dir = os.path.join(path, 'env/pic')
    if policy_name:
        save_dir = os.path.join(save_dir, policy_name)
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
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
    plt.savefig(os.path.join(save_dir, f'exp {exp}, holding events.jpg'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_runs', type=int, default=1, help='number of evaluation episodes')
    parser.add_argument('--enable_plot', default=True, action='store_true', help='enable trajectory plotting')
    parser.add_argument('--policy_path', type=str, default='/home/erzhu419/mine_code/LSTM-RL/model/sac_v2_bus/sac_v2_bus_episode_173_policy')
    # 添加策略名称参数，用于区分不同策略的输出路径
    parser.add_argument('--policy_name', type=str, default='SAC', help='policy name for plot output directory')
    args = parser.parse_args()

    env_path = os.path.join(os.getcwd(), 'env')
    env = env_bus(env_path, debug=True)
    env.reset()
    env.enable_plot = args.enable_plot  # 设置是否启用轨迹绘制

    replay_buffer = ReplayBuffer(1)
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]
    sac_trainer = SAC_Trainer(env, replay_buffer, hidden_dim=32, action_range=action_range)

    sac_trainer.policy_net.load_state_dict(torch.load(args.policy_path, map_location=torch.device('cuda:0'), weights_only=True))

    sac_trainer.policy_net.eval()

    mean_reward, reward_std, events, bunching_events = evaluate_policy_with_holding(
        sac_trainer, env, num_eval_episodes=args.eval_runs, deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} +/- {reward_std:.2f}")

    df = pd.DataFrame(events)
    df = df[df['duration'] > 40]
    os.makedirs('pic', exist_ok=True)
    
    # 如果存在策略名，创建相应子目录并保存到该目录
    if args.policy_name:
        policy_dir = os.path.join('env/pic', args.policy_name)
        os.makedirs(policy_dir, exist_ok=True)
        df.to_csv(os.path.join(policy_dir, 'holding_records.csv'), index=False)
    else:
        df.to_csv(os.path.join('env/pic', 'holding_records.csv'), index=False)
        
    # 调用 plot_holding_events 时传入 policy_name 参数
    plot_holding_events(events, exp=str(args.eval_runs), policy_name=args.policy_name)
    
    # 绘制bunching事件图
    if bunching_events:
        bunching_df = pd.DataFrame(bunching_events).sort_values(['time'])
        # 保存bunching记录
        if args.policy_name:
            policy_dir = os.path.join('env/pic', args.policy_name)
            os.makedirs(policy_dir, exist_ok=True)
            bunching_df.to_csv(os.path.join(policy_dir, 'bunching_records.csv'), index=False)
        else:
            bunching_df.to_csv(os.path.join('env/pic', 'bunching_records.csv'), index=False)
        
        # 绘制bunching事件图
        # env.visualizer.plot_bunching_events(bunching_events, exp=str(args.eval_runs), policy_name=args.policy_name)
    
    # 绘制轨迹图时传入 policy_name 参数，以便输出到正确的目录
    if env.enable_plot:
        env.visualizer.plot(exp=args.eval_runs, policy_name=args.policy_name)