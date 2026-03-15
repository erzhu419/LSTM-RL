"""
评估脚本：使用保存的权重文件评估模型性能
该脚本可以加载训练好的Actor网络权重，并在环境中进行评估，绘制性能曲线
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from env.sim import env_bus
import argparse
import os
from tqdm import tqdm

# 导入网络定义（需要从原始训练脚本中导入）
from sac_v2_bus_ensemble import PolicyNetwork, EmbeddingLayer

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description='Evaluate saved model weights.')
parser.add_argument('--weight_dir', type=str,
                    default='model/sac_v2_bus_ensemble/replay_buffer_size_1000000/critic_actor_ratio_2/maximum_alpha_0.3',
                    help='directory containing weight files')
parser.add_argument('--weight_prefix', type=str, default='weight_reg_0.03',
                    help='prefix of weight files to evaluate')
parser.add_argument('--eval_episodes', type=int, default=10,
                    help='number of episodes for evaluation per weight file')
parser.add_argument('--max_steps', type=int, default=5000,
                    help='max steps per episode')
args = parser.parse_args()


def get_weight_files(weight_dir, prefix):
    """获取所有匹配的权重文件并按episode number排序"""
    all_files = os.listdir(weight_dir)
    weight_files = []

    for f in all_files:
        if f.startswith(prefix + ' ') and os.path.isfile(os.path.join(weight_dir, f)):
            try:
                # 提取episode number
                episode_num = int(f.split(' ')[-1])
                weight_files.append((episode_num, f))
            except:
                pass

    # 按episode number排序
    weight_files.sort(key=lambda x: x[0])
    return weight_files


def evaluate_policy(policy_net, env, num_episodes=10, max_steps=5000):
    """评估policy网络的性能"""
    episode_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # 使用确定性策略（mean action）进行评估
            with torch.no_grad():
                mean, log_std = policy_net.forward(state_tensor)
                # 使用mean action进行评估（不添加噪声）
                action_0 = torch.tanh(mean)
                action = (policy_net.action_range / 2 * action_0 + policy_net.action_range / 2)
                action = action.cpu().numpy()[0]

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards), np.std(episode_rewards)


def main():
    # 初始化环境
    print("Initializing environment...")
    env = env_bus()

    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 64

    # 创建embedding layer（需要根据实际情况调整）
    # 这里的参数需要与训练时保持一致
    cat_code_dict = {
        'bus_id': list(range(25)),
        'station_id': list(range(22)),
        'time_period': list(range(15)),
        'direction': list(range(2))
    }
    cat_cols = ['bus_id', 'station_id', 'time_period', 'direction']
    embedding_layer = EmbeddingLayer(cat_code_dict, cat_cols)

    # 计算embedding后的输入维度
    embedding_dim = sum([min(50, len(cat_code_dict[col]) // 2) for col in cat_cols])
    num_inputs = embedding_dim + (state_dim - len(cat_cols))

    # 创建policy网络
    policy_net = PolicyNetwork(
        num_inputs=num_inputs,
        num_actions=action_dim,
        hidden_size=hidden_dim,
        embedding_layer=embedding_layer,
        action_range=1.0
    ).to(device)

    # 获取所有权重文件
    print(f"Loading weight files from {args.weight_dir}...")
    weight_files = get_weight_files(args.weight_dir, args.weight_prefix)
    print(f"Found {len(weight_files)} weight files")

    if len(weight_files) == 0:
        print(f"No weight files found with prefix '{args.weight_prefix}' in {args.weight_dir}")
        return

    # 评估每个权重文件
    results = []
    episodes = []
    mean_rewards = []
    std_rewards = []

    print(f"\nEvaluating {len(weight_files)} checkpoints...")
    for episode_num, weight_file in tqdm(weight_files):
        weight_path = os.path.join(args.weight_dir, weight_file)

        # 加载权重
        try:
            policy_net.load_state_dict(torch.load(weight_path, map_location=device))
            policy_net.eval()

            # 评估
            mean_reward, std_reward = evaluate_policy(
                policy_net, env,
                num_episodes=args.eval_episodes,
                max_steps=args.max_steps
            )

            episodes.append(episode_num)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)

            results.append({
                'episode': episode_num,
                'mean_reward': mean_reward,
                'std_reward': std_reward
            })

            print(f"Episode {episode_num}: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")

        except Exception as e:
            print(f"Error loading/evaluating {weight_file}: {e}")
            continue

    # 保存结果
    results_file = os.path.join(args.weight_dir, f'{args.weight_prefix}_eval_results.npz')
    np.savez(results_file,
             episodes=episodes,
             mean_rewards=mean_rewards,
             std_rewards=std_rewards)
    print(f"\nResults saved to {results_file}")

    # 绘制曲线
    plt.figure(figsize=(12, 6))

    episodes = np.array(episodes)
    mean_rewards = np.array(mean_rewards)
    std_rewards = np.array(std_rewards)

    plt.plot(episodes, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
    plt.fill_between(episodes,
                     mean_rewards - std_rewards,
                     mean_rewards + std_rewards,
                     alpha=0.3, color='b', label='±1 Std')

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.title(f'Evaluation Results for {args.weight_prefix}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 保存图片
    plot_file = os.path.join(args.weight_dir, f'{args.weight_prefix}_eval_curve.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")

    plt.show()

    # 打印最佳结果
    best_idx = np.argmax(mean_rewards)
    print(f"\n{'='*60}")
    print(f"Best Performance:")
    print(f"Episode: {episodes[best_idx]}")
    print(f"Mean Reward: {mean_rewards[best_idx]:.2f} ± {std_rewards[best_idx]:.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
