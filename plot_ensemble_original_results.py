import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# 解析日志文件
def parse_log_file(log_path):
    episodes = []
    rewards = []

    with open(log_path, 'r') as f:
        for line in f:
            # 匹配 Episode 和 Reward
            match = re.search(r'Episode:\s*(\d+)\s*\|\s*Episode Reward:\s*([-\d.]+)', line)
            if match:
                episode = int(match.group(1))
                reward = float(match.group(2))
                episodes.append(episode)
                rewards.append(reward)

    return np.array(episodes), np.array(rewards)

# 解析文件名获取配置
def parse_filename(filename):
    # wr0.01_a0.3_r2_buf100000.log
    match = re.search(r'wr([\d.]+)_a([\d.]+)_r(\d+)_buf(\d+)', filename)
    if match:
        return {
            'weight_reg': float(match.group(1)),
            'max_alpha': float(match.group(2)),
            'critic_actor_ratio': int(match.group(3)),
            'buffer_size': int(match.group(4))
        }
    return None

# 平滑曲线
def smooth_curve(data, window_size=10):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

# 收集所有日志文件
log_dir = Path('logs_ensemble_original')
log_files = sorted(log_dir.glob('*.log'))

print(f"找到 {len(log_files)} 个日志文件")

# 解析所有日志
results = []
for log_file in log_files:
    config = parse_filename(log_file.name)
    if config:
        episodes, rewards = parse_log_file(log_file)
        if len(episodes) > 0:
            config['episodes'] = episodes
            config['rewards'] = rewards
            config['filename'] = log_file.name
            results.append(config)
            print(f"解析 {log_file.name}: {len(episodes)} episodes, 最终reward: {rewards[-1]:.2f}")

# 创建多个子图
fig = plt.figure(figsize=(20, 12))

# 1. 按 buffer_size 分组
plt.subplot(2, 3, 1)
for buffer_size in [100000, 1000000]:
    subset = [r for r in results if r['buffer_size'] == buffer_size]
    if subset:
        for config in subset:
            smoothed = smooth_curve(config['rewards'], window_size=20)
            label = f"buf={buffer_size//1000}k, α={config['max_alpha']}, r={config['critic_actor_ratio']}"
            plt.plot(config['episodes'], smoothed, label=label, alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Curves by Buffer Size')
plt.legend(fontsize=8, loc='best')
plt.grid(True, alpha=0.3)

# 2. 按 max_alpha 分组
plt.subplot(2, 3, 2)
for alpha in sorted(set(r['max_alpha'] for r in results)):
    subset = [r for r in results if r['max_alpha'] == alpha]
    for config in subset:
        smoothed = smooth_curve(config['rewards'], window_size=20)
        label = f"α={alpha}, r={config['critic_actor_ratio']}, buf={config['buffer_size']//1000}k"
        plt.plot(config['episodes'], smoothed, label=label, alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Curves by Max Alpha')
plt.legend(fontsize=8, loc='best')
plt.grid(True, alpha=0.3)

# 3. 按 critic_actor_ratio 分组
plt.subplot(2, 3, 3)
for ratio in sorted(set(r['critic_actor_ratio'] for r in results)):
    subset = [r for r in results if r['critic_actor_ratio'] == ratio]
    for config in subset:
        smoothed = smooth_curve(config['rewards'], window_size=20)
        label = f"r={ratio}, α={config['max_alpha']}, buf={config['buffer_size']//1000}k"
        plt.plot(config['episodes'], smoothed, label=label, alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Curves by Critic-Actor Ratio')
plt.legend(fontsize=8, loc='best')
plt.grid(True, alpha=0.3)

# 4. Buffer size = 100000 的对比
plt.subplot(2, 3, 4)
subset = [r for r in results if r['buffer_size'] == 100000]
for config in subset:
    smoothed = smooth_curve(config['rewards'], window_size=20)
    label = f"α={config['max_alpha']}, r={config['critic_actor_ratio']}"
    plt.plot(config['episodes'], smoothed, label=label, alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Buffer Size = 100k Comparison')
plt.legend(fontsize=8, loc='best')
plt.grid(True, alpha=0.3)

# 5. Buffer size = 1000000 的对比
plt.subplot(2, 3, 5)
subset = [r for r in results if r['buffer_size'] == 1000000]
for config in subset:
    smoothed = smooth_curve(config['rewards'], window_size=20)
    label = f"α={config['max_alpha']}, r={config['critic_actor_ratio']}"
    plt.plot(config['episodes'], smoothed, label=label, alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Buffer Size = 1000k Comparison')
plt.legend(fontsize=8, loc='best')
plt.grid(True, alpha=0.3)

# 6. 最终性能对比（最后50个episode的平均reward）
plt.subplot(2, 3, 6)
configs_labels = []
final_rewards = []
for config in results:
    last_50_mean = np.mean(config['rewards'][-50:])
    final_rewards.append(last_50_mean)
    label = f"α={config['max_alpha']}\nr={config['critic_actor_ratio']}\nbuf={config['buffer_size']//1000}k"
    configs_labels.append(label)

indices = np.argsort(final_rewards)[::-1]  # 降序排列
colors = plt.cm.viridis(np.linspace(0, 1, len(final_rewards)))
plt.barh(range(len(final_rewards)), [final_rewards[i] for i in indices], color=[colors[i] for i in indices])
plt.yticks(range(len(final_rewards)), [configs_labels[i] for i in indices], fontsize=7)
plt.xlabel('Mean Reward (Last 50 Episodes)')
plt.title('Final Performance Comparison')
plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('ensemble_original_500ep_analysis.png', dpi=300, bbox_inches='tight')
print("\n图表已保存为: ensemble_original_500ep_analysis.png")
plt.close()  # 关闭图形

# 打印统计信息
print("\n=== 最终性能统计 (最后50个episode的平均) ===")
for i in indices:
    config = results[i]
    print(f"{configs_labels[i].replace(chr(10), ', ')}: {final_rewards[i]:.2f}")

# 创建详细的统计表格
print("\n=== 详细统计表格 ===")
stats_data = []
for config in results:
    last_50 = config['rewards'][-50:]
    last_100 = config['rewards'][-100:]
    stats_data.append({
        'buffer_size': config['buffer_size'],
        'max_alpha': config['max_alpha'],
        'critic_actor_ratio': config['critic_actor_ratio'],
        'mean_last_50': np.mean(last_50),
        'std_last_50': np.std(last_50),
        'mean_last_100': np.mean(last_100),
        'std_last_100': np.std(last_100),
        'max_reward': np.max(config['rewards']),
        'min_reward': np.min(config['rewards'])
    })

df = pd.DataFrame(stats_data)
df = df.sort_values('mean_last_50', ascending=False)
print(df.to_string(index=False))

# 保存统计数据到CSV
df.to_csv('ensemble_original_500ep_stats.csv', index=False)
print("\n统计数据已保存为: ensemble_original_500ep_stats.csv")
