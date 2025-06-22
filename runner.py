import numpy as np
import csv
import os, time
from datetime import datetime
import torch, psutil
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor

model_path = '/home/erzhu419/mine_code/Multi-agent-RL/MADDPG_Continous/models/maddpg_individual_models'  # 区分非参数共享版本
def plot(rewards, q_values_episode, path):
    """
    绘制奖励曲线
    :param rewards: 奖励列表
    :param episode_num: 训练的回合数
    :param window_size: 平滑窗口大小
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，避免Tkinter相关问题
    # 计算平滑奖励

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward")
    plt.plot(q_values_episode, label="Q-Value")
    plt.legend()

    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title('Smoothed Rewards over Episodes')

    plt.grid()

    plt.close()

class RUNNER:
    def __init__(self, agent, env, args, device, mode ='evaluate'):
        self.agent = agent
        self.env = env
        self.args = args
        self.device = device
        # 这里为什么新建而不是直接使用用agent.agents.keys()？
        # 因为pettingzoo中智能体死亡，这个字典就没有了，会导致 td target更新出错。所以这里维护一个不变的字典。
        self.env_agents = [agent_id for agent_id in self.agent.agents.keys()]
        self.done = None

        # 添加奖励记录相关的属性
        self.reward_sum_record = []  # 用于平滑的奖励记录
        self.all_reward_record = []  # 保存所有奖励记录，用于最终统计
        self.all_adversary_avg_rewards = []  # 追捕者平均奖励
        self.all_sum_rewards = []  # 所有智能体总奖励
        self.episode_rewards = 0

        # 将 agent 的模型放到指定设备上
        for agent in self.agent.agents.values():
            agent.actor.to(device)
            agent.target_actor.to(device)
            agent.critic.to(device)
            agent.target_critic.to(device)
        '''
        解决使用visdom过程中，输出控制台阻塞的问题。
        ''' #TODO

        if mode == 'train' and self.args.visdom:
            import visdom
            self.viz = visdom.Visdom()
            self.viz.close()
        else: # evaluate模式下不需要visdom
            pass

    def train(self, render):
        """优化的训练循环，使用批处理和并行处理"""

        # 初始化性能监控
        timing_stats = {
            "select_actions": 0.0,
            "step_env": 0.0,
            "add_experiences": 0.0,
            "train_agents": 0.0,
            "total": 0.0
        }

        # 初始化计数器和记录
        transitions_added = 0
        rewards = []
        agent_rewards = {k: [] for k in self.env_agents}
        q_values = []
        q_values_episode = []
        eval_mean_rewards = []  # 评估平均奖励
        eval_reward_stds = []   # 评估奖励方差
        # 创建活跃智能体集合 - 添加这一行
        active_agents = set()
        # 持久化的action字典
        action_dict = {key: None for key in range(self.env.max_agent_num)}
        # 记录训练步数和智能体步数
        trained_steps = {key: 0 for key in range(self.env.max_agent_num)}
        agent_steps = {key: 0 for key in range(self.env.max_agent_num)}

        # 创建线程池
        executor = ThreadPoolExecutor(max_workers=self.args.max_workers)

        # 并行动作选择函数
        def select_actions_batch(obs_dict):
            """批量为多个智能体选择动作"""
            if not obs_dict:
                return {}

            # 收集需要动作的智能体
            agent_ids = []
            observations = []

            for agent_id, obs in obs_dict.items():
                if len(obs) > 0:
                    agent_ids.append(agent_id)
                    observations.append(obs[-1][2:] if isinstance(obs, list) else obs[2:])

            if not agent_ids:
                return {}

            # 批量转换为张量
            try:
                batch_tensor = torch.FloatTensor(observations).to(self.device)

                # 批量计算动作 (这里可能需要修改MADDPG实现以支持批处理)
                with torch.no_grad():
                    actions = {}
                    for i, agent_id in enumerate(agent_ids):
                        a, _ = self.agent.agents[agent_id].action(batch_tensor[i:i + 1])
                        actions[agent_id] = a.squeeze(0).cpu().numpy()

                return actions
            except Exception as e:
                print(f"Error in batch action selection: {e}")
                return {}

        # episode循环
        for episode in range(self.args.episode_num):
            # 重置计数和环境
            ep_start_time = time.time()
            self.episode_rewards = 0
            self.episode_agent_rewards = {k: 0 for k in self.env_agents}
            training_steps = 0

            # print(f"Episode {episode}")
            self.env.reset()
            obs, agent_reward, self.done = self.env.initialize_state(render)

            # 重置action_dict
            for key in action_dict:
                action_dict[key] = None

            # 环境交互循环
            step_counter = 0
            while not self.done:
                step_counter += 1

                ### 1. 批量处理动作选择 ###
                t0 = time.time()

                # 第一次到达某个站点的情况
                first_station_states = {k: obs[k] for k in obs if len(obs[k]) == 1 and action_dict[k] is None}
                if first_station_states:
                    new_actions = select_actions_batch(first_station_states)
                    # 更新action_dict
                    for k, a in new_actions.items():
                        action_dict[k] = a
                # 非第一次到达某个站点的情况
                second_station_states = {k: obs[k] for k in obs if len(obs[k]) == 2}
                # 收集状态转换的智能体
                transitions_list = []
                for k, states in second_station_states.items():
                    if states[0][1] != states[1][1]:  # 站点变化
                        transitions_list.append({
                            "agent_id": k,
                            "old_state": states[0][2:],
                            "new_state": states[1][2:],
                            "action": action_dict[k],
                            "reward": agent_reward[k]
                        })
                        # 记录奖励
                        self.episode_rewards += agent_reward[k]
                        self.episode_agent_rewards[k] += agent_reward[k]
                        transitions_added += 1
                        agent_steps[k] += 1

                    # 更新观察
                    obs[k] = [states[1]]
                    

                # # 更新需要新动作的智能体
                # updated_agents = {k: obs[k] for k in state_transitions if k in state_transitions}
                if second_station_states:
                    new_actions = select_actions_batch(second_station_states)
                    for k, a in new_actions.items():
                        action_dict[k] = a

                timing_stats["select_actions"] += time.time() - t0

                ### 2. 批量添加经验 ###
                t0 = time.time()
                if transitions_list:
                    # 并行添加经验
                    future_adds = []
                    for trans in transitions_list:
                        agent_id = trans["agent_id"]
                        active_agents.add(agent_id)
                        local_obs = {agent_id: [trans["old_state"], trans["new_state"]]}
                        local_action = {agent_id: trans["action"]}
                        local_reward = {agent_id: trans["reward"]}

                        # 提交添加任务
                        future_adds.append(
                            executor.submit(
                                self.agent.add,
                                local_obs,
                                local_action,
                                local_reward,
                                None,
                                self.done
                            )
                        )

                    # 等待所有添加完成
                    for future in future_adds:
                        future.result()


                timing_stats["add_experiences"] += time.time() - t0

                ### 3. 环境步进 ###
                t0 = time.time()
                # all zero action
                action_zero = {k: np.zeros(self.env.action_space.shape[0]) for k in action_dict.keys()}
                obs, agent_reward, self.done = self.env.step(action_dict, render=render)
                timing_stats["step_env"] += time.time() - t0

                ### 4. 训练智能体 ###
                t0 = time.time()
                # 修改并行训练部分
                if transitions_added > self.args.random_steps:
                    # 先检查哪些智能体有足够的数据用于训练
                    valid_agents = []
                    for agent_id in active_agents:
                        # 检查这个智能体的缓冲区是否有足够数据
                        buffer_size = len(self.agent.buffers.get(agent_id, []))
                        if buffer_size >= self.args.batch_size:
                            valid_agents.append(agent_id)
                        else:
                            # print(f"Skipping agent {agent_id}: Buffer size {buffer_size} < batch size {self.args.batch_size}")
                            pass
                    # 只为有足够数据的智能体创建学习任务
                    future_learns = {}
                    for agent_id in valid_agents:
                        if trained_steps[agent_id] != agent_steps[agent_id]:
                            future_learns[agent_id] = executor.submit(
                                self.agent.learn,
                                self.args.batch_size,
                                self.args.gamma,
                                agent_id
                            )
                            trained_steps[agent_id] = agent_steps[agent_id]

                    # 收集学习结果
                    q_values_batch = []
                    for agent_id, future in future_learns.items():
                        try:
                            q_value = future.result()
                            if q_value is not None:
                                q_values_batch.append(q_value)
                        except Exception as e:
                            print(f"Error training agent {agent_id}: {e}")

                    # 记录平均Q值
                    if q_values_batch:
                        q_values.append(np.mean(q_values_batch))
                        training_steps += 1

                    # 更新目标网络 (仅为有效智能体)
                    for agent_id in valid_agents:
                        if trained_steps[agent_id] % self.args.training_freq == 0:
                            try:
                                self.agent.update_target_for_agent(agent_id, self.args.tau)
                            except Exception as e:
                                print(f"Error updating target for agent {agent_id}: {e}")


                timing_stats["train_agents"] += time.time() - t0

            # Episode结束，记录数据
            rewards.append(self.episode_rewards)
            agent_rewards = {k: self.episode_agent_rewards[k] for k in self.env_agents}
            if training_steps > 0:
                q_values_episode.append(np.mean(q_values[-training_steps:]))
            elif q_values:
                q_values_episode.append(q_values[-1])

            active_agents.clear()

            # 计算总时间
            episode_time = time.time() - ep_start_time
            timing_stats["total"] += episode_time

            # 绘图、评估和保存模型
            if episode % self.args.plot_interval == 0:
                # 评估当前策略
                mean_reward, reward_std = self.evaluate_policy(num_eval_episodes=5, render=False)
                eval_mean_rewards.append(mean_reward)
                eval_reward_stds.append(reward_std)
                
                # 绘制并保存奖励曲线
                plot(rewards, q_values_episode, model_path)
                
                # 保存训练和评估结果
                save_dir = os.path.dirname(model_path)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存训练奖励
                np.save(os.path.join(save_dir, "rewards_individual.npy"), rewards)
                
                # 保存每个智能体的奖励
                if episode == 0:
                    agent_rewards_matrix = np.zeros((len(self.env_agents), self.args.episode_num))
                for idx, agent_id in enumerate(self.env_agents):
                    agent_rewards_matrix[idx, episode] = agent_rewards[agent_id]
                np.save(os.path.join(save_dir, "agent_rewards_individual.npy"), agent_rewards_matrix)
                
                # 保存Q值和评估结果
                np.save(os.path.join(save_dir, "q_values_individual.npy"), q_values_episode)
                np.save(os.path.join(save_dir, "eval_mean_rewards_individual.npy"), eval_mean_rewards)
                np.save(os.path.join(save_dir, "eval_reward_stds_individual.npy"), eval_reward_stds)
                
                # 保存模型
                self.agent.save_model(episode=episode)

                # 在日志中记录评估结果
                print(f"Evaluation at Episode {episode}: Mean Reward = {mean_reward:.2f}, Std = {reward_std:.2f}")
                
                # 重置计时器
                timing_stats = {k: 0.0 for k in timing_stats}

            # 打印统计信息
            print(
                f"Episode: {episode} | Reward: {self.episode_rewards:.2f} | "
                f"Time: {episode_time:.2f}s | Transitions: {transitions_added} | "
                f"Training Steps: {sum(trained_steps.values())} | "
                f"CPU: {psutil.Process().memory_info().rss / 1024 ** 2:.1f}MB | "
                f"GPU: {torch.cuda.memory_allocated() / 1024 ** 2:.1f}MB | "
                f"Total Steps: {step_counter}"
            )

        # 关闭线程池
        executor.shutdown()
    def get_running_reward(self, arr):

        if len(arr) == 0:  # 如果传入空数组，使用完整记录
            arr = self.all_reward_record

        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        window = self.args.size_win
        running_reward = np.zeros_like(arr)

        # for i in range(window - 1):
        #     running_reward[i] = np.mean(arr[:i + 1])
        # for i in range(window - 1, len(arr)):
        #     running_reward[i] = np.mean(arr[i - window + 1:i + 1])
            # 确保不会访问超出数组范围的位置
        for i in range(len(arr)):
            # 对每个i，确保窗口大小不会超出数组的实际大小
            start_idx = max(0, i - window + 1)
            running_reward[i] = np.mean(arr[start_idx:i + 1])
        # print(f"running_reward{running_reward}")
        return running_reward

    @staticmethod
    def exponential_moving_average(rewards, alpha=0.1):
        """计算指数移动平均奖励"""
        ema_rewards = np.zeros_like(rewards)
        ema_rewards[0] = rewards[0]
        for t in range(1, len(rewards)):
            ema_rewards[t] = alpha * rewards[t] + (1 - alpha) * ema_rewards[t - 1]
        return ema_rewards

    def moving_average(self, rewards):
        """计算简单移动平均奖励"""
        window_size = self.args.size_win
        sma_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        return sma_rewards
    
    """保存围捕者平均奖励和所有智能体总奖励到 CSV 文件"""
    def save_rewards_to_csv(self, adversary_rewards, sum_rewards, eval_means=None, eval_stds=None, filename=None): # filename="data_rewards.csv"
        # 获取当前时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        if filename is None:
            filename = f"data_rewards_{timestamp}.csv"
        # 获取 runner.py 所在目录，并生成与 utils 同级的 plot 目录路径
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件（runner.py）的绝对路径
        plot_dir = os.path.join(current_dir, '..', 'plot', 'data')  # 获取与 utils 同级的 plot 文件夹
        os.makedirs(plot_dir, exist_ok=True)  # 创建 plot 目录（如果不存在）

        # 构造完整的 CSV 文件路径
        full_filename = os.path.join(plot_dir, filename)

        # 添加评估结果到表头和数据中
        if eval_means is not None and eval_stds is not None:
            header = ['Episode', 'Adversary Average Reward', 'Sum Reward of All Agents', 'Evaluation Mean Reward', 'Evaluation Reward Std']
            
            # 确保所有数据长度一致
            episodes = list(range(1, len(adversary_rewards) + 1))
            eval_data = []
            eval_idx = 0
            
            # 对齐评估数据（评估可能不是每个episode都进行）
            for i in range(len(episodes)):
                if i % self.args.plot_interval == 0 and eval_idx < len(eval_means):
                    eval_data.append((eval_means[eval_idx], eval_stds[eval_idx]))
                    eval_idx += 1
                else:
                    eval_data.append((None, None))
            
            data = []
            for i, (adv_reward, sum_reward) in enumerate(zip(adversary_rewards, sum_rewards)):
                if eval_data[i][0] is not None:
                    data.append((episodes[i], adv_reward, sum_reward, eval_data[i][0], eval_data[i][1]))
                else:
                    data.append((episodes[i], adv_reward, sum_reward, "", ""))
        else:
            header = ['Episode', 'Adversary Average Reward', 'Sum Reward of All Agents']
            data = list(zip(range(1, len(adversary_rewards) + 1), adversary_rewards, sum_rewards))
        # 将数据写入 CSV 文件
        with open(full_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # 写入表头
            writer.writerows(data)  # 写入数据

        print(f"Rewards data saved to {full_filename}")
#============================================================================================================

    def evaluate(self):
        # # 使用visdom实时查看训练曲线
        # viz = None
        # if self.par.visdom:
        #     viz = visdom.Visdom()
        #     viz.close()
        # step = 0
        # 记录每个episode的和奖励 用于平滑，显示平滑奖励函数
        self.reward_sum_record = []
        # 记录每个智能体在每个episode的奖励
        self.episode_rewards = {agent_id: np.zeros(self.args.episode_num) for agent_id in self.env.agents}
        # episode循环
        for episode in range(self.args.episode_num):
            step = 0  # 每回合step重置
            print(f"评估第 {episode + 1} 回合")
            # 初始化环境 返回初始状态 为一个字典 键为智能体名字 即env.agents中的内容，内容为对应智能体的状态
            obs, _ = self.env.reset()  # 重置环境，开始新回合
            self.done = {agent_id: False for agent_id in self.env_agents}
            # 每个智能体当前episode的奖励
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            # 每个智能体与环境进行交互
            while self.env.agents:
                # print(f"While num:{step}")
                step += 1
                # 使用训练好的智能体选择动作（没有随机探索）
                action = self.agent.select_action(obs)
                # 执行动作 获得下一状态 奖励 终止情况
                # 下一状态：字典 键为智能体名字 值为对应的下一状态
                # 奖励：字典 键为智能体名字 值为对应的奖励
                # 终止情况：bool
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.done = {agent_id: bool(terminated[agent_id] or truncated[agent_id]) for agent_id in self.env_agents}

                # 累积每个智能体的奖励
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                obs = next_obs

                
                if step % 10 == 0:
                    print(f"Step {step}, obs: {obs}, action: {action}, reward: {reward}, done: {self.done}")

            sum_reward = sum(agent_reward.values())
            self.reward_sum_record.append(sum_reward)
    def evaluate_policy(self, num_eval_episodes=5, render=False):
        """
        评估当前策略，返回多次评估的平均奖励和方差
        :param num_eval_episodes: 评估的episode数量
        :param render: 是否渲染
        :return: (平均奖励, 奖励方差)
        """
        eval_rewards = []
        
        for eval_ep in range(num_eval_episodes):
            episode_reward = 0
            step = 0
            
            # 重置环境
            self.env.reset()
            obs, agent_reward, done = self.env.initialize_state(render)
            
            # 初始化action字典
            action_dict = {key: None for key in range(self.env.max_agent_num)}
            
            while not done:  # 添加步数限制防止无限循环
                step += 1
                
                # 处理第一次到达站点的情况
                first_station_states = {k: obs[k] for k in obs if len(obs[k]) == 1 and action_dict[k] is None}
                if first_station_states:
                    for agent_id, obs_list in first_station_states.items():
                        if len(obs_list) > 0:
                            state_features = obs_list[0][2:]  # 排除bus_id和station_id
                            with torch.no_grad():
                                obs_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
                                action, _ = self.agent.agents[agent_id].action(obs_tensor)  # 评估时不添加噪声
                                action_dict[agent_id] = action.cpu().numpy().squeeze(0)
                
                # 处理状态转换
                second_station_states = {k: obs[k] for k in obs if len(obs[k]) == 2}
                for k, states in second_station_states.items():
                    if states[0][1] != states[1][1]:  # 站点变化
                        episode_reward += agent_reward[k]
                    
                    # 更新观察并选择新动作
                    obs[k] = [states[1]]
                    state_features = states[1][2:]
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
                        action, _ = self.agent.agents[k].action(obs_tensor)  # 评估时不添加噪声
                        action_dict[k] = action.cpu().numpy().squeeze(0)
                
                # 环境步进
                obs, agent_reward, done = self.env.step(action_dict, render=render)
            
            eval_rewards.append(episode_reward)
            # print(f"Eval Episode {eval_ep + 1}/{num_eval_episodes}: Reward = {episode_reward:.2f}")
        
        mean_reward = np.mean(eval_rewards)
        reward_std = np.std(eval_rewards)
        
        return mean_reward, reward_std