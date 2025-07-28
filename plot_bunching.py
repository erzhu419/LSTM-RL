import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取轨迹数据
control_df = pd.read_csv('/home/erzhu419/mine_code/LSTM-RL/pic/exp 0, bus_trajectories.csv')
uncontrol_df = pd.read_csv('/home/erzhu419/mine_code/LSTM-RL/env/pic/exp 1, bus_trajectories.csv')

# 定义与visualize.py相同的颜色映射
cnames = {
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgreen':            '#006400',
    'darkmagenta':          '#8B008B',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darkslateblue':        '#483D8B',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'forestgreen':          '#228B22',
    'gold':                 '#FFD700',
    'green':                '#008000',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'limegreen':            '#32CD32',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumvioletred':      '#C71585',
    'navy':                 '#000080',
    'olive':                '#808000',
    'orangered':            '#FF4500',
    'purple':               '#800080',
    'red':                  '#FF0000',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'seagreen':             '#2E8B57',
    'sienna':               '#A0522D',
    'slateblue':            '#6A5ACD',
    'steelblue':            '#4682B4',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'yellow':               '#FFFF00'
}

# 创建两个图表进行对比
def plot_trajectories_comparison(control_df, uncontrol_df, save_path=None):
    """
    在一张图上对比绘制控制策略和无控制策略的轨迹，并高亮bunching点
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(96, 48), dpi=120)
    
    # 与visualize.py一样，使用有序颜色列表
    color_list = list(cnames.keys())
    
    # 绘制控制策略轨迹（上图）
    bus_ids_control = control_df['bus_id'].unique()
    min_time_control = control_df['time'].min()
    max_time_control = control_df['time'].max()
    
    for i, bus_id in enumerate(bus_ids_control):
        bus_data = control_df[control_df['bus_id'] == bus_id]
        bus_data = bus_data.sort_values('time')
        
        x = bus_data['time'].values
        y = bus_data['position'].values
        direction = bus_data['direction'].iloc[0]
        
        color_idx = bus_id % len(color_list)
        color = cnames[color_list[color_idx]]
        
        # 绘制正常轨迹
        ax1.plot(x, y, label=f'Bus {bus_id}', color=color)
        ax1.scatter(x, y, s=5, color=color)
        
        # 高亮bunching点
        if 'is_bunching' in bus_data.columns:
            bunching_points = bus_data[bus_data['is_bunching'] == True]
            if not bunching_points.empty:
                ax1.scatter(bunching_points['time'], bunching_points['position'], 
                           s=160, color='red', alpha=1.0, edgecolors='red', zorder=5)
    
    # 绘制控制策略的站点参考线
    x1 = np.linspace(min_time_control, max_time_control, num=500)
    station_names = ['Terminal up'] + [f'X{i:02d}' for i in range(1, 21)] + ['Terminal down']
    for j in range(len(station_names)):
        y1 = [j * 500] * len(x1)
        ax1.plot(x1, y1, color="red", linewidth=0.3, linestyle='-')
    
    # 添加bunching标记到图例
    ax1.scatter([], [], s=300, color='red', alpha=1.0, edgecolors='red', label='Bunching Events')
    
    # 设置控制策略图的属性
    # 设置时间轴刻度 - 从6:00开始，每小时一个刻度
    time_range = max_time_control - min_time_control
    hour_interval = 3600  # 1小时 = 3600秒
    num_hours = int(time_range / hour_interval) + 1
    time_ticks = np.arange(min_time_control, max_time_control + hour_interval, hour_interval)
    time_labels = [f'{6 + i}:00' for i in range(len(time_ticks))]
    
    ax1.set_xticks(time_ticks)
    ax1.set_xticklabels(time_labels, fontsize=70)
    ax1.set_yticks([j * 500 for j in range(len(station_names))])
    ax1.set_yticklabels(station_names, fontsize=70)
    ax1.legend(fontsize=40, loc='upper right')
    # 不设置xlabel，保持x轴无标签
    # ax1.set_xlabel('time', fontsize=40)
    # ax1.set_ylabel('station', fontsize=40)
    # Bus Trajectories with SAC Control
    ax1.set_title('(a)', fontsize=70, loc='center', fontweight='bold')
    ax1.set_xlim(min_time_control, max_time_control)
    
    # 绘制无控制策略轨迹（下图）
    bus_ids_uncontrol = uncontrol_df['bus_id'].unique()
    min_time_uncontrol = uncontrol_df['time'].min()
    max_time_uncontrol = uncontrol_df['time'].max()
    
    for i, bus_id in enumerate(bus_ids_uncontrol):
        bus_data = uncontrol_df[uncontrol_df['bus_id'] == bus_id]
        bus_data = bus_data.sort_values('time')
        
        x = bus_data['time'].values
        y = bus_data['position'].values
        direction = bus_data['direction'].iloc[0]
        
        color_idx = bus_id % len(color_list)
        color = cnames[color_list[color_idx]]
        
        # 绘制正常轨迹
        ax2.plot(x, y, label=f'Bus {bus_id}', color=color)
        ax2.scatter(x, y, s=5, color=color)
        
        # 高亮bunching点
        if 'is_bunching' in bus_data.columns:
            bunching_points = bus_data[bus_data['is_bunching'] == True]
            if not bunching_points.empty:
                ax2.scatter(bunching_points['time'], bunching_points['position'], 
                           s=200, color='red', alpha=1.0, edgecolors='red', zorder=5)
    
    # 绘制无控制策略的站点参考线
    x2 = np.linspace(min_time_uncontrol, max_time_uncontrol, num=500)
    for j in range(len(station_names)):
        y2 = [j * 500] * len(x2)
        ax2.plot(x2, y2, color="red", linewidth=0.3, linestyle='-')
    
    # 添加bunching标记到图例
    ax2.scatter([], [], s=160, color='red', alpha=1.0, edgecolors='red', label='Bunching Events')
    
    # 设置无控制策略图的属性
    # 设置时间轴刻度 - 从6:00开始，每小时一个刻度
    time_range_uncontrol = max_time_uncontrol - min_time_uncontrol
    hour_interval = 3600  # 1小时 = 3600秒
    num_hours_uncontrol = int(time_range_uncontrol / hour_interval) + 1
    time_ticks_uncontrol = np.arange(min_time_uncontrol, max_time_uncontrol + hour_interval, hour_interval)
    time_labels_uncontrol = [f'{6 + i}:00' for i in range(len(time_ticks_uncontrol))]
    
    ax2.set_xticks(time_ticks_uncontrol)
    ax2.set_xticklabels(time_labels_uncontrol, fontsize=70)
    ax2.set_yticks([j * 500 for j in range(len(station_names))])
    ax2.set_yticklabels(station_names, fontsize=70)
    ax2.legend(fontsize=40, loc='upper right')
    # ax2.set_xlabel('time', fontsize=40)
    # ax2.set_ylabel('station', fontsize=40)
    # Bus Trajectories without Control
    ax2.set_title('(b)', fontsize=70, loc='center', fontweight='bold')
    ax2.set_xlim(min_time_uncontrol, max_time_uncontrol)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path)
        print(f"对比图已保存至: {save_path}")
    
    # plt.show()
    

def calculate_speed(df):
    """
    为数据框添加速度列
    
    参数：
    - df: 包含 bus_id, time, position, trip_id 的 DataFrame
    
    返回：
    - df: 添加了 speed 列的 DataFrame
    """
    df = df.copy()
    df = df.sort_values(['trip_id', 'time'])
    df['speed'] = 0.0
    
    # 终点站位置
    terminal_up_pos = 0
    terminal_down_pos = 21 * 500  # 10500
    
    for trip_id in df['trip_id'].unique():
        trip_data = df[df['trip_id'] == trip_id].copy()
        trip_indices = trip_data.index
        
        # 计算位置差（因为time step是1秒，所以位置差就是速度）
        positions = trip_data['position'].values
        speeds = np.zeros(len(positions))
        
        for i in range(1, len(positions)):
            pos_diff = abs(positions[i] - positions[i-1])
            # 如果在终点站，速度为0
            if positions[i] in [terminal_up_pos, terminal_down_pos] or positions[i-1] in [terminal_up_pos, terminal_down_pos]:
                speeds[i] = 0.0
            else:
                speeds[i] = pos_diff
        
        # 第一个点的速度设为0
        speeds[0] = 0.0
        
        # 更新原始数据框
        df.loc[trip_indices, 'speed'] = speeds
    
    return df


def find_bunching_trips(df, time_threshold=30.0):
    """
    找出轨迹点在每个时间步中时距过近（可能发生bunching）的上下行trip组合。

    参数：
    - df: 包含 bus_id, time, position, trip_id, direction, speed 的 DataFrame
    - time_threshold: 当两辆车的时距小于该值时认为存在 bunching（单位：秒）

    返回：
    - potential_bunching: 记录发生bunching的时间点、trip对、以及时距
    """
    df = df.copy()
    df = df.sort_values(by=['trip_id', 'time'])

    # 保存结果
    bunching_records = []

    # 只检查同方向（同为奇或偶）且trip_id相邻的组合
    unique_trips = sorted(df['trip_id'].unique())
    for i in range(len(unique_trips) - 2):
        trip_a = unique_trips[i]
        trip_b = unique_trips[i + 2]

        # 跳过不同行方向的组合
        if trip_a % 2 != trip_b % 2:
            continue

        # 提取两条轨迹
        df_a = df[df['trip_id'] == trip_a][['time', 'position', 'speed']].rename(columns={'position': 'pos_a', 'speed': 'speed_a'})
        df_b = df[df['trip_id'] == trip_b][['time', 'position', 'speed']].rename(columns={'position': 'pos_b', 'speed': 'speed_b'})

        # 合并同一时间点的位置信息
        merged = pd.merge(df_a, df_b, on='time')
        merged['distance'] = np.abs(merged['pos_a'] - merged['pos_b'])
        
        # 计算时距：使用两车平均速度来计算时距
        merged['avg_speed'] = (merged['speed_a'] + merged['speed_b']) / 2
        # 避免除以0，当速度为0时设为很大的时距
        merged['time_headway'] = np.where(merged['avg_speed'] > 0, 
                                         merged['distance'] / merged['avg_speed'], 
                                         999999)  # 很大的数值表示无穷大时距

        # 筛选出时距过近的时间点，但排除终点站
        close_points = merged[merged['time_headway'] < time_threshold].copy()
        if not close_points.empty:
            # 过滤掉Terminal up (position=0) 和 Terminal down (position=10500) 的bunching
            # Terminal up 对应 position = 0，Terminal down 对应 position = 21*500 = 10500
            terminal_up_pos = 0
            terminal_down_pos = 21 * 500  # 10500
            
            # 过滤掉在终点站的bunching事件
            close_points = close_points[
                (close_points['pos_a'] != terminal_up_pos) & 
                (close_points['pos_a'] != terminal_down_pos) &
                (close_points['pos_b'] != terminal_up_pos) & 
                (close_points['pos_b'] != terminal_down_pos)
            ]
            
            if not close_points.empty:
                close_points['trip_a'] = trip_a
                close_points['trip_b'] = trip_b
                bunching_records.append(close_points[['time', 'trip_a', 'trip_b', 'pos_a', 'pos_b', 'distance', 'time_headway']])

    # 合并所有结果
    if bunching_records:
        potential_bunching = pd.concat(bunching_records, ignore_index=True)
    else:
        potential_bunching = pd.DataFrame(columns=['time', 'trip_a', 'trip_b', 'pos_a', 'pos_b', 'distance', 'time_headway'])

    return potential_bunching

# 首先为数据添加速度列
print("正在计算速度...")
uncontrol_df = calculate_speed(uncontrol_df)
control_df = calculate_speed(control_df)

# 应用函数，使用时距阈值（秒）
# 处理无控制策略数据
bunching_result_uncontrol = find_bunching_trips(uncontrol_df, time_threshold=45.0)

# 标注uncontrol_df中每一行是否为bunching状态
uncontrol_df = uncontrol_df.copy()
uncontrol_df['is_bunching'] = False

# 如果有bunching结果，需要标记相应的行
if not bunching_result_uncontrol.empty:
    # 处理uncontrol_df的bunching信息
    for _, row in bunching_result_uncontrol.iterrows():
        # 找到trip_a在原始数据中对应的行
        trip_a_mask = (uncontrol_df['time'] == row['time']) & (uncontrol_df['trip_id'] == row['trip_a'])
        trip_a_indices = uncontrol_df[trip_a_mask].index
        
        # 找到trip_b在原始数据中对应的行
        trip_b_mask = (uncontrol_df['time'] == row['time']) & (uncontrol_df['trip_id'] == row['trip_b'])
        trip_b_indices = uncontrol_df[trip_b_mask].index
        
        # 将这些行标记为bunching
        uncontrol_df.loc[trip_a_indices, 'is_bunching'] = True
        uncontrol_df.loc[trip_b_indices, 'is_bunching'] = True

# 处理控制策略数据
bunching_result_control = find_bunching_trips(control_df, time_threshold=30.0)

# 标注control_df中每一行是否为bunching状态
control_df = control_df.copy()
control_df['is_bunching'] = False

# 如果有bunching结果，需要标记相应的行
if not bunching_result_control.empty:
    # 处理control_df的bunching信息
    for _, row in bunching_result_control.iterrows():
        # 找到trip_a在原始数据中对应的行
        trip_a_mask = (control_df['time'] == row['time']) & (control_df['trip_id'] == row['trip_a'])
        trip_a_indices = control_df[trip_a_mask].index
        
        # 找到trip_b在原始数据中对应的行
        trip_b_mask = (control_df['time'] == row['time']) & (control_df['trip_id'] == row['trip_b'])
        trip_b_indices = control_df[trip_b_mask].index
        
        # 将这些行标记为bunching
        control_df.loc[trip_a_indices, 'is_bunching'] = True
        control_df.loc[trip_b_indices, 'is_bunching'] = True

# 统计bunching状态
bunching_count_uncontrol = uncontrol_df['is_bunching'].sum()
total_count_uncontrol = len(uncontrol_df)
bunching_count_control = control_df['is_bunching'].sum()
total_count_control = len(control_df)

print(f"无控制策略 - 总数据点数: {total_count_uncontrol}")
print(f"无控制策略 - 标记为bunching的数据点数: {bunching_count_uncontrol}")
print(f"无控制策略 - bunching比例: {bunching_count_uncontrol/total_count_uncontrol*100:.2f}%")

print(f"控制策略 - 总数据点数: {total_count_control}")
print(f"控制策略 - 标记为bunching的数据点数: {bunching_count_control}")
print(f"控制策略 - bunching比例: {bunching_count_control/total_count_control*100:.2f}%")


# 创建保存目录
os.makedirs('comparison_plots', exist_ok=True)

# 绘制对比图（控制策略和无控制策略在一张图上）
plot_trajectories_comparison(control_df, uncontrol_df, 
                           'comparison_plots/trajectories_comparison_with_bunching.jpg')

# 保存带有bunching标记的数据
control_df.to_csv('comparison_plots/control_with_bunching_status.csv', index=False)
uncontrol_df.to_csv('comparison_plots/uncontrolled_with_bunching_status.csv', index=False)
print("带有bunching标记的控制策略数据已保存至: comparison_plots/control_with_bunching_status.csv")
print("带有bunching标记的无控制策略数据已保存至: comparison_plots/uncontrolled_with_bunching_status.csv")