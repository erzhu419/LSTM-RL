import numpy as np
import plotly.graph_objs as go

# Headway范围
forward_headway = np.linspace(100, 600, 100)
backward_headway = np.linspace(100, 600, 100)
X, Y = np.meshgrid(forward_headway, backward_headway)

# 简化版的 headway 奖励函数
def headway_reward(headway):
    return -abs(headway - 360)

# 计算总体 reward
def calculate_reward(fh, bh):
    forward_reward = headway_reward(fh)
    backward_reward = headway_reward(bh)

    if forward_reward is not None and backward_reward is not None:
        weight = abs(fh - 360) / (abs(fh - 360) + abs(bh - 360) + 1e-6)
        similarity_bonus = -abs(fh - bh) * 0.5
        reward = forward_reward * weight + backward_reward * (1 - weight) + similarity_bonus
    elif forward_reward is not None:
        reward = forward_reward
    elif backward_reward is not None:
        reward = backward_reward
    else:
        reward = -50

    if abs(fh - 360) > 180 or abs(bh - 360) > 180:
        reward -= 20

    return reward

# 计算 Z 值
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = calculate_reward(X[i, j], Y[i, j])

# 展平
x_vals = X.flatten()
y_vals = Y.flatten()
z_vals = Z.flatten()

# 画图
trace = go.Scatter3d(
    x=x_vals, y=y_vals, z=z_vals,
    mode='markers',
    marker=dict(size=3, color=z_vals, colorscale='Viridis', opacity=0.8)
)

layout = go.Layout(
    title='Reward Landscape with Simplified Headway Function',
    scene=dict(
        xaxis_title='Forward Headway (s)',
        yaxis_title='Backward Headway (s)',
        zaxis_title='Reward'
    )
)

fig = go.Figure(data=[trace], layout=layout)
fig.show()
