import numpy as np
# read .npy file
data = np.load('logs/ddpg_bus_sigma1p5_embed-full_plot500/rewards.npy', allow_pickle=True)
print(data)