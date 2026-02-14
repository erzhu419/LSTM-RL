
import numpy as np
import os

path = "q_comparasion/q_comparasion/results/logs/route_sigma_1p5/replay_buffer_size_1000000/critic_actor_ratio_2/maximum_alpha_0.6/weight_reg_0.01"

try:
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        exit(1)
        
    q_real = np.load(os.path.join(path, "q_real_log.npy"))
    q_pred = np.load(os.path.join(path, "q_pred_log.npy"))

    print(f"Loaded {len(q_real)} samples from {path}")
    print(f"{'Index':>5} | {'Q_Pred':>10} | {'Q_Real':>10} | {'Diff':>10}")
    print("-" * 45)
    
    # Print first 20
    for i in range(min(20, len(q_real))):
        print(f"{i:5d} | {q_pred[i]:10.2f} | {q_real[i]:10.2f} | {q_pred[i]-q_real[i]:10.2f}")
        
    print("-" * 45)
    print(f"Mean Q_Pred: {np.mean(q_pred):.2f}")
    print(f"Mean Q_Real: {np.mean(q_real):.2f}")
    print(f"Mean Diff: {np.mean(q_pred - q_real):.2f}")
    print(f"MAE: {np.mean(np.abs(q_pred - q_real)):.2f}")
    
except Exception as e:
    print(f"Error: {e}")
