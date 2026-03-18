import os
import pickle
import numpy as np
import glob
from tqdm import tqdm

DATA_DIR = "/home/erzhu419/mine_code/LSTM-RL/ensemble_paper/comparasion_q/offline_timeline_data_20eps"

# Log directories to save q_stds_episode.npy
SAC_OUT_DIR = "/home/erzhu419/mine_code/LSTM-RL/model/sac_v2_bus"
DSAC_OUT_DIR = "/home/erzhu419/mine_code/LSTM-RL/ensemble_paper/logs/dsac_bus_sigma1p5_embed-full_exp_dsac_amax2p0_20251025_221923"
BAC_OUT_DIR = "/home/erzhu419/mine_code/LSTM-RL/ensemble_paper/logs_bac/logs/bac_v1_lambda0p5_q0p7"

def extract_std(model_prefix, out_dir, total_episodes=500, checkpoint_step=1):
    print(f"Extracting std for {model_prefix}...")
    
    stds = np.zeros(total_episodes)
    valid_eps = []
    valid_stds = []
    
    for ep in tqdm(range(0, total_episodes, checkpoint_step)):
        file_path = os.path.join(DATA_DIR, f"data_{model_prefix}_{ep}.pkl")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'rb') as f:
            try:
                data_pkg = pickle.load(f)
                data_list = data_pkg['data']
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
        # Filter for predict events
        q_vals = [d['q_vals'] for d in data_list if d.get('event') == 'predict' and 'q_vals' in d]
        
        if len(q_vals) > 0:
            q_vals_array = np.array(q_vals)
            # Standard deviation across critics (axis=1), then mean across the batch (axis=0)
            avg_std = np.std(q_vals_array, axis=1).mean()
            
            # Since Variance of 2 critics is very small compared to 10 or 20, 
            # and DSAC outputs standard deviation explicitly for ALEATORIC,
            # wait, DSAC z1, z2 are QUANTILE FRACTIONS!!
            # If it's DSAC, q_vals_array has shape [Batch, 20] (20 quantiles)
            # We take the standard deviation across quantiles to get the predictive uncertainty!
            
            valid_eps.append(ep)
            valid_stds.append(avg_std)
            
    if not valid_eps:
        print(f"No valid data found for {model_prefix}")
        return
        
    # Interpolate missing values (e.g. for DSAC step=5)
    stds = np.interp(np.arange(total_episodes), valid_eps, valid_stds)
    
    out_path = os.path.join(out_dir, "q_stds_episode.npy")
    os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, stds)
    print(f"Saved {out_path} with shape {stds.shape}, Mean std: {stds.mean():.4f}")

if __name__ == "__main__":
    extract_std("sac", SAC_OUT_DIR, total_episodes=500, checkpoint_step=1)
    extract_std("dsac", DSAC_OUT_DIR, total_episodes=500, checkpoint_step=5)
    extract_std("bac", BAC_OUT_DIR, total_episodes=500, checkpoint_step=1)

