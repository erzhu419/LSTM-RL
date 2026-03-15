import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ensemble_paper.q_accuracy_analysis.model_loader import ModelLoader
from env.sim import env_bus

def probe_model(path, name):
    print(f"\nProbing {name} at {path}")
    try:
        sd = torch.load(path, map_location='cpu', weights_only=True)
        for k, v in sd.items():
            if 'linear' in k or 'fc' in k or 'weight' in k:
                print(f"  {k}: {v.shape}")
    except Exception as e:
        print(f"  Error loading {path}: {e}")

# BAC path
bac_path = "/home/erzhu419/mine_code/LSTM-RL/ensemble_paper/logs_bac/model/bac_v1_lambda0p5_q0p7/checkpoint_episode_499_q"
probe_model(bac_path, "BAC Q")

# DSAC path
dsac_path = "/home/erzhu419/mine_code/LSTM-RL/model/dsac_bus_sigma1p5_embed-full_exp_dsac_amax2p0_20251025_221923/dsac_bus_episode_495_z1"
probe_model(dsac_path, "DSAC Z1")
