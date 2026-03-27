import torch
import numpy as np
import os
import sys
from env.sim import env_bus
from model_loader import ModelLoader

# Path to SAC model checkpoint (prefix)
SAC_MODEL_PATH = "/home/erzhu419/mine_code/LSTM-RL/model/sac_v2_bus_sigma1p5_embed-full_wreg0p0_exp_sac_amax2p0_20251025_221923/sac_v2_bus_episode_450"
# Path to Ensemble model checkpoint (prefix)
ENSEMBLE_MODEL_PATH = "/home/erzhu419/mine_code/LSTM-RL/RE-SAC/ensemble_10/model/checkpoint_episode_499"

def probe_scale():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize environment
    env_path = "/home/erzhu419/mine_code/LSTM-RL/env"
    env = env_bus(env_path)
    env.reset() # CRITICAL FIX
    print(f"Environment state_dim: {env.state_dim}")
    
    # 1. Probe SAC Model (using hidden_dim=64 from args.json)
    print("\n--- Probing SAC Model ---")
    sac_loader = ModelLoader(env, hidden_dim=64, action_range=15.0, device=device)
    try:
        sac_model = sac_loader.load_sac(SAC_MODEL_PATH)
        q1 = sac_model['q1']
        
        # Test input
        state_dict, reward_dict, _ = env.initialize_state()
        
        # Find first non-empty bus state
        raw_state = None
        for bus_id in state_dict:
            if len(state_dict[bus_id]) > 0:
                raw_state = np.array(state_dict[bus_id][0])
                break
        
        if raw_state is None:
            print("Error: No bus states found after initialization")
            return

        state_input = sac_model['state_norm'](raw_state, update=False)
        state_tensor = torch.FloatTensor(state_input).unsqueeze(0).to(device)
        action_tensor = torch.FloatTensor([[5.0]]).to(device) # Random action
        
        with torch.no_grad():
            q_val = q1(state_tensor, action_tensor).item()
        print(f"SAC Q1 output for state (bus_id={raw_state[0]}): {q_val}")
    except Exception as e:
        print(f"Error loading/probing SAC: {e}")
        import traceback
        traceback.print_exc()

    # 2. Probe Ensemble Model (using hidden_dim=256)
    print("\n--- Probing Ensemble Model ---")
    ens_loader = ModelLoader(env, hidden_dim=256, action_range=15.0, device=device)
    try:
        ens_model = ens_loader.load_ensemble(ENSEMBLE_MODEL_PATH, ensemble_size=10)
        q_net = ens_model['q_net']
        
        # Test input
        state_dict, reward_dict, _ = env.initialize_state()
        raw_state = None
        for bus_id in state_dict:
            if len(state_dict[bus_id]) > 0:
                raw_state = np.array(state_dict[bus_id][0])
                break
        
        state_input = ens_model['state_norm'](raw_state, update=False)
        state_tensor = torch.FloatTensor(state_input).unsqueeze(0).to(device)
        action_tensor = torch.FloatTensor([[5.0]]).to(device)
        
        with torch.no_grad():
            q_vals = q_net(state_tensor, action_tensor)
            min_q = torch.min(q_vals).item()
            mean_q = torch.mean(q_vals).item()
        print(f"Ensemble Q min: {min_q}, mean: {mean_q}")
    except Exception as e:
        print(f"Error loading/probing Ensemble: {e}")

if __name__ == "__main__":
    probe_scale()
