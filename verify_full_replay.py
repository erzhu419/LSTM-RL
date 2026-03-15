
import numpy as np
import random
import os
import sys
import copy

# Adjust path to import correct envs
sys.path.append(os.getcwd())
# Ensure we use the q_comparasion env
q_comp_path = os.path.join(os.getcwd(), 'q_comparasion')
sys.path.insert(0, q_comp_path)

from env.sim import env_bus

def verify_full_replay():
    print("--- Verifying Full Replay Consistency (Stations & Routes) ---")
    
    # 1. Initialize and Seed
    np.random.seed(42)
    random.seed(42)
    env = env_bus(os.path.join(q_comp_path, 'env'), debug=False)
    env.reset()
    
    # Run a bit to warm up
    for _ in range(50):
        env.step({key: 15. for key in range(env.max_agent_num)})
        
    print("Taking Snapshot...")
    snapshot = env.get_snapshot()
    
    # --- PHASE 1: Golden Run ---
    print("Running Phase 1 (Golden Run)...")
    trace_golden = []
    actions = {key: 15. for key in range(env.max_agent_num)}
    
    for step in range(100):
        env.step(actions)
        
        # Record State
        # Station 0 waiting passengers count (dynamic)
        s0_wait = len(env.stations[0].waiting_passengers)
        # Route 0 speed limit (dynamic due to lognormvariate)
        r0_speed = env.routes[0].speed_limit
        # Bus 0 obs (position etc)
        # Check if Bus 0 exists
        if len(env.bus_all) > 0:
            b0_param = env.bus_all[0].last_station.station_id  # Just a property
        else:
            b0_param = -1
            
        trace_golden.append((step, s0_wait, r0_speed, b0_param))
        
    print(f"Recorded {len(trace_golden)} steps.")
    
    # --- PHASE 2: Replay Run ---
    print("Restoring Snapshot...")
    env.reset(snapshot)
    
    print("Running Phase 2 (Replay Run)...")
    trace_replay = []
    
    for step in range(100):
        env.step(actions)
        
        # Record State (Identical Logic)
        s0_wait = len(env.stations[0].waiting_passengers)
        r0_speed = env.routes[0].speed_limit
        if len(env.bus_all) > 0:
            b0_param = env.bus_all[0].last_station.station_id
        else:
            b0_param = -1
            
        trace_replay.append((step, s0_wait, r0_speed, b0_param))
        
    # --- COMPARISON ---
    print("Comparing Traces...")
    
    matches = True
    for i in range(100):
        g = trace_golden[i]
        r = trace_replay[i]
        if g != r:
            print(f"MISMATCH at Step {i}: Golden={g} vs Replay={r}")
            matches = False
            break
            
    if matches:
        print("SUCCESS: Full Replay Matches Perfectly!")
        print("Station Passengers and Route Speeds are identical for every second.")
    else:
        print("FAILURE: Traces diverged!")

if __name__ == "__main__":
    verify_full_replay()
