
import os
import sys
import copy
import numpy as np

# Adjust path to import correct envs
sys.path.append(os.getcwd())
# Ensure we use the q_comparasion env
q_comp_path = os.path.join(os.getcwd(), 'q_comparasion')
sys.path.insert(0, q_comp_path)

from env.sim import env_bus

def check_snapshot_integrity():
    print("--- Checking Snapshot Integrity ---")
    env = env_bus(os.path.join(q_comp_path, 'env'), debug=False)
    env.reset()
    
    # helper to find station by id
    def get_station_by_id(stations, s_id):
        for s in stations:
            if s.station_id == s_id: return s
        return None

    # 1. Run a few steps to ensure buses are launched and bound to stations
    for _ in range(100):
        env.step({key: 15. for key in range(env.max_agent_num)})
        
    bus0 = env.bus_all[0]
    if not bus0.on_route:
        print("Bus 0 not on route yet, running more steps...")
        for _ in range(500):
             env.step({key: 15. for key in range(env.max_agent_num)})
        bus0 = env.bus_all[0]

    current_station_ref = bus0.last_station
    env_station_ref = get_station_by_id(env.stations, current_station_ref.station_id)
    
    print(f"Before Snapshot: Bus Station ID({id(current_station_ref)}) == Env Station ID({id(env_station_ref)})? {id(current_station_ref) == id(env_station_ref)}")
    
    # 2. Take Snapshot
    print("Taking Snapshot...")
    snapshot = env.get_snapshot()
    
    # 3. Restore Snapshot
    print("Restoring Snapshot...")
    env.reset(snapshot)
    
    # 4. Check Integrity
    bus0_restored = env.bus_all[0]
    current_station_ref_r = bus0_restored.last_station
    env_station_ref_r = get_station_by_id(env.stations, current_station_ref_r.station_id)
    
    print(f"After Restore: Bus Station ID({id(current_station_ref_r)}) == Env Station ID({id(env_station_ref_r)})? {id(current_station_ref_r) == id(env_station_ref_r)}")
    
    if id(current_station_ref_r) != id(env_station_ref_r):
        print("CRITICAL ISSUE FOUND: Buses are disconnected from Environment Stations!")
        print("The Simulation loop updates 'Env Stations', but Buses interact with 'Ghost Stations'.")
        print("Passengers generated on Env Stations will NEVER be picked up by Buses.")
    else:
        print("Snapshot integrity seems OK.")

if __name__ == "__main__":
    check_snapshot_integrity()
