
import os
import sys
import numpy as np

# Adjust path to import correct envs
# Run from root, but simulate q_comparison import path
sys.path.append(os.getcwd())
# Also append q_comparasion so env import works if needed, but q_comparison_main uses 'from env.sim' where local folder is env.
# So we run this script FROM q_comparasion folder? No, let's run from root but adjust sys.path.

def run_env_diagnosis():
    print(f"\n--- Diagnosing q_comparasion/env ---")
    try:
        # We want to load 'env.sim' from 'q_comparasion' folder.
        # We insert 'q_comparasion' to sys.path[0]
        q_comp_path = os.path.join(os.getcwd(), 'q_comparasion')
        sys.path.insert(0, q_comp_path)
        
        from env.sim import env_bus
        path = os.path.join(q_comp_path, 'env')
        
        print(f"Loading env from {path}")
        env = env_bus(path, debug=False)
        env.reset()
        
        found_neighbors = 0
        missing_neighbors = 0
        total_steps = 0
        
        actions = {key: 15. for key in range(env.max_agent_num)}
        
        # Run for 2000 steps
        for _ in range(2000):
            state, reward, done = env.step(actions)
            total_steps += 1
            if done: break
            
            # Introspect buses
            for bus in env.bus_all:
                if bus.on_route and bus.trip_id > 2: # Check only if enough trips launched
                    fwd = list(filter(lambda x: bus.trip_id - 2 in x.trip_id_list, env.bus_all))
                    if fwd:
                        found_neighbors += 1
                    else:
                        missing_neighbors += 1
                        
        print(f"Steps: {total_steps}")
        print(f"Found Neighbors: {found_neighbors}")
        print(f"Missing Neighbors: {missing_neighbors}")
        if found_neighbors + missing_neighbors > 0:
             print(f"Success Rate: {found_neighbors / (found_neighbors + missing_neighbors):.2%}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_env_diagnosis()
