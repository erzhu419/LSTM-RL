
import os
import sys
import numpy as np

# Adjust path to import correct envs
sys.path.append(os.getcwd())

def run_env_diagnosis(env_module_name, env_path_suffix):
    print(f"\n--- Diagnosing {env_module_name} ---")
    try:
        if env_module_name == 'env.sim':
             from env.sim import env_bus
             path = os.path.join(os.getcwd(), 'env') # q_comparasion/env usually or root env?
             # q_comparison_main.py runs from 'q_comparasion' dir and imports env.sim
             # So it uses 'q_comparasion/env'
             if 'q_comparasion' not in os.getcwd():
                 path = os.path.join(os.getcwd(), 'q_comparasion/env')
        else:
             from env_original.sim import env_bus
             path = os.path.join(os.getcwd(), 'env_original')

        print(f"Loading env from {path}")
        env = env_bus(path, debug=False)
        env.reset()
        
        found_neighbors = 0
        missing_neighbors = 0
        total_steps = 0
        
        actions = {key: 15. for key in range(env.max_agent_num)}
        
        # Run for 2000 steps (enough for some buses to launch and meet)
        for _ in range(2000):
            state, reward, done = env.step(actions)
            total_steps += 1
            if done: break
            
            # Introspect buses
            for bus in env.bus_all:
                if bus.on_route and bus.trip_id > 2: # Check only if enough trips launched
                    # Check manually if logic finds neighbor
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
    # We need to run this from root
    # Test env_original (Baseline)
    run_env_diagnosis('env_original.sim', 'env_original')
    
    # Test q_comparison env (env.sim)
    # We must be careful with imports. Python caches imports.
    # So we might need to run this as two separate processes or reload.
    # For simplicity, let's just run env_original first, then I'll create another script or run twice with args.
