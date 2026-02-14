
import numpy as np
import os
import sys

# Adjust path to import correct envs
sys.path.append(os.getcwd())
# Ensure we use the q_comparasion env
q_comp_path = os.path.join(os.getcwd(), 'q_comparasion')
sys.path.insert(0, q_comp_path)

from env.sim import env_bus

def verify_rng_restore():
    print("--- Verifying RNG State Preservation ---")
    
    # 1. Initialize and Seed
    np.random.seed(42)
    env = env_bus(os.path.join(q_comp_path, 'env'), debug=False)
    env.reset()
    
    # 2. Generate a reference number (Pre-Snapshot)
    val_pre = np.random.rand()
    print(f"Pre-Snapshot Random Value: {val_pre}")
    
    # 3. Take Snapshot
    print("Taking Snapshot (Should save RNG state)...")
    snapshot = env.get_snapshot()
    
    # 4. Generate numbers (Simulate Rollout consumption)
    print("Simulating Rollout (consuming 5 random numbers)...")
    rollout_vals = [np.random.rand() for _ in range(5)]
    print(f"Rollout Values: {rollout_vals}")
    
    # Capture the value that would occur if we continued WITHOUT restore
    val_post_rollout = np.random.rand()
    print(f"Value if we continued directly: {val_post_rollout}")
    
    # 5. Restore Snapshot
    print("Restoring Snapshot...")
    env.reset(snapshot)
    
    # 6. Generate number (Should match what would have happened right after snapshot)
    # Note: If we saved state at T, restoring to T means the NEXT number generated 
    # should be the *first* number of the "Rollout Values" sequence?
    # Wait, no. If we restore to T, we are back at T.
    # So the next number generated should be `rollout_vals[0]`.
    # Yes, because `rollout_vals[0]` was the first thing generated AFTER snapshot.
    
    val_restored_1 = np.random.rand()
    print(f"Restored Value 1: {val_restored_1}")
    
    if val_restored_1 == rollout_vals[0]:
        print("SUCCESS: Restored Value matches the first value of the original future!")
        print("This proves the simulation will replay the EXACT same random sequence (1:1 control).")
    else:
        print(f"FAILURE: Restored Value {val_restored_1} != Expected {rollout_vals[0]}")
        
    # 7. Check Side Effect Free (Pasivity)
    # Theoretically, if we restore, do our work, and then... wait.
    # The question is: does `load_snapshot` inadvertently advance the state? No.
    # Does this allow us to return to the "main loop"?
    # If the main loop continues from the snapshot point, it will see `rollout_vals[0]`.
    # Correct.
    
if __name__ == "__main__":
    verify_rng_restore()
