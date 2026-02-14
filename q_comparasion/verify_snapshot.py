
import os
import sys
import numpy as np
import copy
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.getcwd())

from env.sim import env_bus

def verify_snapshot_determinism():
    logger.info("Starting snapshot verification...")
    
    # Initialize environment
    path = os.getcwd() + '/env'
    env = env_bus(path, debug=False, route_sigma=1.5)
    
    # Seed numpy for reproducibility
    np.random.seed(42)
    
    # Reset env
    env.reset()
    
    # Run for T1 steps (e.g., 500 steps)
    T1 = 500
    actions = {key: 20. for key in range(env.max_agent_num)} # Constant action for determinism check
    
    logger.info(f"Running for {T1} steps...")
    for _ in range(T1):
        if env.done:
            env.reset()
        env.step(actions)
        
    # Take Snapshot at T1
    logger.info("Taking snapshot at T1...")
    snapshot = env.get_snapshot()
    time_t1 = env.current_time
    logger.info(f"Current time at T1: {time_t1}")
    
    # Run for another Delta steps (e.g., 100 steps)
    Delta = 100
    logger.info(f"Running for additional {Delta} steps (Run 1)...")
    
    history_1 = []
    
    for _ in range(Delta):
        state, reward, done = env.step(actions)
        # Store some signature of the state to compare
        # Since state is a dict of lists, we can flatten it or hash it.
        # Let's simple store the reward dict and current time
        history_1.append((env.current_time, copy.deepcopy(reward)))
        if done:
            break
            
    final_time_1 = env.current_time
    logger.info(f"Run 1 finished at time: {final_time_1}")
    
    # --- RESET TO SNAPSHOT ---
    logger.info("Resetting to T1 snapshot...")
    # Important: Reset random seed to the state it would have if we continued? 
    # Or just rely on the fact that if we use constant actions and no internal randomness is used *during* step (except route_sigma which is used in update headers?), it should match.
    # Note: env_bus uses route object which might have randomness if sigma > 0.
    # Route sigma is used in Route.__init__.
    # But does step() use randomness?
    # bus.drive uses randomness if passengers arrival uses random.
    # Stations use random passenger arrival?
    # station.station_update -> set.passenger_arrival_rate?
    # Let's assume we need to be careful with randomness.
    
    # Ideally, snapshot should capture random state too, but we didn't save it.
    # For this verification, let's re-seed right after reset to ensure same sequence of random numbers is generated IF the sequence matters.
    # BUT, if we re-seed to 42, we are repeating the *start* of the episode randomness, not the continuation.
    # A robust snapshot should verify that *given the same random seed sequence from that point*, it behaves same.
    # So we should save the RNG state or key.
    
    # However, let's first test if it works with fixed actions and explicit re-seeding to a KNOWN state before T2.
    # Actually, to verify snapshot *restoration*, we should ensure the environment *state* is restored.
    # If the environment has internal randomness (e.g. passenger arrival), it will diverge if RNG state is not restored.
    # Our snapshot does NOT store np.random.get_state().
    # Let's add it manually in this script for now to isolate 'state' vs 'rng' issues.
    
    rng_state = np.random.get_state()
    py_rng_state = random.getstate()
    
    env.reset(snapshot)
    
    # Restore RNG state
    np.random.set_state(rng_state)
    random.setstate(py_rng_state)
    
    # Run for Delta steps (Run 2)
    logger.info(f"Running for additional {Delta} steps (Run 2)...")
    history_2 = []
    
    for _ in range(Delta):
        state, reward, done = env.step(actions)
        history_2.append((env.current_time, copy.deepcopy(reward)))
        if done:
            break
            
    final_time_2 = env.current_time
    logger.info(f"Run 2 finished at time: {final_time_2}")
    
    # Compare
    logger.info("Comparing histories...")
    match = True
    for i, (h1, h2) in enumerate(zip(history_1, history_2)):
        t1, r1 = h1
        t2, r2 = h2
        if t1 != t2:
            logger.error(f"Time mismatch at step {i}: {t1} vs {t2}")
            match = False
            break
            
        # Compare dictionaries
        # keys might be different order, but content same
        for k in r1:
            if k not in r2 or r1[k] != r2[k]:
                # Floating point comparison
                if not np.isclose(r1.get(k, 0), r2.get(k, 0)):
                    logger.error(f"Reward mismatch at step {i}, bus {k}: {r1.get(k)} vs {r2.get(k)}")
                    match = False
                    
    if match:
        logger.info("SUCCESS: History matches perfectly after snapshot reset!")
    else:
        logger.error("FAILURE: History mismatch.")

if __name__ == "__main__":
    verify_snapshot_determinism()
