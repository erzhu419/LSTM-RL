import sys
import os
import random
sys.path.append(os.getcwd())
from q_accuracy_analysis.env.sim import env_bus

env_path = os.path.join(os.getcwd(), 'q_accuracy_analysis', 'env')
print(f"Initializing Environment at {env_path}...")
env = env_bus(env_path, debug=True)
env.reset()
env.current_time = 21600 # Start at 6:00 AM
steps = 0
prev_reward_sum = 0
sticky_count = 0
reward_occured = False

from collections import defaultdict
print("Running simulation...")
while steps < 50: # Extended run for movement check
    actions = defaultdict(lambda: 15.0)
    for bus in env.bus_all:
        actions[bus.bus_id] = 15.0
    state, reward, done = env.step(actions, debug=True)
    
    current_reward_sum = sum(reward.values())
    
    if current_reward_sum != 0:
        reward_occured = True
        # print(f"Step {steps}: Total Reward {current_reward_sum}")
        if current_reward_sum == prev_reward_sum:
            print(f"Step {steps}: Sticky reward detected! Value: {current_reward_sum}")
            sticky_count += 1
            if sticky_count > 5:
                break
    
    prev_reward_sum = current_reward_sum
    steps += 1
    if done:
        break

if sticky_count > 0:
    print(f"CONFIRMED: Sticky reward detected {sticky_count} times.")
elif not reward_occured:
    print("WARNING: No rewards generated at all (simulation too short?).")
else:
    print("CLEAN: No stickiness detected.")
