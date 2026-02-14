
import os
import sys
import pandas as pd
from env.sim import env_bus

def check_env():
    path = os.getcwd() + '/env'
    print(f"Loading environment from {path}")
    env = env_bus(path, debug=False)
    
    print(f"Total Timetable Size: {len(env.timetable_set)}")
    print(f"Loaded Timetables (Trips): {len(env.timetables)}")
    
    # Check effective_trip_num usage
    print(f"Effective Trip Num attribute: {env.effective_trip_num}")
    
    # Check if timetable matches full set
    if len(env.timetables) == len(env.timetable_set):
        print("SUCCESS: Full timetable loaded.")
    else:
        print(f"WARNING: Timetable truncated. {len(env.timetables)} vs {len(env.timetable_set)}")

if __name__ == "__main__":
    check_env()
