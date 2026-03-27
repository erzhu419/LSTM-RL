import pickle
import numpy as np

with open('offline_dataset_full/data_sac_499.pkl', 'rb') as f:
    pkg = pickle.load(f)

events = pkg['data']
preds = [e for e in events if e['event'] == 'predict']
rewards = [e for e in events if e['event'] == 'reward']

q_vals = [e['q_vals'] for e in preds[:10]]
print("First 10 Q-values predicted SAC:", q_vals)

