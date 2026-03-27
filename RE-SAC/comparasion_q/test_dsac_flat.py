import pickle
import pandas as pd
with open('offline_dataset_full/data_dsac_v1_495.pkl', 'rb') as f:
    data = pickle.load(f)

events = data.get('data', data.get('events', []))
df = pd.DataFrame(events)
preds = df[df['event']=='predict']
print("Number of predicts:", len(preds))
import numpy as np
q_means = preds['q_vals'].apply(np.mean)
print("Q Mean Stats:")
print(q_means.describe())
