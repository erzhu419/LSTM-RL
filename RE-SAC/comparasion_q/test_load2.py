import pickle, glob, os
import pandas as pd
import numpy as np
import sys

def process_chunk_vectorized(events, ckpt_idx, gamma=0.99):
    if not events: return pd.DataFrame()
    df = pd.DataFrame(events)
    if 'episode' in df.columns and 'episode_sub_idx' not in df.columns:
        df = df.rename(columns={'episode': 'episode_sub_idx'})
    preds = df[df['event'] == 'predict'].drop(columns=['reward'], errors='ignore').copy()
    rewards = df[df['event'] == 'reward'].copy()
    if preds.empty or rewards.empty: return pd.DataFrame()
    preds['seq'] = preds.groupby(['episode_sub_idx', 'bus_id']).cumcount()
    rewards['seq'] = rewards.groupby(['episode_sub_idx', 'bus_id']).cumcount()
    merged = pd.merge(preds, rewards[['episode_sub_idx', 'bus_id', 'seq', 'reward']], 
                      on=['episode_sub_idx', 'bus_id', 'seq'], how='inner')
    if merged.empty: return pd.DataFrame()
    merged = merged.sort_values(['episode_sub_idx', 'bus_id', 'seq'], ascending=[True, True, False])
    
    def compute_returns_for_group(grp):
        r = grp['reward'].values
        qs = np.zeros_like(r)
        curr = 0
        for i in range(len(r)):
            curr = r[i] + gamma * curr
            qs[i] = curr
        grp['q_real'] = qs
        return grp

    merged = merged.groupby(['episode_sub_idx', 'bus_id'], group_keys=False).apply(compute_returns_for_group)
    state_list = merged['state_raw'].values.tolist()
    states = np.array(state_list)
    final_df = pd.DataFrame({
        'checkpoint': ckpt_idx,
        'station_id': states[:, 1].astype(int),
        'direction': states[:, 3].astype(int),
        'forward_headway': states[:, 4].astype(float),
        'backward_headway': states[:, 5].astype(float),
        'q_pred_mean': merged['q_vals'].apply(np.mean).values,
        'q_pred_std': merged['q_vals'].apply(np.std).values,
        'q_real': merged['q_real'].values,
        'q_vals': merged['q_vals'].values
    })
    return final_df

files = sorted(glob.glob('offline_timeline_data_20eps/data_ensemble_*.pkl'))[:10]
dfs = []
for f in files:
    with open(f, 'rb') as fh:
        data = pickle.load(fh)
    events = data if isinstance(data, list) else data.get('data', data.get('events', []))
    ckpt = int(os.path.basename(f).split('_')[-1].replace('.pkl',''))
    df = process_chunk_vectorized(events, ckpt)
    dfs.append(df)
df = pd.concat(dfs)
print("Memory usage of 10 files DataFrame:", df.memory_usage(deep=True).sum() / 1e6, "MB")
print("Total rows:", len(df))
