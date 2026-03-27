import pickle, glob, os
import pandas as pd
import sys
from analyze_mahalanobis_rareness import process_chunk_vectorized

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
