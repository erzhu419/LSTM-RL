import pandas as pd
import os

path = "/home/erzhu419/mine_code/LSTM-RL/env/data"
df = pd.read_excel(os.path.join(path, "time_table.xlsx"))
print(df.head())
print("Min launch time:", df['launch_time'].min())
