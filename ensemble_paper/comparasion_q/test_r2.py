import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from analyze_mahalanobis_rareness import load_all_data

df = load_all_data('test_dataset', 'ensemble')
X = df['q_pred_mean'].values.reshape(-1, 1)
y = df['q_real'].values
reg = LinearRegression().fit(X, y)
r2 = reg.score(X, y)
print(f"Ensemble R²: {r2:.4f}, coef: {reg.coef_[0]:.4f}")
