"""
Inspect decision tree depths and node counts from the Random Forest model.

Author: Tomasz M. Stawski (tomasz.stawski@bam.de)
License: MIT License
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
file_path = 'HEMPs.csv'
if not os.path.exists(file_path):
    alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    if os.path.exists(alt_path):
        file_path = alt_path

df_raw = pd.read_csv(file_path, header=[0, 1])
df = pd.DataFrame()

metals = ['Ni', 'Co', 'Fe', 'Mn', 'Cd', 'Cu', 'Mg', 'Zn']
for metal in metals:
    try:
        col_data = df_raw.loc[:, (slice(None), metal)].iloc[:, 0]
        df[metal] = pd.to_numeric(col_data.replace('-', '0'), errors='coerce').fillna(0)
    except KeyError:
        df[metal] = 0

df['Entropy'] = pd.to_numeric(df_raw.loc[:, (slice(None), 'R')].iloc[:, 0], errors='coerce')

for temp_str, col_name in [('25 °C', 'Cond25C'), ('80 °C', 'Cond80C')]:
    try:
        raw_series = df_raw.loc[:, (slice(None), temp_str)].iloc[:, 0]
        df[col_name] = pd.to_numeric(raw_series, errors='coerce')
        df[f'log{col_name}'] = np.log10(df[col_name])
    except KeyError:
        pass

# 80 °C dataset
df_80 = df[np.isfinite(df['logCond80C'])].copy()
X = df_80[metals + ['Entropy']]
y = df_80['logCond80C']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
rf.fit(X_train, y_train)

# Inspect depths and node counts of trees 0, 2, and 4
for idx in [0, 2, 4]:
    tree = rf.estimators_[idx]
    print(f"Tree {idx} depth: {tree.get_depth()}")
    print(f"Tree {idx} nodes: {tree.tree_.node_count}")
