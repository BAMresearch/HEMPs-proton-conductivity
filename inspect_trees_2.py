"""
Inspect root node sample splits across Random Forest trees using child node indexing.

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

try:
    df['Entropy'] = pd.to_numeric(df_raw.loc[:, (slice(None), 'R')].iloc[:, 0], errors='coerce')
except Exception:
    df['Entropy'] = 0

try:
    df['NumCations'] = df_raw.iloc[:, 0]
except Exception:
    df['NumCations'] = 0

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

best_diff = 100
best_tree = None

for i, tree in enumerate(rf.estimators_):
    n_samples = tree.tree_.n_node_samples
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    
    if children_left[0] != -1:
        left_samples = n_samples[children_left[0]]
        right_samples = n_samples[children_right[0]]
        
        if i < 10:
            print(f"Tree {i}: root split {left_samples} vs {right_samples}")
        
        diff = abs(left_samples - right_samples)
        if diff < best_diff and left_samples > 3 and right_samples > 3:
            best_diff = diff
            best_tree = i
            
        if i == 4:
            print(f"** Tree 4 root split: {left_samples} vs {right_samples} **")

print(f"\nBest tree index: {best_tree} with diff {best_diff}")
