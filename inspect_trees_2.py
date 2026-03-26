"""
Detailed Optimal Tree Search (Root Node Validation)

This script analyses High-Entropy Metal Phosphates (HEMPs) experimental 
compositional features corresponding to 80°C log-transformed solid conductivity. 
A Random Forest ensemble is constructed to model physical parameters, and 
the architectural topology of generated estimators is interrogated. The 
algorithm correctly traverses the decision graph using structural indexing 
(`children_left` and `children_right`) to assess the root sample split balance.

Author: Tomasz Stawski (tomasz.stawski@bam.de)
License: MIT License
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Process source data using a hierarchical level mapping
df_raw = pd.read_csv('HEMPs.csv', header=[0, 1])
df = pd.DataFrame()

# Categorise and organize constituent variables
metals = ['Ni', 'Co', 'Fe', 'Mn', 'Cd', 'Cu', 'Mg', 'Zn']
for metal in metals:
    try:
        col_data = df_raw.loc[:, (slice(None), metal)].iloc[:, 0]
        # Nullify invalid indicators
        df[metal] = pd.to_numeric(col_data.replace('-', '0'), errors='coerce').fillna(0)
    except KeyError:
        df[metal] = 0

# Extract specific structural descriptors
try:
    df['Entropy'] = pd.to_numeric(df_raw.loc[:, (slice(None), 'R')].iloc[:, 0], errors='coerce')
except Exception:
    df['Entropy'] = 0

try:
    df['NumCations'] = df_raw.iloc[:, 0]
except Exception:
    df['NumCations'] = 0

# Evaluate logarithmic transfer mechanisms for varying regimes
for temp_str, col_name in [('25 °C', 'Cond25C'), ('80 °C', 'Cond80C')]:
    try:
        raw_series = df_raw.loc[:, (slice(None), temp_str)].iloc[:, 0]
        df[col_name] = pd.to_numeric(raw_series, errors='coerce')
        df[f'log{col_name}'] = np.log10(df[col_name])
    except KeyError:
        pass

# Mask undefined calculations
mask = np.isfinite(df['logCond80C'])
df_80 = df[mask].copy()

# Construct fundamental components for regression mapping
X = df_80[metals + ['Entropy']]
y = df_80['logCond80C']

# Ensure standard randomization strategy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Engage main ensemble configuration 
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
rf.fit(X_train, y_train)

best_diff = 100
best_tree = None

# Interrogate discrete ensemble structures for balanced top-tier splits
for i, tree in enumerate(rf.estimators_):
    n_samples = tree.tree_.n_node_samples
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    
    # Confirm the existence of child nodes extending from the root
    if children_left[0] != -1:
        # Utilise accurate edge mappings to locate sub-node distribution
        left_samples = n_samples[children_left[0]]
        right_samples = n_samples[children_right[0]]
        
        # Display sample distributions for the first 10 evaluated models
        if i < 10:
            print(f"Tree {i}: root split {left_samples} vs {right_samples}")
        
        # Record trees demonstrating uniform parity across the primary split
        diff = abs(left_samples - right_samples)
        if diff < best_diff and left_samples > 3 and right_samples > 3:
            best_diff = diff
            best_tree = i
            
        # Highlight manually validated structural anomaly instances
        if i == 4:
            print(f"** Tree 4 root split: {left_samples} vs {right_samples} **")

# Report optimal tree index
print(f"\nBest tree index: {best_tree} with diff {best_diff}")
