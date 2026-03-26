"""
Optimal Tree Search (Ensemble Inspection)

This module processes the primary experimental data for High-Entropy Metal Phosphates 
(HEMPs) and builds a robust Random Forest Regressor targeting 80°C conductivity. 
Subsequently, it evaluates structural balance across the internal trees in the random 
forest ensemble to locate an optimal representative tree defined by its initial 
node sample partition parity.

Author: Tomasz Stawski (tomasz.stawski@bam.de)
License: MIT License
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Retrieve source data with corresponding multi-indexed columns
df_raw = pd.read_csv('HEMPs.csv', header=[0, 1])
df = pd.DataFrame()

# Designated predictor chemical species
metals = ['Ni', 'Co', 'Fe', 'Mn', 'Cd', 'Cu', 'Mg', 'Zn']

# Systematically convert experimental concentration records into an analytical frame
for metal in metals:
    try:
        col_data = df_raw.loc[:, (slice(None), metal)].iloc[:, 0]
        df[metal] = pd.to_numeric(col_data.replace('-', '0'), errors='coerce').fillna(0)
    except KeyError:
        df[metal] = 0

# Extract configurational entropy safely
try:
    df['Entropy'] = pd.to_numeric(df_raw.loc[:, (slice(None), 'R')].iloc[:, 0], errors='coerce')
except Exception:
    df['Entropy'] = 0

# Extract overall cation count
try:
    df['NumCations'] = df_raw.iloc[:, 0]
except Exception:
    df['NumCations'] = 0

# Parse empirical temperature-dependent conductivity and formulate its logarithmic analogue
for temp_str, col_name in [('25 °C', 'Cond25C'), ('80 °C', 'Cond80C')]:
    try:
        raw_series = df_raw.loc[:, (slice(None), temp_str)].iloc[:, 0]
        df[col_name] = pd.to_numeric(raw_series, errors='coerce')
        df[f'log{col_name}'] = np.log10(df[col_name])
    except KeyError:
        pass

# Ensure computation includes exclusively quantitative targets
mask = np.isfinite(df['logCond80C'])
df_80 = df[mask].copy()

# Allocate statistical predictors and output vectors
X = df_80[metals + ['Entropy']]
y = df_80['logCond80C']

# Generate generalized evaluation partitions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compile the baseline forest containing 100 constituent decision trees
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
rf.fit(X_train, y_train)

best_diff = 100
best_tree = None

# Iterate across all ensemble members to assess initial node balance constraint
for i, tree in enumerate(rf.estimators_):
    n_samples = tree.tree_.n_node_samples
    
    # Confirm tree has more than one node
    if len(n_samples) > 2:
        left_samples = n_samples[1]
        right_samples = n_samples[2]
        print(f"Tree {i}: root split {left_samples} vs {right_samples}")
        
        # Calculate sample allocation variance across the root node
        diff = abs(left_samples - right_samples)
        
        # Assess potential representative trees emphasizing symmetrical primary splits
        if diff < best_diff and left_samples > 3 and right_samples > 3:
            best_diff = diff
            best_tree = i

# Document the maximally balanced representative estimator
print(f"\nBest tree index: {best_tree} with diff {best_diff}")
