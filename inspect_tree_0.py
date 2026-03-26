"""
Tree Structure Inspection (Tree 0, 2, 4)

This script loads the initial High-Entropy Metal Phosphates (HEMPs) dataset,
preprocesses it to extract relevant metal compositions and configurational entropy,
and trains a Random Forest Regressor on the 80°C conductivity data.
It subsequently inspects the structural properties (depth and total node count)
of selected estimator trees (trees 0, 2, and 4) from the trained ensemble.

Author: Tomasz Stawski (tomasz.stawski@bam.de)
License: MIT License
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load raw experimental data from the CSV file utilising a multi-level header
df_raw = pd.read_csv('HEMPs.csv', header=[0, 1])
df = pd.DataFrame()

# Define the primary chemical constituents of interest
metals = ['Ni', 'Co', 'Fe', 'Mn', 'Cd', 'Cu', 'Mg', 'Zn']

# Extract composition data for each metal, handling missing occurrences
for metal in metals:
    try:
        col_data = df_raw.loc[:, (slice(None), metal)].iloc[:, 0]
        # Coerce non-numeric entries like '-' to zero
        df[metal] = pd.to_numeric(col_data.replace('-', '0'), errors='coerce').fillna(0)
    except KeyError:
        df[metal] = 0

# Extract configurational entropy
df['Entropy'] = pd.to_numeric(df_raw.loc[:, (slice(None), 'R')].iloc[:, 0], errors='coerce')

# Process conductivity measurements for the 25°C and 80°C conditions
for temp_str, col_name in [('25 °C', 'Cond25C'), ('80 °C', 'Cond80C')]:
    try:
        raw_series = df_raw.loc[:, (slice(None), temp_str)].iloc[:, 0]
        df[col_name] = pd.to_numeric(raw_series, errors='coerce')
        # Apply logarithmic transformation to the conductivity target variable
        df[f'log{col_name}'] = np.log10(df[col_name])
    except KeyError:
        pass

# Filter the dataset for valid 80°C conductivity measurements
mask = np.isfinite(df['logCond80C'])
df_80 = df[mask].copy()

# Define feature matrix and target vector for the 80°C regime
X = df_80[metals + ['Entropy']]
y = df_80['logCond80C']

# Perform a standardised train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise and train the Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
rf.fit(X_train, y_train)

# Inspect specific estimators within the random forest ensemble
# Evaluate tree index 0
tree = rf.estimators_[0]
print("Tree 0 depth:", tree.get_depth())
print("Tree 0 nodes:", tree.tree_.node_count)

# Evaluate tree index 2
tree2 = rf.estimators_[2]
print("Tree 2 depth:", tree2.get_depth())
print("Tree 2 nodes:", tree2.tree_.node_count)

# Evaluate tree index 4
tree4 = rf.estimators_[4]
print("Tree 4 depth:", tree4.get_depth())
print("Tree 4 nodes:", tree4.tree_.node_count)
