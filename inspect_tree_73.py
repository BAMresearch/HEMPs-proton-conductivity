"""
Tree Textual Export (Tree 73)

This script imports the High-Entropy Metal Phosphates (HEMPs) dataset,
prepares the compositional and entropic features, and trains a
Random Forest model to predict the transformed conductivity at 80°C.
It then isolates tree index 73 from the ensemble and exports its 
internal decision structure using a textual representation, restricted
to a specified maximal depth.

Author: Tomasz Stawski (tomasz.stawski@bam.de)
License: MIT License
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

# Ingest data using a multi-index configuration
df_raw = pd.read_csv('HEMPs.csv', header=[0, 1])
df = pd.DataFrame()

# Specify the target chemical components
metals = ['Ni', 'Co', 'Fe', 'Mn', 'Cd', 'Cu', 'Mg', 'Zn']

# Iteratively parse and structure the atomic composition percentages
for metal in metals:
    try:
        col_data = df_raw.loc[:, (slice(None), metal)].iloc[:, 0]
        # Resolve non-quantitative indicators to zero
        df[metal] = pd.to_numeric(col_data.replace('-', '0'), errors='coerce').fillna(0)
    except KeyError:
        df[metal] = 0

# Parse the configurational entropy feature
try:
    df['Entropy'] = pd.to_numeric(df_raw.loc[:, (slice(None), 'R')].iloc[:, 0], errors='coerce')
except Exception:
    df['Entropy'] = 0

# Convert target variables to logarithmic scale for analysis
for temp_str, col_name in [('25 °C', 'Cond25C'), ('80 °C', 'Cond80C')]:
    try:
        raw_series = df_raw.loc[:, (slice(None), temp_str)].iloc[:, 0]
        df[col_name] = pd.to_numeric(raw_series, errors='coerce')
        df[f'log{col_name}'] = np.log10(df[col_name])
    except KeyError:
        pass

# Ensure only valid numeric samples at 80°C are retained
mask = np.isfinite(df['logCond80C'])
df_80 = df[mask].copy()

# Construct the feature array and target vector
X = df_80[metals + ['Entropy']]
y = df_80['logCond80C']

# Create training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the primary predictive model
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
rf.fit(X_train, y_train)

# Identify tree 73 from the estimator ensemble
tree = rf.estimators_[73]

# Export the decision rules as text limited to depth 4
r = export_text(tree, feature_names=list(X.columns), max_depth=4)
print(r)
