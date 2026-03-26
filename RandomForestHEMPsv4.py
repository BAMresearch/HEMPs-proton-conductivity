"""
RandomForest for HEMPs - Version 4 (Extended Analysis)

This module systematically analyses disproportionation behaviours within High-Entropy 
Metal Phosphates (HEMPs) by correlating early-stage solution compositions against 
final consolidated solid-phase attributes.

Primary Analytical Procedures:
1. Liquid-to-Solid Partitioning: Evaluates the differential concentration parameters mathematically represented as ΔX (Solid Atomic % - Solution Atomic %).
2. Predictive Disproportionation: Deploys Random Forest regression to establish functional mappings linking initial solvent compositions to consolidated metal integrations.
3. Property Diagnostics: Utilises regularisation algorithms (Ridge Regression) in tandem with structural ensemble metrics to distinguish influential conductivity drivers originating from resultant compositional profiles.

Author: Tomasz Stawski (tomasz.stawski@bam.de)
License: MIT License
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import contextlib
import warnings

# Suppress warnings that might clutter output
warnings.filterwarnings('ignore', category=RuntimeWarning)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# Global matplotlib configuration for data plotting
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.frameon": False
})

# --- Data Loading and Cleaning ---

def load_and_preprocess_extended_data(file_path):
    """
    Loads HEMPsExtended.csv and extracts both Solution and Solid compositions.
    """
    df_raw = pd.read_csv(file_path, header=[0, 1])
    df = pd.DataFrame()
    metals = ['Ni', 'Co', 'Fe', 'Mn', 'Cd', 'Cu', 'Mg', 'Zn']
    
    metals_sol_cols = []
    metals_solid_cols = []

    try:
        df['NumCations'] = pd.to_numeric(df_raw.iloc[:, 0].replace('-', '0'), errors='coerce').fillna(0)
    except Exception:
        df['NumCations'] = 0

    # Extract metal compositions based on known column indices
    # Indices 1-8: Solution, Indices 9-16: Solid
    for i, metal in enumerate(metals):
        col_sol = f'{metal}_sol'
        col_solid = f'{metal}_solid'
        metals_sol_cols.append(col_sol)
        metals_solid_cols.append(col_solid)
        
        df[col_sol] = pd.to_numeric(df_raw.iloc[:, i + 1].replace('-', '0'), errors='coerce').fillna(0)
        df[col_solid] = pd.to_numeric(df_raw.iloc[:, i + 9].replace('-', '0'), errors='coerce').fillna(0)

    # Extract Entropy
    try:
        df['Entropy'] = pd.to_numeric(df_raw.iloc[:, 17].replace('-', 'NaN'), errors='coerce').fillna(0)
    except Exception:
        df['Entropy'] = 0

    return df, metals, metals_sol_cols, metals_solid_cols



# --- Statistical Engines ---

def _train_single_model(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
    rf.fit(X_train, y_train)
    return rf.feature_importances_

def run_stability_analysis(X, y, features, iterations=1000):
    with tqdm_joblib(tqdm(desc="Bootstrapping", total=iterations)):
        importances = Parallel(n_jobs=-1, verbose=0)(
            delayed(_train_single_model)(X, y, i) for i in range(iterations)
        )
    importance_df = pd.DataFrame(importances, columns=features)
    mean_imp = importance_df.mean().sort_values(ascending=False)
    std_imp = importance_df.std().reindex(mean_imp.index)
    return mean_imp, std_imp

def run_ridge_analysis(X, y, features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0).fit(X_scaled, y)
    return pd.Series(ridge.coef_, index=features).sort_values(ascending=False)


# --- Elemental Partitioning and Disproportionation Analysis ---

def plot_partitioning_analysis(df, metals, metals_sol_cols, metals_solid_cols):
    """
    Computes component differentials (ΔX = X_solid - X_sol) across the specified elemental 
    series. Synthesises a distributional boxplot portraying systematic target 
    enrichment and depletion tendencies.
    """
    print("\n--- Running Partitioning Analysis (Enrichment / Depletion) ---")
    delta_df = pd.DataFrame()
    for metal, col_sol, col_solid in zip(metals, metals_sol_cols, metals_solid_cols):
        # Only consider rows where the metal was at least present in the solution OR solid
        mask = (df[col_sol] > 0) | (df[col_solid] > 0)
        delta = df.loc[mask, col_solid] - df.loc[mask, col_sol]
        delta_df[metal] = delta

    plt.figure(figsize=(10, 6), dpi=300)
    sns.boxplot(data=delta_df, palette="vlag", showfliers=False)
    sns.stripplot(data=delta_df, color="black", alpha=0.5, jitter=True, size=4)
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='No Partitioning')
    plt.title("Elemental Partitioning into Solid ($\Delta X = X_{solid} - X_{solution}$)", fontsize=14)
    plt.ylabel("$\Delta X$ / %")
    plt.xlabel("Target Metal")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('SI_Figure_Partitioning.pdf')
    plt.close()


def predict_cross_interactions(df, metals, metals_sol_cols, metals_solid_cols):
    """
    Generates predictive models (Random Forest and Ridge Regression) to evaluate 
    elemental enrichment or depletion (ΔX) using remaining initial solution components.
    This methodology isolates inter-elemental scaling relationships (antagonistic 
    vs. synergistic interactions) while removing singular autocorrelation variables.
    """
    print("\n--- Running Inter-element Cross-Interaction Mapping ---")
    
    important_metals = []
    
    # Identify metals that actually vary in the solid phase
    for metal, col_sol, col_solid in zip(metals, metals_sol_cols, metals_solid_cols):
        if df[col_solid].std() > 5.0:  # Threshold to ignore trace/inactive elements
            important_metals.append((metal, col_sol, col_solid))

    if not important_metals:
        print("Not enough variance in solid composition to run mapping.")
        return

    # Prepare subplot grid based on number of important metals
    n_metals = len(important_metals)
    fig, axes = plt.subplots(2, n_metals, figsize=(4 * max(n_metals, 2), 8), dpi=300)
    if n_metals == 1:
        axes = axes.reshape(2, 1)

    for i, (metal, col_sol, col_solid) in enumerate(important_metals):
        print(f"   ...Mapping cross-interactions for $\Delta${metal} partitioning...")
        
        # LEAVE-SELF-OUT: Remove the target metal's solution concentration from predictors
        cross_features = [col for col in metals_sol_cols if col != col_sol]
        X = df[cross_features]
        
        # Target: Delta X (Solid% - Sol%) representing enrichment or depletion
        y = df[col_solid] - df[col_sol]
        
        # Use 1000 iterations for robust stability analysis
        imp, std = run_stability_analysis(X, y, cross_features, iterations=1000)
        coefs = run_ridge_analysis(X, y, cross_features)
        
        # Plotting Importance (Top Row)
        ax_imp = axes[0, i]
        imp.plot(kind='bar', yerr=std, capsize=3, ax=ax_imp, color='#2ca25f', edgecolor='black', lw=0.8)
        
        ax_imp.set_title(f'Predicting $\Delta${metal}')
        if i == 0:
            ax_imp.set_ylabel('Mean RF Importance')
        ax_imp.grid(True, axis='both', linestyle='--', alpha=0.3)
        ax_imp.set_xticklabels([label.get_text().replace('_sol', '') for label in ax_imp.get_xticklabels()], rotation=45)

        # Plotting Directionality (Bottom Row)
        ax_dir = axes[1, i]
        # Colours: red for antagonistic (negative), blue for synergistic (positive)
        colors_ridge = ['#d7191c' if c < 0 else '#2b83ba' for c in coefs]
        coefs.plot(kind='barh', ax=ax_dir, color=colors_ridge, edgecolor='black', lw=0.8)
        
        ax_dir.set_xlabel('Ridge Coefficient')
        if i == 0:
            ax_dir.set_ylabel('Promoter (Synergistic) vs Blocker (Antagonistic)')
        ax_dir.axvline(0, color='black', lw=1)
        ax_dir.grid(True, axis='both', linestyle='--', alpha=0.3)
        ax_dir.set_yticklabels([label.get_text().replace('_sol', '') for label in ax_dir.get_yticklabels()])

    plt.suptitle("Cross-Element Interactions Driving Partitioning ($\Delta X$)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('SI_Figure_Cross_Interactions.pdf')
    plt.close()




# --- Main Execution Pipeline ---

def run_analysis_pipeline_extended():
    try:
        df, metals, metals_sol_cols, metals_solid_cols = load_and_preprocess_extended_data('HEMPsExtended.csv')
    except Exception as e:
        print(f"Critical Error: Failed reading dataset. {e}")
        return

    # 1. New Partitioning Analysis
    plot_partitioning_analysis(df, metals, metals_sol_cols, metals_solid_cols)

    # 2. Predictive Modelling (Solution -> Solid Cross-Interactions)
    predict_cross_interactions(df, metals, metals_sol_cols, metals_solid_cols)

    print("Extended analysis pipeline completed successfully.")

if __name__ == "__main__":
    run_analysis_pipeline_extended()
