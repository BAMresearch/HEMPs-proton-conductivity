"""
Random Forest analysis for HEMPs (Extended dataset)

Analyses liquid-to-solid partitioning and element interactions in High-Entropy 
Metal Phosphates (HEMPs):
1. Partitioning (ΔX): difference between solid and solution concentrations (atomic %).
2. Cross-interactions: Leave-Self-Out regressions to evaluate how solution elements affect solid composition.
3. Covariance & solubility limits: correlation with conductivity at 80 °C and elemental limits.

Author: Tomasz M. Stawski (tomasz.stawski@bam.de)
License: MIT License
"""

import os
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
    if not os.path.exists(file_path):
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if os.path.exists(alt_path):
            file_path = alt_path

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

    # Extract conductivity data for both experimental temperatures (25 °C and 80 °C)
    for idx, col_name in [(18, 'Cond25C'), (19, 'Cond80C')]:
        try:
            raw_series = df_raw.iloc[:, idx]
            df[col_name] = pd.to_numeric(raw_series.replace('-', 'NaN'), errors='coerce')
            df[f'log10_{col_name}'] = np.log10(df[col_name])
        except Exception:
            pass

    return df, metals, metals_sol_cols, metals_solid_cols



# --- Statistical Engines ---

def _train_single_model(X, y, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
    rf.fit(X_train, y_train)
    return rf.feature_importances_

def run_stability_analysis(X, y, features, iterations=1000):
    with tqdm_joblib(tqdm(desc="Bootstrapping", total=iterations)):
        importances = Parallel(n_jobs=-1, verbose=0, prefer="threads")(
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

def plot_partitioning_analysis(df, metals, metals_sol_cols, metals_solid_cols, output_dir='output'):
    """
    Plots elemental partitioning (ΔX = X_solid - X_solution) across metals.
    """
    os.makedirs(output_dir, exist_ok=True)
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
    plt.title(r"Elemental Partitioning into Solid ($\Delta X = X_{solid} - X_{solution}$)", fontsize=14)
    plt.ylabel(r"$\Delta X$ / %")
    plt.xlabel("Target Metal")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SI_Figure_Partitioning.pdf'))
    plt.close()


def predict_cross_interactions(df, metals, metals_sol_cols, metals_solid_cols, output_dir='output'):
    """
    Predicts ΔX for each metal from the remaining solution concentrations
    using Random Forest and Ridge regression.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Running Inter-element Cross-Interaction Mapping ---")
    
    important_metals = []
    
    # Identify metals that vary in the solid phase
    for metal, col_sol, col_solid in zip(metals, metals_sol_cols, metals_solid_cols):
        if df[col_solid].std() > 5.0:
            important_metals.append((metal, col_sol, col_solid))

    if not important_metals:
        print("Not enough variance in solid composition to run mapping.")
        return

    n_metals = len(important_metals)
    fig, axes = plt.subplots(2, n_metals, figsize=(4 * max(n_metals, 2), 8), dpi=300)
    if n_metals == 1:
        axes = axes.reshape(2, 1)

    for i, (metal, col_sol, col_solid) in enumerate(important_metals):
        print(f"   ...Mapping cross-interactions for $\\Delta${metal} partitioning...")
        
        # Leave self out: exclude target metal solution concentration
        cross_features = [col for col in metals_sol_cols if col != col_sol]
        X = df[cross_features]
        y = df[col_solid] - df[col_sol]
        
        imp, std = run_stability_analysis(X, y, cross_features, iterations=1000)
        coefs = run_ridge_analysis(X, y, cross_features)
        
        # Plot importance
        ax_imp = axes[0, i]
        imp.plot(kind='bar', yerr=std, capsize=3, ax=ax_imp, color='#2ca25f', edgecolor='black', lw=0.8)
        ax_imp.set_title(f'Predicting $\\Delta${metal}')
        if i == 0:
            ax_imp.set_ylabel('Mean RF Importance')
        ax_imp.grid(True, axis='both', linestyle='--', alpha=0.3)
        ax_imp.set_xticklabels([label.get_text().replace('_sol', '') for label in ax_imp.get_xticklabels()], rotation=45)

        # Plot directionality
        ax_dir = axes[1, i]
        colors_ridge = ['#d7191c' if c < 0 else '#2b83ba' for c in coefs]
        coefs.plot(kind='barh', ax=ax_dir, color=colors_ridge, edgecolor='black', lw=0.8)
        ax_dir.set_xlabel('Ridge Coefficient')
        if i == 0:
            ax_dir.set_ylabel('Promoter (positive) vs Blocker (negative)')
        ax_dir.axvline(0, color='black', lw=1)
        ax_dir.grid(True, axis='both', linestyle='--', alpha=0.3)
        ax_dir.set_yticklabels([label.get_text().replace('_sol', '') for label in ax_dir.get_yticklabels()])

    plt.suptitle(r"Cross-Element Interactions Driving Partitioning ($\Delta X$)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SI_Figure_Cross_Interactions.pdf'))
    plt.close()


def plot_covariance_analysis(df, metals, metals_solid_cols, output_dir='output'):
    """
    Computes compositional covariance, elemental solubility boundaries, and
    property correlations against conductivity at 80 °C.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Running Multivariate Covariance and Solubility Analysis ---")
    try:
        from CovarianceAnalysis import (
            analyse_solubility_limits,
            compute_covariance_metrics,
            plot_covariance_analysis_figure
        )
        
        # Prepare DataFrame formatted with standard metal column names
        df_cov = pd.DataFrame()
        for metal, solid_col in zip(metals, metals_solid_cols):
            df_cov[metal] = df[solid_col]
        df_cov['Entropy (R)'] = df['Entropy']
        if 'log10_Cond80C' in df.columns:
            df_cov['log10_Cond80C'] = df['log10_Cond80C']
            
        # 1. Print Solubility Limits Table
        sol_df = analyse_solubility_limits(df_cov, metals)
        print("\nSolid-Solution Solubility Limits & Structural Roles:")
        print(sol_df.to_string(index=False))

        # 2. Print Target Correlations Table
        corr_matrix, target_stats = compute_covariance_metrics(df_cov, metals, target_col='log10_Cond80C')
        print("\nCorrelation with log10(Cond 80°C):")
        print(target_stats.to_string(index=False))

        # 3. Generate Publication Figure
        out_paths = [
            os.path.join(output_dir, 'SI_Figure_Covariance_Analysis.pdf'),
            os.path.join(output_dir, 'SI_Figure_Covariance_Analysis.png')
        ]
        plot_covariance_analysis_figure(
            df_cov, metals, target_col='log10_Cond80C',
            temp_label='80C', output_paths=out_paths
        )
        print("   [+] Covariance analysis figures generated successfully.")
    except Exception as e:
        print(f"Warning: Covariance analysis encountered an error: {e}")


# --- Main Execution Pipeline ---

def run_analysis_pipeline_extended():
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    try:
        df, metals, metals_sol_cols, metals_solid_cols = load_and_preprocess_extended_data('HEMPsExtended.csv')
    except Exception as e:
        print(f"Critical Error: Failed reading dataset. {e}")
        return

    # 1. Partitioning Analysis (Enrichment / Depletion)
    plot_partitioning_analysis(df, metals, metals_sol_cols, metals_solid_cols, output_dir=output_dir)

    # 2. Predictive Modelling (Solution -> Solid Cross-Interactions)
    predict_cross_interactions(df, metals, metals_sol_cols, metals_solid_cols, output_dir=output_dir)

    # 3. Multivariate Covariance and Solubility Limits Analysis
    plot_covariance_analysis(df, metals, metals_solid_cols, output_dir=output_dir)

    print("\nExtended analysis pipeline completed successfully.")

if __name__ == "__main__":
    run_analysis_pipeline_extended()
