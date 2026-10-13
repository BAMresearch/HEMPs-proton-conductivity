"""
Covariance and solubility limits analysis for High-Entropy Metal Phosphates (HEMPs)

Analyses compositional covariance, solid-solution solubility boundaries, and
correlations with proton conductivity (log10(sigma)) at 25 °C and 80 °C.

Author: Tomasz M. Stawski (tomasz.stawski@bam.de)
License: MIT License
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress runtime warnings that may clutter terminal output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Global matplotlib configuration for publication-standard aesthetics
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.frameon": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold"
})


# --- Data Loading and Cleaning ---

def load_data(file_path='HEMPs.csv'):
    """
    Loads and cleans HEMP experimental composition and conductivity data.
    Supports both HEMPs.csv and HEMPsExtended.csv formats.

    Parameters
    ----------
    file_path : str
        Path to the CSV data file.

    Returns
    -------
    df : pd.DataFrame
        Cleaned flat DataFrame containing cation percentages, entropy, and log10 conductivity.
    metals : list of str
        List of target chemical symbols.
    """
    if not os.path.exists(file_path):
        # Fallback to search in relative directory or codeForPublication
        alt_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find dataset at {file_path}")

    df_raw = pd.read_csv(file_path, header=[0, 1])
    df = pd.DataFrame()
    metals = ['Ni', 'Co', 'Fe', 'Mn', 'Cd', 'Cu', 'Mg', 'Zn']

    # Detect if file is extended format (with solution and solid prefixes)
    is_extended = any('in solution' in str(col).lower() for col in df_raw.columns)

    if is_extended:
        # Extract solid-phase concentrations (indices 9 to 16)
        for i, metal in enumerate(metals):
            col_data = df_raw.iloc[:, i + 9]
            df[metal] = pd.to_numeric(col_data.replace('-', '0'), errors='coerce').fillna(0)
        # Extract solution concentrations as well
        for i, metal in enumerate(metals):
            col_data = df_raw.iloc[:, i + 1]
            df[f'{metal}_sol'] = pd.to_numeric(col_data.replace('-', '0'), errors='coerce').fillna(0)
    else:
        # Standard HEMPs.csv format
        for metal in metals:
            try:
                col_data = df_raw.loc[:, (slice(None), metal)].iloc[:, 0]
                df[metal] = pd.to_numeric(col_data.replace('-', '0'), errors='coerce').fillna(0)
            except KeyError:
                df[metal] = 0.0

    # Extract Configurational Entropy (R)
    try:
        entropy_col = df_raw.loc[:, (slice(None), 'R')].iloc[:, 0]
        df['Entropy (R)'] = pd.to_numeric(entropy_col.replace('-', 'NaN'), errors='coerce').fillna(0.0)
    except Exception:
        df['Entropy (R)'] = 0.0

    # Extract conductivity data for both thermal regimes (25 °C and 80 °C)
    for temp_str, col_name in [('25 °C', 'Cond25C'), ('80 °C', 'Cond80C')]:
        try:
            raw_series = df_raw.loc[:, (slice(None), temp_str)].iloc[:, 0]
            df[col_name] = pd.to_numeric(raw_series.replace('-', 'NaN'), errors='coerce')
            df[f'log10_{col_name}'] = np.log10(df[col_name])
        except Exception:
            pass

    return df, metals


# --- Analytical Engines ---

def analyse_solubility_limits(df, metals):
    """
    Evaluates maximum solid-phase concentrations (X_solid,max) and 
    classifies solid-solution behaviour and crystallographic boundaries.

    Returns
    -------
    solubility_df : pd.DataFrame
        Summary table detailing solubility limits and behaviour categories.
    """
    solubility_data = []
    
    classification_rules = {
        'Complete': 'Complete Solid Solution',
        'Moderate': 'Moderate Solubility Limit',
        'Severe': 'Severe Solubility Limit (Verge)',
        'Negligible': 'Negligible Solubility Limit'
    }

    for metal in metals:
        max_val = float(df[metal].max())
        mean_val = float(df[df[metal] > 0][metal].mean()) if (df[metal] > 0).any() else 0.0
        
        if max_val >= 99.0:
            behaviour = classification_rules['Complete']
            role = 'Primary Structural Scaffold / Solvent'
        elif max_val >= 30.0:
            behaviour = classification_rules['Moderate']
            role = 'Secondary Component / Entropy Contributor'
        elif max_val >= 20.0:
            behaviour = classification_rules['Severe']
            role = 'Functional Dopant / Transport Booster'
        else:
            behaviour = classification_rules['Negligible']
            role = 'Minor Guest / Entropy Bottleneck'

        solubility_data.append({
            'Cation': metal,
            'Max Solid Concentration (%)': max_val,
            'Mean Non-Zero (%)': round(mean_val, 1),
            'Solid Solution Behaviour': behaviour,
            'Structural Role': role
        })

    solubility_df = pd.DataFrame(solubility_data).sort_values(
        by='Max Solid Concentration (%)', ascending=False
    ).reset_index(drop=True)

    return solubility_df


# Backward-compatible alias for existing callers
analyze_solubility_limits = analyse_solubility_limits


def compute_covariance_metrics(df, metals, target_col='log10_Cond80C'):
    """
    Computes Pearson correlation coefficients, covariance matrices, Spearman rank 
    correlations, and two-tailed p-values against target conductivity.

    Returns
    -------
    corr_matrix : pd.DataFrame
        Pairwise Pearson correlation matrix across composition variables and target.
    target_summary_df : pd.DataFrame
        Detailed statistical summary of correlation with the target variable.
    """
    feature_cols = metals + ['Entropy (R)']
    analysis_cols = feature_cols + [target_col]
    
    # Filter valid complete cases
    df_valid = df[analysis_cols].dropna()
    corr_matrix = df_valid.corr(method='pearson')

    target_stats = []
    for feat in feature_cols:
        r_val, p_val = stats.pearsonr(df_valid[feat], df_valid[target_col])
        rho_val, p_rho = stats.spearmanr(df_valid[feat], df_valid[target_col])
        cov_val = np.cov(df_valid[feat], df_valid[target_col])[0, 1]
        
        target_stats.append({
            'Feature': feat,
            'Pearson r': r_val,
            'p-value (Pearson)': p_val,
            'Spearman rho': rho_val,
            'p-value (Spearman)': p_rho,
            'Covariance': cov_val
        })

    target_summary_df = pd.DataFrame(target_stats).sort_values(
        by='Pearson r', ascending=False
    ).reset_index(drop=True)

    return corr_matrix, target_summary_df


# --- Publication Visualisation ---

def plot_covariance_analysis_figure(df, metals, target_col='log10_Cond80C',
                                   temp_label='80C',
                                   output_paths=None):
    """
    Generates publication Figure 1: Covariance network heatmap and target correlation profile.
    Strictly formatted to match SI_Figure_Regression_80C.pdf dimensions (11 x 5 in) and typography.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned HEMP data.
    metals : list of str
        List of metal symbols.
    target_col : str
        Target conductivity column name.
    temp_label : str
        Temperature label string for titles and axis annotations (e.g. '80C').
    output_paths : list of str, optional
        List of file paths to export figures to (.pdf and .png).
    """
    feature_cols = metals + ['Entropy (R)']
    analysis_cols = feature_cols + [target_col]
    target_clean_name = f'log10(Cond {temp_label})'
    
    # Clean subset
    df_valid = df[analysis_cols].dropna()
    
    # Compute correlation
    corr = df_valid.corr(method='pearson')
    
    # Custom display labels for the heatmap
    display_labels = metals + ['Entropy (R)', target_clean_name]
    corr_display = corr.copy()
    corr_display.columns = display_labels
    corr_display.index = display_labels

    # Initialize publication figure strictly with (11, 5) dimensions matching SI_Figure_Regression_80C.pdf
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # --- Subplot 1: Covariance Network Heatmap ---
    sns.heatmap(
        corr_display,
        ax=ax1,
        cmap='coolwarm',
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 7.5, 'family': 'Arial'},
        cbar_kws={'label': 'Pearson Correlation (r)', 'shrink': 0.8},
        square=True,
        linewidths=0.5,
        linecolor='white'
    )

    ax1.set_title(f'Covariance Network ({temp_label})')
    ax1.tick_params(axis='x', rotation=90, labelsize=9.5)
    ax1.tick_params(axis='y', rotation=0, labelsize=9.5)

    # --- Subplot 2: Target Correlation Profile ---
    target_series = corr_display[target_clean_name].drop(target_clean_name)
    target_sorted = target_series.sort_values(ascending=True)

    # Colour mapping strictly identical to RandomForestHEMPsv3.py: Blue (#2c7bb6) for positive, Red (#d7191c) for negative
    bar_colors = ['#d7191c' if v < 0 else '#2c7bb6' for v in target_sorted.values]
    
    bars = ax2.barh(
        target_sorted.index,
        target_sorted.values,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.8,
        height=0.6
    )

    ax2.axvline(0, color='black', linewidth=1.0)
    ax2.set_xlim(-0.45, 0.45)
    ax2.set_xlabel('Pearson Correlation (r)')
    ax2.set_title(f'Target Correlation ({temp_label})')
    ax2.grid(True, axis='both', linestyle='--', alpha=0.3)

    # Annotate bar values cleanly with Arial font
    for bar, val in zip(bars, target_sorted.values):
        offset = 0.015 if val >= 0 else -0.015
        ha = 'left' if val >= 0 else 'right'
        ax2.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2.0,
            f'{val:+.3f}',
            ha=ha,
            va='center',
            fontsize=9.5,
            family='Arial'
        )

    plt.tight_layout()

    if output_paths:
        for out_path in output_paths:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            plt.savefig(out_path, dpi=300)
            print(f"   [+] Saved figure to {out_path}")

    plt.close()


def plot_dual_temperature_comparison(df, metals, output_paths=None):
    """
    Generates a dual-temperature correlation comparison contrasting 
    elemental impacts at 25 °C vs. 80 °C. Formatted strictly with (11, 5) dimensions and Arial typography.
    """
    features = metals + ['Entropy (R)']
    
    df_25 = df[features + ['log10_Cond25C']].dropna()
    df_80 = df[features + ['log10_Cond80C']].dropna()

    corr_25 = df_25.corr()['log10_Cond25C'].drop('log10_Cond25C')
    corr_80 = df_80.corr()['log10_Cond80C'].drop('log10_Cond80C')

    comp_df = pd.DataFrame({
        '25 °C': corr_25,
        '80 °C': corr_80
    }).sort_values(by='80 °C', ascending=True)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    y = np.arange(len(comp_df))
    h = 0.35

    ax.barh(y - h/2, comp_df['25 °C'], height=h, label='25 °C', color='#2c7bb6', edgecolor='black', lw=0.8)
    ax.barh(y + h/2, comp_df['80 °C'], height=h, label='80 °C', color='#d7191c', edgecolor='black', lw=0.8)

    ax.axvline(0, color='black', lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(comp_df.index)
    ax.set_xlabel('Pearson Correlation (r)')
    ax.set_title('Thermal Regime Sensitivity (25C vs. 80C)')
    ax.legend(loc='lower right')
    ax.grid(True, axis='both', linestyle='--', alpha=0.3)
    ax.set_xlim(-0.45, 0.45)

    plt.tight_layout()
    if output_paths:
        for out_path in output_paths:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            plt.savefig(out_path, dpi=300)
            print(f"   [+] Saved dual-temperature figure to {out_path}")
    plt.close()


# --- Main Pipeline Execution ---

def run_covariance_pipeline(data_file='HEMPs.csv', output_dir=None):
    """
    Executes the complete covariance and solubility analysis pipeline.
    """
    print("=" * 70)
    print("   HEMPs MULTIVARIATE COVARIANCE & SOLUBILITY ANALYSIS PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1] Loading dataset from '{data_file}'...")
    df, metals = load_data(data_file)
    print(f"    Loaded {len(df)} records across cations: {', '.join(metals)}")

    # 2. Solubility Limits Analysis
    print("\n[2] Evaluating solid-solution solubility boundaries...")
    solubility_df = analyse_solubility_limits(df, metals)
    print("\n" + solubility_df.to_string(index=False))

    # 3. Covariance & Correlation Diagnostics (80 °C)
    print("\n[3] Computing compositional covariance and property correlations at 80 °C...")
    corr_matrix_80, target_stats_80 = compute_covariance_metrics(df, metals, target_col='log10_Cond80C')
    print("\nTarget Property Correlation Summary (80 °C):")
    print(target_stats_80.to_string(index=False))

    # Key physical metrics
    r_co_fe = corr_matrix_80.loc['Co', 'Fe']
    r_co_ni = corr_matrix_80.loc['Co', 'Ni']
    r_cu_cond = corr_matrix_80.loc['Cu', 'log10_Cond80C']
    r_ent_cond = corr_matrix_80.loc['Entropy (R)', 'log10_Cond80C']
    r_ent_mn = corr_matrix_80.loc['Entropy (R)', 'Mn']
    r_mn_cond = corr_matrix_80.loc['Mn', 'log10_Cond80C']

    print("\nKey Analytical Diagnostics:")
    print(f"   * Co <-> Fe Covariance (Closure Trade-off): r = {r_co_fe:+.2f}")
    print(f"   * Co <-> Ni Covariance (Closure Trade-off): r = {r_co_ni:+.2f}")
    print(f"   * Cu Target Correlation (Promoter):        r = {r_cu_cond:+.3f} (Solubility Limit: 28%)")
    print(f"   * Entropy Target Correlation:              r = {r_ent_cond:+.3f}")
    print(f"   * Entropy <-> Mn Confounding Coupling:     r = {r_ent_mn:+.2f} (Mn Target Correlation: {r_mn_cond:+.3f})")

    # 4. Generate Publication Figures
    print("\n[4] Generating publication-quality figures...")
    
    # Destination paths in output directory
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    fig_80_paths = [
        os.path.join(output_dir, 'SI_Figure_Covariance_Analysis.pdf'),
        os.path.join(output_dir, 'SI_Figure_Covariance_Analysis.png')
    ]
    
    plot_covariance_analysis_figure(
        df, metals, target_col='log10_Cond80C',
        temp_label='80C',
        output_paths=fig_80_paths
    )

    dual_paths = [
        os.path.join(output_dir, 'SI_Figure_Covariance_Analysis_Dual.pdf')
    ]
    plot_dual_temperature_comparison(df, metals, output_paths=dual_paths)

    # 5. Export Summary CSV Table
    summary_csv_path = os.path.join(output_dir, 'covariance_summary_table.csv')
    target_stats_80.to_csv(summary_csv_path, index=False)
    print(f"   [+] Exported summary statistics to {summary_csv_path}")

    print("\n" + "=" * 70)
    print("   COVARIANCE ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == '__main__':
    # When executed directly in codeForPublication directory
    run_covariance_pipeline('HEMPs.csv')
