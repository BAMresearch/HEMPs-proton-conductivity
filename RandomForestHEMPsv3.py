"""
RandomForest for HEMPs

This module implements a statistical analysis pipeline for evaluating
the structure-property relationships in High-Entropy Metal Phosphates (HEMPs).

Scientific Context:
    The script is designed to identify the primary drivers of proton conductivity
    in complex, multi-component solid solutions. It decouples the effects of specific
    chemical identity (e.g., cobalt content) from configurational entropy
    using non-linear ensemble learning and linear regularisation techniques.

Methodology:
    1.  **Data Preprocessing**: Handles multi-index CSV data, cleans non-numeric entries,
        and performs logarithmic transformation of conductivity values 
        to address the exponential nature of thermally activated transport.
    2.  **Ensemble Learning (Random Forest)**: Utilises a bootstrapped Random Forest Regressor
        to determine non-linear Feature Importance. Stability is ensured by aggregating
        results over 1000 randomized training/testing splits.
    3.  **Linear Regularisation (Ridge Regression)**: Uses L2-regularised linear regression
        on standardised features to determine the directional influence (positive/negative)
        of each descriptor.
    4.  **Unified Analysis**: restructuring data to treat Temperature as a feature,
        allowing for the validation of entropy effects across thermal regimes.
    5.  **Tree Visualisation**: Extracts and plots a single representative decision tree 
        from the ensemble to illustrate the non-linear feature splitting logic.

Dependencies:
    - **pandas**: Advanced data manipulation and DataFrame management.
    - **numpy**: Numerical operations and vectorization.
    - **scikit-learn**: Machine learning algorithms (RandomForest, Ridge),
      preprocessing (StandardScaler), model selection tools, and tree visualisation.
    - **joblib**: Parallel computing to accelerate the iterative bootstrapping process.
    - **matplotlib / seaborn**: Publication-quality data visualisation.

Author: Tomasz M. Stawski (tomasz.stawski@bam.de)
Date:   February 2026
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
from sklearn.tree import plot_tree  # Added for single tree visualisation
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import contextlib

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

def load_and_preprocess_data(file_path):
    """
    Loads raw experimental data from a CSV file and preprocesses it for machine learning analysis.

    This function handles hierarchical column structures (multi-indexing), cleans
    non-numeric placeholders (e.g., '-'), and computes logarithmic transformations
    of the target variable (conductivity).

    Parameters
    ----------
    file_path : str
        The relative or absolute path to the CSV file containing the HEMP dataset.
        Expected structure: Multi-level header where level 0 describes the category
        (e.g., 'Composition') and level 1 describes the specific element/property.

    Returns
    -------
    tuple (pd.DataFrame, list)
        - **df**: A cleaned, flat-index DataFrame containing numeric features and targets.
        - **metals**: A list of strings representing the chemical symbols of the
          cations included in the analysis.

    Raises
    ------
    FileNotFoundError
        If the specified file_path does not exist.
    """
    # Load data treating the first two rows as a MultiIndex header
    df_raw = pd.read_csv(file_path, header=[0, 1])
    df = pd.DataFrame()

    # Define the chemical species of interest
    metals = ['Ni', 'Co', 'Fe', 'Mn', 'Cd', 'Cu', 'Mg', 'Zn']

    # Iteratively extract metal compositions, handling missing or malformed data
    for metal in metals:
        try:
            # Accessing MultiIndex: Slice(None) selects all top-level keys
            col_data = df_raw.loc[:, (slice(None), metal)].iloc[:, 0]
            # Coerce errors to NaN, then fill with 0 (assumption: missing = 0% content)
            df[metal] = pd.to_numeric(col_data.replace('-', '0'), errors='coerce').fillna(0)
        except KeyError:
            # If a metal column is entirely missing from the CSV, initialise as 0
            df[metal] = 0

    # Extract Configurational Entropy (R)
    try:
        df['Entropy'] = pd.to_numeric(df_raw.loc[:, (slice(None), 'R')].iloc[:, 0], errors='coerce')
    except Exception:
        df['Entropy'] = 0

    # Extract Total Number of Cations (complexity metric)
    try:
        df['NumCations'] = df_raw.iloc[:, 0]
    except Exception:
        df['NumCations'] = 0

    # Process conductivity data for both experimental temperatures (25 °C and 80 °C)
    # Log-transform is critical for transport properties which span orders of magnitude.
    for temp_str, col_name in [('25 °C', 'Cond25C'), ('80 °C', 'Cond80C')]:
        try:
            raw_series = df_raw.loc[:, (slice(None), temp_str)].iloc[:, 0]
            df[col_name] = pd.to_numeric(raw_series, errors='coerce')
            # Calculate log10(sigma), handling potential zeros/negatives implicitly via NaNs
            df[f'log{col_name}'] = np.log10(df[col_name])
        except KeyError:
            pass  # Fail silently if temperature data is missing

    return df, metals


def prepare_combined_dataset(df, metals):
    """
    Restructures the dataset from 'wide' to 'long' format to incorporate
    Temperature as a predictive feature.

    This transformation enables the statistical models to learn the global influence
    of features (like entropy) across different thermal regimes, rather than treating
    each temperature as a separate dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed wide-format DataFrame containing 'logCond25C' and 'logCond80C'.
    metals : list of str
        List of metal cation column names.

    Returns
    -------
    pd.DataFrame
        A long-form DataFrame with a new 'Temperature' column and a single 'logCond' target.
        Rows with infinite or NaN conductivity values are filtered out.
    """
    # Prepare 25°C subset
    df_25 = df[metals + ['Entropy', 'NumCations', 'logCond25C']].copy()
    df_25['Temperature'] = 25
    df_25 = df_25.rename(columns={'logCond25C': 'logCond'})

    # Prepare 80°C subset
    df_80 = df[metals + ['Entropy', 'NumCations', 'logCond80C']].copy()
    df_80['Temperature'] = 80
    df_80 = df_80.rename(columns={'logCond80C': 'logCond'})

    # Concatenate subsets
    df_combined = pd.concat([df_25, df_80], axis=0).reset_index(drop=True)

    # Filter invalid data (inf/-inf/NaN) resulting from log(0) or missing measurements
    df_combined = df_combined[np.isfinite(df_combined['logCond'])]

    return df_combined


# --- Statistical Engines ---

def _train_single_model(X, y, random_state):
    """
    Helper function to train a single instance of a Random Forest Regressor.
    Designed to be executed in parallel via Joblib.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    random_state : int
        Seed for reproducibility of the specific iteration.

    Returns
    -------
    np.ndarray
        Feature importance scores for the trained model.
    """
    # Perform a random 80/20 train/test split for this specific iteration
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Initialise Random Forest Regressor:
    # n_estimators=100 achieves statistical stability while constraining computational requirements.
    # n_jobs=1 restricts parameter calculation structurally, relying entirely on external parallel processes.
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
    rf.fit(X_train, y_train)

    return rf.feature_importances_


def run_stability_analysis(X, y, features, iterations=1000):
    """
    Executes a bootstrapped Random Forest analysis to assess the stability of feature importance.

    Standard Feature Importance from a single run can be biased by the specific data split,
    especially with small datasets (N < 50). This function iterates the modeling process
    1000 times with different random seeds to generate a statistical distribution of importance.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    features : list
        List of feature names.
    iterations : int, optional
        Number of bootstrap iterations (default is 1000).

    Returns
    -------
    tuple (pd.Series, pd.Series)
        - **mean_imp**: The average feature importance score across all iterations.
        - **std_imp**: The standard deviation of the importance scores (error bars).
    """
    print(f"   ...Running Random Forest Stability Analysis ({iterations} iterations)...")

    # Use Joblib's Parallel to utilise all available CPU cores (n_jobs=-1)
    with tqdm_joblib(tqdm(desc="Bootstrapping", total=iterations)):
        importances = Parallel(n_jobs=-1, verbose=0)(
            delayed(_train_single_model)(X, y, i) for i in range(iterations)
        )

    # Aggregation
    importance_df = pd.DataFrame(importances, columns=features)
    mean_imp = importance_df.mean().sort_values(ascending=False)
    std_imp = importance_df.std().reindex(mean_imp.index)

    return mean_imp, std_imp


def run_ridge_analysis(X, y, features):
    """
    Performs Ridge Regression (L2 Regularisation) to determine feature directionality.

    While Random Forest provides magnitude of importance, it is non-parametric and does
    not explicitly provide sign (positive/negative influence). Ridge regression complements
    this by providing signed coefficients.

    Note: Features are standardised (Z-score normalization) prior to regression. This is
    mandatory for penalised regression to ensuring that coefficient magnitudes are directly
    comparable regardless of feature scale.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    features : list
        List of feature names.

    Returns
    -------
    pd.Series
        Sorted regression coefficients indexed by feature name.
    """
    print("   ...Running Ridge Regression for Directionality...")

    # Standardisation: (x - u) / s
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Alpha=1.0 is the regularisation strength. Higher values reduce variance but increase bias.
    ridge = Ridge(alpha=1.0).fit(X_scaled, y)

    return pd.Series(ridge.coef_, index=features).sort_values(ascending=False)


# --- Plotting Functions ---

def plot_standard_results(mean_imp, std_imp, ridge_coefs, df_subset, target_col, temp_name):
    """
    Generates and saves diagnostic plots for a single-temperature analysis.

    Produces two files:
    1. **Regression Analysis**: Bar charts of Random Forest Importance and Ridge Coefficients.
    2. **Hypothesis Testing**: Scatter plots correlating Entropy and Cobalt content with Conductivity.

    Parameters
    ----------
    mean_imp : pd.Series
        Mean feature importance from RF.
    std_imp : pd.Series
        Standard deviation of importance.
    ridge_coefs : pd.Series
        Coefficients from Ridge regression.
    df_subset : pd.DataFrame
        The data subset used for the analysis.
    target_col : str
        The name of the target column (e.g., 'logCond80C').
    temp_name : str
        Label for the analysis (e.g., '80C') used in titles and filenames.
    """
    # --- Figure 1: Statistical Model Outputs ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # Subplot 1: Random Forest Feature Importance with Error Bars
    mean_imp.plot(kind='bar', yerr=std_imp, capsize=3, ax=ax1, color='#2c7bb6', edgecolor='black', lw=0.8)
    ax1.set_title(f'Feature Importance ({temp_name})')
    ax1.set_ylabel('Mean Importance Score')
    ax1.grid(True, axis='both', linestyle='--', alpha=0.3)

    # Subplot 2: Ridge Coefficients (Directionality)
    ridge_coefs.plot(kind='barh', ax=ax2, color='#d7191c', edgecolor='black', lw=0.8)
    ax2.set_title(f'Directionality ({temp_name})')
    ax2.set_xlabel('Standardised Coefficient Magnitude')
    ax2.axvline(0, color='black', lw=1)
    ax2.grid(True, axis='both', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'SI_Figure_Regression_{temp_name}.pdf')
    plt.close()

    # --- Figure 2: Physical Hypothesis Visualisation ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # Subplot 3: Conductivity vs. Entropy (Size=Complexity, Colour=Cobalt)
    scatter = ax3.scatter(df_subset['Entropy'], df_subset[target_col],
                          c=df_subset['Co'], cmap='viridis', s=df_subset['NumCations']*30,
                          alpha=0.85, edgecolors='black', lw=0.5)
    ax3.set_xlabel(r'Configurational Entropy / $R$')
    ax3.set_ylabel(r'$\log_{10}(\sigma / S \cdot cm^{-1})$')
    ax3.set_title(f'Performance vs. Complexity ({temp_name})')
    ax3.grid(True, axis='both', linestyle='--', alpha=0.3)
    plt.colorbar(scatter, ax=ax3).set_label('Cobalt Content / %', rotation=270, labelpad=15)

    # Subplot 4: Conductivity vs. Cobalt Content (Linear Trend)
    sns.regplot(data=df_subset, x='Co', y=target_col, ax=ax4,
                scatter_kws={'alpha':0.4, 'color':'gray', 's':40},
                line_kws={'color':'#e66101', 'label':'Trend'})
    ax4.set_xlabel('Cobalt Content / %')
    ax4.set_ylabel(r'$\log_{10}(\sigma / S \cdot cm^{-1})$')
    ax4.set_title(f'Cobalt Correlation ({temp_name})')
    ax4.grid(True, axis='both', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'SI_Figure_Hypothesis_{temp_name}.pdf')
    plt.close()


def plot_combined_results(mean_imp, std_imp, ridge_coefs, df_combined):
    """
    Generates plots for the combined analysis where Temperature is treated as a feature.
    """
    # --- Figure 1: Regression Summary (Combined) ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # Conditional colouring: Highlight 'Temperature' feature to distinguish from chemical features
    colours = ['#2c7bb6' if 'Temp' not in idx else '#e66101' for idx in mean_imp.index]
    mean_imp.plot(kind='bar', yerr=std_imp, capsize=3, ax=ax1, color=colours, edgecolor='black', lw=0.8)
    ax1.set_title('Feature Importance (Combined)')
    ax1.set_ylabel('Mean Importance Score')
    ax1.grid(True, axis='both', linestyle='--', alpha=0.3)

    colors_ridge = ['#d7191c' if 'Temp' not in idx else '#e66101' for idx in ridge_coefs.index]
    ridge_coefs.plot(kind='barh', ax=ax2, color=colors_ridge, edgecolor='black', lw=0.8)
    ax2.set_title('Directionality (Combined)')
    ax2.set_xlabel('Standardised Coefficient Magnitude')
    ax2.axvline(0, color='black', lw=1)
    ax2.grid(True, axis='both', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('SI_Figure_Regression_Combined.pdf')
    plt.close()

    # --- Figure 2: Hypothesis Visualisation (Split by Temperature) ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    # Subplot 3: Entropy Impact separated by Temperature regime
    sns.scatterplot(data=df_combined, x='Entropy', y='logCond', hue='Temperature',
                    palette='coolwarm', style='Temperature', s=80, alpha=0.8, ax=ax3)
    ax3.set_xlabel(r'Configurational Entropy / $R$')
    ax3.set_ylabel(r'$\log_{10}(\sigma / S \cdot cm^{-1})$')
    ax3.set_title('Entropy Impact (Split by Temp)')
    ax3.grid(True, axis='both', linestyle='--', alpha=0.3)
    ax3.legend(title='Temp / °C')

    # Subplot 4: Cobalt Impact separated by Temperature regime with manual regression lines
    sns.scatterplot(data=df_combined, x='Co', y='logCond', hue='Temperature',
                    palette='coolwarm', alpha=0.4, s=50, ax=ax4, legend=False)

    # Overlay regression lines manually to ensure clear visual distinction
    sns.regplot(data=df_combined[df_combined['Temperature']==25], x='Co', y='logCond',
                scatter=False, ax=ax4, color='blue', label='25 °C Trend')
    sns.regplot(data=df_combined[df_combined['Temperature']==80], x='Co', y='logCond',
                scatter=False, ax=ax4, color='red', label='80 °C Trend')

    ax4.set_xlabel('Cobalt Content / %')
    ax4.set_ylabel(r'$\log_{10}(\sigma / S \cdot cm^{-1})$')
    ax4.set_title('Cobalt Impact (Combined)')
    ax4.grid(True, axis='both', linestyle='--', alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('SI_Figure_Hypothesis_Combined.pdf')
    plt.close()


def plot_representative_tree(X, y, features, temp_name):
    """
    Trains a representative Random Forest model and plots a single decision tree
    from the ensemble to visualise the non-linear splitting rules.
    """
    print(f"   ...Generating representative Decision Tree visualisation for {temp_name}...")
    
    # Train a single standard model (no need to iterate 1000 times for this)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
    rf.fit(X_train, y_train)

    # Extract one tree from the forest with a more balanced root split
    single_tree = rf.estimators_[73]

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 16), dpi=300)
    annotations = plot_tree(single_tree, 
              feature_names=features, 
              filled=True, 
              rounded=True, 
              max_depth=4,  # Truncates render depth to maintain annotation legibility
              precision=3, 
              fontsize=16,
              ax=ax)
              
    import matplotlib.colors as mcolors
    root_x, root_y = annotations[0].get_position()
    
    # Extract values for colour normalization
    values = []
    for ann in annotations:
        text = ann.get_text()
        if "value = " in text:
            try:
                val_str = text.split("value = ")[1].split("\n")[0]
                values.append(float(val_str))
            except Exception:
                pass
                
    if values:
        norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
        # Apply standard continuous colormap ('coolwarm') for node value visualisation
        cmap = plt.get_cmap('coolwarm')
        
    for ann in annotations:
        text = ann.get_text()
        
        # Substitute 'squared_error' string with corresponding mathematical notation \epsilon^2
        if "squared_error" in text:
            text = text.replace("squared_error", r"$\epsilon^2$")
            
        # Relocate binary split annotations linearly above respective connecting edges
        if text.strip() in ["True", "False"]:
            curr_x, curr_y = ann.get_position()
            # Move towards the root (set fixed high Y)
            new_y = root_y - 0.05
            # Shift slightly inwards for centering
            new_x = curr_x + (root_x - curr_x) * 0.15
            ann.set_position((new_x, new_y))
            ann.set_weight('bold')
            ann.set_fontsize(20)
            
            # Make sure labels are drawn on top of the node boxes and lines
            ann.set_zorder(100)
            ann.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
            
        ann.set_text(text)
        
        # Allocate diverging node shading corresponding directly to split criteria values
        if "value = " in text and values:
            try:
                val = float(text.split("value = ")[1].split("\n")[0])
                colour = cmap(norm(val))
                bbox = ann.get_bbox_patch()
                if bbox is not None:
                    bbox.set_facecolor(colour)
                    bbox.set_alpha(0.85)
            except Exception:
                pass

    plt.title(f"Representative Decision Tree (Random Forest Ensemble) - {temp_name}", fontsize=18)
    plt.tight_layout()
    plt.savefig(f'SI_Figure_Single_Tree_Visualization_{temp_name}.pdf', format='pdf')
    plt.close()


# --- Main Execution Pipeline ---

def run_analysis_pipeline():
    """
    Main controller function that performs the data loading, processing,
    and analysis workflows.
    """
    # Attempt to load the dataset
    try:
        df, metals = load_and_preprocess_data('HEMPs.csv')
    except FileNotFoundError:
        print("Critical Error: Dataset not found in the working directory.")
        return

    # --- Phase 1: Analysis at 25°C ---
    if 'logCond25C' in df.columns:
        print("\n--- Initiating Analysis: 25°C Regime ---")
        mask = np.isfinite(df['logCond25C'])
        df_25 = df[mask].copy()
        
        # Ensure sufficient sample size for regression
        if len(df_25) > 5:
            X = df_25[metals + ['Entropy']]
            y = df_25['logCond25C']
            
            # Execute Statistical Engines
            imp, std = run_stability_analysis(X, y, metals + ['Entropy'])
            coefs = run_ridge_analysis(X, y, metals + ['Entropy'])
            
            # Generate Visualisations
            plot_standard_results(imp, std, coefs, df_25, 'logCond25C', '25C')
        else:
            print("Warning: Insufficient data points for 25°C analysis.")

    # --- Phase 2: Analysis at 80°C ---
    if 'logCond80C' in df.columns:
        print("\n--- Initiating Analysis: 80°C Regime ---")
        mask = np.isfinite(df['logCond80C'])
        df_80 = df[mask].copy()
        
        if len(df_80) > 5:
            X = df_80[metals + ['Entropy']]
            y = df_80['logCond80C']
            
            imp, std = run_stability_analysis(X, y, metals + ['Entropy'])
            coefs = run_ridge_analysis(X, y, metals + ['Entropy'])
            plot_standard_results(imp, std, coefs, df_80, 'logCond80C', '80C')
            
            # Generate the single tree visualisation for 80C
            plot_representative_tree(X, y, metals + ['Entropy'], '80C')
        else:
            print("Warning: Insufficient data points for 80°C analysis.")

    # --- Phase 3: Combined Temperature Analysis ---
    print("\n--- Initiating Analysis: Combined Thermal Regimes ---")
    df_combined = prepare_combined_dataset(df, metals)
    
    if len(df_combined) > 10:
        features = metals + ['Entropy', 'Temperature']
        X = df_combined[features]
        y = df_combined['logCond']
        
        imp, std = run_stability_analysis(X, y, features)
        coefs = run_ridge_analysis(X, y, features)
        plot_combined_results(imp, std, coefs, df_combined)
        print("Combined analysis pipeline completed successfully.")
    else:
        print("Warning: Not enough aggregate data for combined analysis.")

if __name__ == "__main__":
    run_analysis_pipeline()