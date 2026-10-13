# High-Entropy Metal Phosphates (HEMPs) Statistical Analysis

This repository contains data analysis scripts and datasets used to evaluate structure–property relationships in High-Entropy Metal Phosphates (HEMPs), focusing on proton conductivity, elemental partitioning, and compositional covariance.

---

## Overview

We use Random Forest regression and Ridge regression to evaluate the factors influencing proton conductivity in multicomponent solid solutions. By including configurational entropy ($R$) alongside elemental concentrations, the models assess the impact of composition and entropy on conductivity at 25 °C and 80 °C under 98% relative humidity (RH).

In addition, the repository provides analysis of:
1. **Precursor partitioning**: Difference between solid and solution concentrations ($\Delta X = X_{\text{solid}} - X_{\text{solution}}$) and Leave-Self-Out regressions to evaluate cross-element interactions during synthesis.
2. **Compositional covariance and solubility limits**: Constant-sum closure constraints ($\sum X = 100\%$), solid solubility limits (e.g. Cu limit at ~28%), and correlation profiles with conductivity.

---

## Repository Organisation

```
codeForPublication/
├── HEMPs.csv                               # Experimental dataset (25 °C and 80 °C)
├── HEMPsExtended.csv                       # Solution vs solid concentration dataset
├── RandomForestHEMPsv3.py                  # Random Forest & Ridge regression models
├── RandomForestHEMPsv4.py                  # Partitioning (ΔX) & cross-interaction models
├── CovarianceAnalysis.py                   # Covariance matrix & solubility limits analysis
├── inspect_trees.py                        # Tree split balance inspection
├── inspect_trees_2.py                      # Tree split balance inspection via child node indices
├── inspect_tree_0.py                       # Tree depth and node counts
├── inspect_tree_73.py                      # Text export of decision rules for tree index 73
├── output/                                 # Generated figures and CSV tables
│   ├── SI_Figure_Regression_25C.pdf
│   ├── SI_Figure_Hypothesis_25C.pdf
│   ├── SI_Figure_Regression_80C.pdf
│   ├── SI_Figure_Hypothesis_80C.pdf
│   ├── SI_Figure_Regression_Combined.pdf
│   ├── SI_Figure_Hypothesis_Combined.pdf
│   ├── SI_Figure_Single_Tree_Visualization_80C.pdf
│   ├── SI_Figure_Partitioning.pdf
│   ├── SI_Figure_Cross_Interactions.pdf
│   ├── SI_Figure_Covariance_Analysis.pdf
│   ├── SI_Figure_Covariance_Analysis.png
│   ├── SI_Figure_Covariance_Analysis_Dual.pdf
│   └── covariance_summary_table.csv
├── LICENSE                                 # MIT License
└── RELEASE.md                              # Release notes
```

---

## Datasets

* **`HEMPs.csv`**  
  Experimental dataset containing elemental compositions, configurational entropy ($R$), total cation count, and conductivities at 25 °C and 80 °C (98% RH) across 39 samples.
* **`HEMPsExtended.csv`**  
  Dataset pairing precursor solution concentrations directly with crystallised solid-phase concentrations.

---

## Python Scripts

* **`RandomForestHEMPsv3.py`**  
  Fits Random Forest regressors (1,000 bootstrap iterations) and Ridge regression to evaluate feature importances and directionalities at 25 °C, 80 °C, and across combined temperatures. Extracts and visualises representative decision tree index 73.
* **`RandomForestHEMPsv4.py`**  
  Evaluates liquid-to-solid partitioning ($\Delta X$) and fits Leave-Self-Out regressions to evaluate inter-element cross-interactions during crystallisation. Integrates the covariance analysis pipeline.
* **`CovarianceAnalysis.py`**  
  Calculates compositional covariance matrices, evaluates solid-solution solubility boundaries (e.g. Cu verge at ~28%), and examines linear correlations with $\log_{10}(\sigma)$ at 25 °C and 80 °C.

---

## Tree Inspection Utilities

* **`inspect_trees.py`**: Evaluates root split sample balance across Random Forest trees to identify a balanced representative tree.
* **`inspect_trees_2.py`**: Evaluates root split balance using child node pointers (`children_left` and `children_right`).
* **`inspect_tree_0.py`**: Reports tree depth and node counts for trees 0, 2, and 4.
* **`inspect_tree_73.py`**: Exports decision rules for representative tree index 73 up to depth 4.

---

## Generated Visualisations

Figures are saved directly in `output/`:
* `SI_Figure_Regression_*.pdf`: Feature importances (Random Forest) and coefficients (Ridge regression).
* `SI_Figure_Hypothesis_*.pdf`: Scatter plots of conductivity vs configurational entropy.
* `SI_Figure_Single_Tree_Visualization_80C.pdf`: Decision tree structure for tree index 73.
* `SI_Figure_Cross_Interactions.pdf`: Cross-interaction models showing element effects on solid composition.
* `SI_Figure_Partitioning.pdf`: Boxplots of elemental partitioning ($\Delta X$).
* `SI_Figure_Covariance_Analysis.pdf` / `.png`: Compositional covariance matrix and correlation profile (80 °C).
* `SI_Figure_Covariance_Analysis_Dual.pdf`: Dual-temperature correlation comparison (25 °C vs 80 °C).
* `covariance_summary_table.csv`: Statistical summary of covariance and solubility limits.

---

## Requirements

Python 3.8+ with standard scientific packages:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib tqdm
```

---

## Usage

Run the scripts from the repository root:
```bash
# Random Forest and Ridge regression (25 °C, 80 °C, and combined)
python RandomForestHEMPsv3.py

# Partitioning and cross-interaction analysis
python RandomForestHEMPsv4.py

# Covariance and solubility limits analysis
python CovarianceAnalysis.py
```

---

## Author

**Tomasz M. Stawski**  
Federal Institute for Materials Research and Testing (BAM), Berlin, Germany  
Email: tomasz.stawski@bam.de

---

## License

MIT License. See [LICENSE](LICENSE) for details.
