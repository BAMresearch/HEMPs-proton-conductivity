# High-Entropy Metal Phosphates (HEMPs) Statistical Analysis

This repository contains the statistical and empirical data analysis pipelines used to evaluate the structure-property relationships within High-Entropy Metal Phosphates (HEMPs), specifically focusing on proton conductivity and elemental disproportionation.

## Scientific Context

This project utilises advanced machine learning techniques—primarily bootstrapped Random Forest Ensembles and L2-Regularised Ridge Regression—to decouple the fundamental drivers of proton conductivity in complex, multi-component solid solutions. By incorporating both configurational entropy algorithms and individual chemical constituents, the codebase identifies the fundamental synergistic and antagonistic elemental interactions governing HEMP behaviour across various thermal regimes limit constraints.

---

## Repository Contents

### Primary Datasets
* **`HEMPs.csv`**
  The initial quantitative experimental dataset containing starting compositions, configurational entropy calculations ($R$), total cation presence, and empirically measured conductivities evaluated at 25°C and 80°C.
* **`HEMPsExtended.csv`**
  An expanded dataset detailing liquid-to-solid partitioning profiles matching the initial precursor solution parameters against the ultimately crystallized material compositions.

### Core Analytical Pipelines
* **`RandomForestHEMPsv3.py`**
  *The core structural property investigation script.* It trains bootstrapped decision forest models on both singular and combined thermal regimes (25°C and 80°C) to isolate non-linear feature importance scores, whilst deploying aligned Ridge Regressions to categorise those influences computationally as positively or negatively correlated. Generates comprehensive visual output summarising the predictive hierarchy.
* **`RandomForestHEMPsv4.py`**
  *The Disproportionation & Partitioning module.* Expands upon the principal analytics by establishing predictive models regarding elemental liquid-to-solid integrations. It quantifies component differentials ($\Delta X$) through Leave-Self-Out (LSO) regressions to map how extraneous elements synergistically or antagonistically govern solid incorporation.

### Ensemble Structural Inspection Tools
These utility scripts conduct deep architectural assessments concerning the topology, node partitioning, and structural viability of randomly generated decision trees inside the primary ensemble targeting 80°C conductivity metrics. 

* **`inspect_trees.py`**
  Iterates broadly through generated tree ensembles to assess primary initial split parity across root nodes, aiding in identifying structurally representative standard trees.
* **`inspect_trees_2.py`**
  Detailed topological parser validating root node balance dynamically by tracing the discrete decision graphs (examining internal `children_left` and `children_right` matrices).
* **`inspect_tree_0.py`**
  Structural probe diagnosing the architectural total depth limits and node counts for a pre-selected sample of estimator variants (specifically trees 0, 2, and 4).
* **`inspect_tree_73.py`**
  Extracts the absolute textual translation depicting the internal branching logic parameters governing optimal tree variant index 73.

### Generated Visualisations (Output)
The analytical pipelines automatically formulate scientific graphs mapping findings into scalable vector formats (`.pdf`):
* `SI_Figure_Regression_*.pdf`: Component evaluations (Importance vs. Directionality).
* `SI_Figure_Hypothesis_*.pdf`: Scatter correlations contrasting complexity (Entropy) directly against empirical attributes.
* `SI_Figure_Single_Tree_Visualization_80C.pdf`: A fully-rendered diagram detailing a representative non-linear algorithmic decision pathway.
* `SI_Figure_Cross_Interactions.pdf`: Ridge mappings delineating inter-element relationships.
* `SI_Figure_Partitioning.pdf`: Standard elemental distributions describing material incorporation variance.

---

## Dependencies
The analytical scripts rely exclusively upon standard scientific Python configurations.
* **Python 3.8+**
* `pandas`, `numpy` (Memory manipulation and array handling)
* `scikit-learn` (Primary ensemble regression, preprocessing scaling, and graphing limits)
* `matplotlib`, `seaborn` (Graphical visualisation)
* `joblib`, `tqdm` (Multithreading process parallelization and console progression reporting)

## Usage

Executing the fundamental evaluation paradigms automatically extracts raw data, computes models up to predefined 1000-iteration stabilities, and exports visual figures to the current working environment. 

```bash
python RandomForestHEMPsv3.py
python RandomForestHEMPsv4.py
```

## Author
**Tomasz Stawski**  
Email: tomasz.stawski@bam.de

## License

Copyright (c) 2026 Federal Institute for Materials Research and Testing (BAM)

This project is open-sourced under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
