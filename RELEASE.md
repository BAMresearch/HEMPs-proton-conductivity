# Release 0.2

This release updates the statistical analysis pipelines and project organisation for High-Entropy Metal Phosphates (HEMPs).

### Summary of Changes
* **Multivariate Covariance Analysis (`CovarianceAnalysis.py`)**:
  * Added compositional covariance analysis to evaluate constant-sum closure effects ($\sum X = 100\%$) across the 8-metal system.
  * Evaluates solid-solution solubility boundaries (e.g. copper solubility limit at ~28%).
  * Calculates correlation profiles with conductivity at 25 °C and 80 °C.
* **Integrated Analysis Pipeline (`RandomForestHEMPsv4.py`)**:
  * Integrates covariance analysis with precursor-to-solid partitioning ($\Delta X$) and Leave-Self-Out cross-interaction models.
* **Output Organisation**:
  * All generated publication figures (PDF and PNG) and CSV summary tables are now saved into a dedicated `output/` directory.
* **Code & Documentation Clean-up**:
  * Simplified docstrings and comments across all scripts.
  * Neutral scientific style with British spelling throughout.

---

**Author:** Tomasz M. Stawski (tomasz.stawski@bam.de)  
Federal Institute for Materials Research and Testing (BAM), Berlin, Germany
