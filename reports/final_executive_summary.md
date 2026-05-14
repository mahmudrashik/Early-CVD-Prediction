# Final Executive Summary

The selected model for the primary UCI site-aware analysis is `logistic_l2`. It was chosen because its overall ranking balanced discrimination, calibration, stability across held-out centers, and interpretability rather than optimizing test accuracy alone.

The original notebook's neural-network work has been preserved as a comparator. Its reported tuned-model accuracy is not treated as final evidence because the notebook used a different 303-row file and used the test split during tuning.

Primary aggregate held-out-center performance for `logistic_l2`:

- AUROC: 0.799
- AUPRC: 0.880
- Brier score: 0.168
- Balanced accuracy: 0.745

The localhost application serves this selected model artifact and provides calibrated probability, risk category, and local explanation text. It includes a research-prototype disclaimer.
