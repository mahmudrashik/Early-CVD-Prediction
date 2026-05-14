# Structured Abstract

## Background
Public heart disease datasets are widely used for machine-learning demonstrations, but many analyses rely on random splits, duplicate-prone benchmark files, and incomplete leakage assessment.

## Objective
To develop a reproducible, leakage-audited early cardiovascular disease prediction benchmark using center-aware internal-external validation on UCI-style heart disease data.

## Methods
The primary dataset was `heart_disease_uci.csv` with four centers. The binary outcome was defined as `num > 0`. Candidate models included penalized logistic regression, random forest, gradient boosting, calibrated gradient boosting, and a feedforward multilayer perceptron comparator preserving the original notebook baseline. Preprocessing, imputation, scaling, and encoding were fitted inside resampling folds. Each center was held out once as an external-style validation fold.

## Results
The selected model was `logistic_l2`. Across held-out centers, mean AUROC was 0.799, mean AUPRC was 0.880, mean Brier score was 0.168, and mean balanced accuracy at threshold 0.50 was 0.745. These results should be interpreted as public-dataset benchmark performance, not clinical deployment evidence.

## Conclusions
The project provides a reproducible benchmark and localhost research prototype. Prospective validation, clinical workflow evaluation, and dataset modernization would be required before clinical use.
