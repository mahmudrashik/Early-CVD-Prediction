# Proposal Alignment

Project: Early CVD Prediction

## Re-evaluation Summary

The proposal describes an explainable early cardiovascular disease prediction project with a deep neural network baseline, classical model comparison, SHAP/LIME-style explanations, robustness checks, and a clinical decision-support style interface. The implemented repository matches that scientific direction, but it intentionally changes the dataset from the proposed Framingham dataset to the provided UCI-style heart disease files.

This dataset change is appropriate for the current submission because the supplied `heart_disease_uci.csv` includes hospital/center information, which supports stronger site-aware internal-external validation. The final project should therefore be presented as a public-dataset benchmark and research prototype, not as a direct Framingham 10-year CVD risk model.

## Proposal Item vs. Implemented Project

| Proposal requirement | Implemented status |
|---|---|
| Early cardiovascular disease prediction | Implemented as binary heart disease presence prediction. |
| Deep learning model | Preserved as `mlp_notebook_baseline`, but not selected as champion. |
| Classical baselines | Implemented: logistic regression, random forest, gradient boosting, calibrated gradient boosting. |
| Explainability | Implemented with global importance, SHAP-style summary, and local explanation records in the app. |
| Robust validation | Improved beyond proposal: leave-one-center-out internal-external validation. |
| Missing values and cleaning | Implemented with fold-contained imputation and data audit reports. |
| Duplicate handling | Implemented, especially for duplicate-heavy `heart.csv`. |
| Metrics | Implemented: AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1, MCC, Brier, calibration metrics, bootstrap CI, threshold analysis. |
| Local user interface | Implemented with FastAPI templates and one public prediction workflow. |

## Notebook Decision

The original neural-network notebook was audited because it shows the starting point of the work. It should not be used as the final scientific notebook because it used a different 303-row `heart.csv`, tuned with the test set as validation data, and did not perform site-aware validation.

The final notebook is `notebooks/Early_CVD_Prediction_ML_Lab.ipynb`. It is aligned with the report sections, reads the real project artifacts, presents the data audit, shows comparative and sensitivity studies, loads the final model bundle, and runs an individual explainable prediction.

## How to Present the Dataset Change

Recommended wording:

> The proposal initially planned to use the Framingham dataset. During implementation, the available project materials were the UCI-style heart disease dataset and a Kaggle-style `heart.csv`. I therefore used `heart_disease_uci.csv` as the primary dataset because it contains center/hospital identifiers, allowing a stronger site-aware validation design. The project goal remains explainable early CVD/heart disease prediction, but the final claims are limited to this public benchmark setting.
