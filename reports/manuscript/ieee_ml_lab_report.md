# Early CVD Prediction: A Leakage-Audited Machine Learning Project for Heart Disease Prediction

Mahmud Rashik  
Department of Computer Science and Engineering  
Bangladesh University of Professionals  
Dhaka, Bangladesh

## Abstract

Heart disease prediction is a common machine-learning laboratory problem, but public benchmark datasets can produce misleading results when duplicates, leakage, validation design, and calibration are not handled carefully. This project develops a reproducible early cardiovascular disease prediction workflow using the supplied UCI-style heart disease dataset and a secondary `heart.csv` benchmark. The main binary outcome was disease absence for `num = 0` and disease presence for `num > 0`. The primary dataset contained 920 records from 4 centers; the secondary `heart.csv` file contained 1025 rows but only 302 unique rows after exact deduplication. The main validation used leave-one-center-out internal-external validation. Penalized logistic regression, random forest, gradient boosting, calibrated gradient boosting, and an MLP baseline were compared. Penalized logistic regression was selected because it provided the best overall balance of calibration, interpretability, and site-aware validation performance. It achieved mean held-out-center AUROC 0.799, AUPRC 0.880, Brier score 0.168, and balanced accuracy 0.745. The project includes a notebook, generated figures and tables, tests, and a FastAPI localhost application with explanation output. The system is a research and educational prototype, not a clinical diagnostic device.

**Keywords**: cardiovascular disease prediction, heart disease, machine learning, calibration, explainable AI, site-aware validation

## I. Introduction

Early recognition of cardiovascular disease risk is clinically important and is also a practical machine-learning task because the available features are understandable: age, chest pain type, resting blood pressure, cholesterol, exercise-induced angina, ST depression, ST slope, major vessels, and thalassemia/stress-test category. The starting point of this work was an exploratory neural-network notebook. After auditing the notebook and datasets, the project was rebuilt as a reproducible prediction pipeline because the original workflow used a single random split and a different small `heart.csv` file.

The contribution of this project is a compact but complete ML workflow: data audit, duplicate analysis, leakage control, fold-contained preprocessing, site-aware validation, model comparison, calibration-aware model selection, explainability, and localhost deployment.

## II. Literature Review

Clinical prediction models should be judged by discrimination, calibration, stability, and interpretability rather than accuracy alone. TRIPOD+AI emphasizes transparent reporting of data sources, preprocessing, validation, and limitations for prediction models [1]. PROBAST+AI highlights risk of bias and applicability concerns in AI prediction studies [2]. Public UCI-style heart disease data are useful for reproducible learning, but they are small, historical, and not enough to claim clinical deployment readiness [3]. Explainability methods such as coefficient-based importance and SHAP-style summaries help check whether model behavior is plausible, while still avoiding causal claims [5].

## III. Methodology

The primary dataset was `heart_disease_uci.csv`. The target was converted to a binary label: `0` for no recorded disease and `1` for disease when `num > 0`. The center/source variable was used for validation grouping and was not used as a predictor. The secondary `heart.csv` file was treated as a supplementary benchmark only because it contained 723 exact duplicate rows.

Numeric predictors were median-imputed and standardized. Categorical predictors were most-frequent-imputed and one-hot encoded. All preprocessing was placed inside scikit-learn pipelines, so imputation, scaling, and encoding were fitted only on training folds. The main validation was leave-one-center-out internal-external validation: one center was held out, models were selected using the remaining centers, and performance was measured on the held-out center. This process was repeated for all centers.

**Table I. Dataset Audit Summary**

| Dataset | Role | Rows | Key finding |
|---|---:|---:|---|
| heart_disease_uci.csv | Primary | 920 | 4 centers; 411 absent and 509 present cases |
| heart.csv | Supplementary | 1025 | 723 exact duplicates; 302 unique rows after deduplication |

Missingness was substantial for several predictors, especially ca 66.4%, thal 52.8%, slope 33.6%, fbs 9.8%, oldpeak 6.7%. This supported the decision to keep imputation inside the validation pipeline.

## IV. Experimentation

Five model families were compared: penalized logistic regression, random forest, gradient boosting, calibrated gradient boosting, and an MLP baseline derived from the earlier neural-network direction. Hyperparameter selection was performed only within the development centers for each outer fold. Evaluation included AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, precision, NPV, F1 score, MCC, Brier score, calibration slope/intercept, bootstrap confidence intervals, ROC curves, PR curves, calibration plots, decision curve analysis, threshold analysis, and explanation outputs.

## V. Results and Discussion

**Table II. Primary Model Comparison**

| Model | AUROC | AUPRC | Brier | Balanced Acc. | MCC |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.799 | 0.880 | 0.168 | 0.745 | 0.441 |
| Calibrated gradient boosting | 0.809 | 0.877 | 0.174 | 0.714 | 0.385 |
| Random forest | 0.800 | 0.873 | 0.170 | 0.716 | 0.387 |
| Gradient boosting | 0.793 | 0.861 | 0.180 | 0.732 | 0.410 |
| MLP baseline | 0.779 | 0.858 | 0.187 | 0.678 | 0.342 |

Calibrated gradient boosting had the highest mean AUROC, but penalized logistic regression was selected because it had the best overall ranking after considering AUPRC, Brier score, MCC, calibration slope, stability, and interpretability. For pooled held-out-center predictions, logistic regression achieved AUROC 0.855 (95% bootstrap CI 0.828-0.878), AUPRC 0.864 (0.825-0.893), Brier score 0.159 (0.145-0.173), and balanced accuracy 0.785 (0.756-0.811).

**Fig. 1. Model comparison across center-aware validation folds.**

![Model comparison](../figures/model_comparison.png)

**Fig. 2. ROC curves for compared models.**

![ROC curves](../figures/roc_curves.png)

**Fig. 3. Calibration curves for compared models.**

![Calibration plots](../figures/calibration_plots.png)

At threshold 0.50, the selected model reached sensitivity 0.729, specificity 0.842, PPV 0.851, NPV 0.715, and MCC 0.568. Threshold 0.30 increased sensitivity to 0.876 but reduced specificity to 0.637; threshold 0.70 increased specificity to 0.925 but reduced sensitivity to 0.525. Therefore the app reports probability and risk category rather than claiming one universal clinical threshold.

The strongest global predictors included cp, sex, exang, thal, slope, oldpeak. These variables are clinically plausible for a benchmark heart disease task, but the model should not be interpreted causally.

**Fig. 4. Global feature importance for the selected model.**

![Global feature importance](../figures/global_feature_importance.png)

## VI. Ablation Studies / Comparative Studies

The main comparative study was the model comparison in Table II. It showed that the MLP baseline was not the strongest choice for this small tabular dataset; the simpler logistic regression model was better calibrated and easier to interpret.

The duplicate sensitivity study used the secondary `heart.csv` file. The duplicate-expanded version produced higher apparent performance than the deduplicated version, confirming that duplicate rows can inflate benchmark results.

**Table III. Duplicate Sensitivity on Secondary Dataset**

| Dataset version | AUROC | AUPRC | Balanced Acc. | Brier |
|---|---:|---:|---:|---:|
| Deduplicated heart.csv | 0.909 | 0.917 | 0.832 | 0.125 |
| Duplicate-expanded heart.csv | 0.924 | 0.926 | 0.855 | 0.110 |

## VII. Localhost Application

The final model is served through a FastAPI application. The interface has one prediction workflow with a clinical-style input form, probability output, risk category, explanation panel, model metadata, and a research-prototype disclaimer. The app uses the saved preprocessing and model bundle, so predictions in the interface match the trained artifact.

## VIII. Conclusion and Future Work

The project satisfies the ML Lab requirements by combining a clean notebook, reproducible model training, comparative experiments, ablation/sensitivity analysis, figures, tables, and a localhost application. Penalized logistic regression was selected because it had the best balance of calibration, interpretability, and center-aware validation performance. Future work should use larger modern cohorts, independent external validation, fairness analysis, improved missingness modeling, and clinical usability testing. The current project should be presented as a research and educational prototype only.

## References

[1] G. S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," BMJ, 2024.  
[2] G. S. Collins et al., "PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods," BMJ, 2025.  
[3] R. Detrano et al., "International application of a new probability algorithm for the diagnosis of coronary artery disease," American Journal of Cardiology, 1989.  
[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, 2011.  
[5] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS, 2017.
