# Early CVD Prediction: A Leakage-Audited and Site-Aware Machine Learning Study on Public Heart Disease Data

Mahmud Rashik  
Department of Computer Science and Engineering  
Bangladesh University of Professionals  
Dhaka, Bangladesh

## Abstract

Cardiovascular disease prediction is a common machine-learning laboratory problem, but many public-dataset studies rely on random train-test splits, duplicate-prone benchmark files, and limited calibration assessment. This project develops a reproducible early cardiovascular disease prediction system using public heart disease data while preserving and improving an existing neural-network notebook. The primary dataset was the UCI-style `heart_disease_uci.csv` file containing 920 records from four centers. The binary target was defined as absence of disease for `num = 0` and presence of disease for `num > 0`. The supplementary `heart.csv` file was audited separately and found to contain 1,025 rows with 723 exact duplicates, leaving 302 unique rows. The main validation design used leave-one-center-out internal-external validation. Candidate models included penalized logistic regression, random forest, gradient boosting, calibrated gradient boosting, and a multilayer perceptron baseline derived from the original notebook. The selected model was penalized logistic regression, which achieved mean held-out-center AUROC of 0.799, AUPRC of 0.880, Brier score of 0.168, and balanced accuracy of 0.745. The project also includes explainable prediction output and a FastAPI localhost application. The system is intended as a reproducible research and educational prototype, not as a clinically deployable diagnostic device.

**Keywords**: cardiovascular disease prediction, heart disease, machine learning, calibration, explainable AI, site-aware validation

## I. Introduction

Early identification of cardiovascular disease risk is an important healthcare problem and a useful applied machine-learning task. Public heart disease datasets are frequently used for student projects and benchmark experiments because they contain interpretable clinical variables such as age, chest pain type, resting blood pressure, cholesterol, exercise-induced angina, ST depression, and thalassemia-related test findings. However, simple machine-learning analyses can easily overstate performance if dataset duplication, preprocessing leakage, target definition, and validation design are not handled carefully.

The starting point of this work was an existing Jupyter notebook that trained feedforward neural networks for heart disease prediction. The notebook was useful as exploratory work, but it depended on a local 303-row `heart.csv` file and used a single random train-test split. The currently uploaded `heart.csv` was later found to contain extensive duplicate rows. Therefore, the final project reframed the work as a reproducible prediction-model study rather than a neural-network-only exercise.

This project has three goals. First, it audits the original notebook and datasets to identify reusable work and leakage risks. Second, it builds a clean machine-learning pipeline using center-aware validation on the UCI-style dataset. Third, it provides a localhost prediction application with transparent output and a research-prototype disclaimer.

## II. Literature Review

Clinical prediction models should be evaluated not only by accuracy but also by discrimination, calibration, stability, and applicability. In healthcare settings, a model with good ranking ability can still be unsafe or misleading if its probability estimates are poorly calibrated. Recent prediction-model reporting standards such as TRIPOD+AI emphasize transparent reporting of data sources, preprocessing, validation, model development, and limitations for regression and machine-learning prediction models [1]. PROBAST+AI similarly highlights the need to assess risk of bias and applicability in artificial-intelligence prediction-model studies [2].

The UCI heart disease data originate from historical coronary artery disease prediction research and have become common in machine-learning education [3]. Although these data are valuable for reproducible experimentation, they are small, historical, and not representative of modern prospective clinical deployment. Therefore, this study treats the dataset as a public benchmark rather than as evidence of clinical readiness.

Scikit-learn was used for the core modeling workflow because it provides consistent APIs for preprocessing pipelines, cross-validation, linear models, ensemble methods, and model evaluation [4]. Explainability was supported through model coefficients, global feature importance, and SHAP-style summaries, following the broader principle that prediction systems should expose interpretable evidence rather than only a class label [5].

## III. Methodology

### A. Data Sources and Audit

Three source materials were audited: the original neural-network notebook, `heart_disease_uci.csv`, and `heart.csv`. The primary research dataset was `heart_disease_uci.csv`, containing 920 records from Cleveland, Hungary, Switzerland, and VA Long Beach. The dataset includes a `dataset` variable, which was used as the center identifier for internal-external validation.

The secondary benchmark dataset was `heart.csv`. It contained 1,025 rows but 723 exact duplicate rows. After exact deduplication, only 302 unique rows remained. Because duplicate rows can inflate performance under random splitting, this dataset was used only for supplementary sensitivity analysis.

### B. Target Definition

The main task was binary prediction of heart disease presence. For the UCI-style data, the multiclass `num` variable was collapsed as:

- `target_binary = 0` if `num = 0`
- `target_binary = 1` if `num` is 1, 2, 3, or 4

The `heart.csv` target was not pooled with the UCI target because its label semantics and duplicate-expanded structure were not sufficiently reliable for primary research claims.

### C. Feature Processing

The canonical feature set included age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, oldpeak, ST slope, number of major vessels, and thalassemia/stress-test defect category. The center variable was used for validation grouping and was not used as a prediction feature.

All preprocessing was placed inside scikit-learn pipelines. Numeric features were median-imputed and standardized. Categorical features were imputed using the most frequent value and one-hot encoded. This design ensured that imputation, encoding, and scaling were fitted only on the training portion of each fold.

### D. Validation Design

The main validation used leave-one-center-out internal-external validation. In each outer fold, one center was held out for testing while the remaining centers were used for model development and hyperparameter selection. This design is stronger than a single random split because it evaluates whether a model generalizes across hospital-like data sources.

### E. Models

The following models were compared:

1. Penalized logistic regression.
2. Random forest.
3. Gradient boosting.
4. Calibrated gradient boosting.
5. Multilayer perceptron baseline based on the original notebook.

The final model was selected using a combined view of AUROC, AUPRC, Brier score, calibration slope, MCC, center-to-center stability, and interpretability.

## IV. Experimentation

The experiments were run through a reproducible command-line pipeline. The pipeline generated data audits, model comparison tables, center-wise metrics, threshold analysis, bootstrap confidence intervals, plots, model metadata, and a serialized model bundle.

The primary dataset contained 411 records without heart disease and 509 records with heart disease. Center distributions were imbalanced: Switzerland and VA Long Beach had high disease prevalence, while Hungary had lower prevalence. Missingness was also substantial for some variables: `ca` was missing in 66.4% of records, `thal` in 52.8%, and `slope` in 33.6%. These missingness patterns support the decision to use fold-contained imputation and cautious interpretation.

Evaluation metrics included AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, precision, negative predictive value, F1 score, MCC, Brier score, calibration intercept, calibration slope, confusion matrix, ROC curves, precision-recall curves, calibration plots, decision-curve analysis, threshold analysis, and bootstrap confidence intervals.

## V. Results and Discussion

Table I summarizes the model comparison under center-aware validation.

**Table I. Primary Model Comparison**

| Model | AUROC | AUPRC | Brier | Balanced Acc. | MCC |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.799 | 0.880 | 0.168 | 0.745 | 0.441 |
| Calibrated gradient boosting | 0.809 | 0.877 | 0.174 | 0.714 | 0.385 |
| Random forest | 0.800 | 0.873 | 0.170 | 0.716 | 0.387 |
| Gradient boosting | 0.793 | 0.861 | 0.180 | 0.732 | 0.410 |
| MLP notebook baseline | 0.779 | 0.858 | 0.187 | 0.678 | 0.342 |

Calibrated gradient boosting had the highest mean AUROC, but penalized logistic regression was selected as the final model because it had the best overall selection score after considering calibration, Brier score, MCC, stability, and interpretability. Its calibration slope was closer to 1 than the more complex models, and its simpler structure made the prediction behavior easier to explain.

At the pooled outer-fold level, logistic regression achieved AUROC of 0.855 with a 95% bootstrap confidence interval of 0.828 to 0.878. AUPRC was 0.864, Brier score was 0.159, sensitivity was 0.729, and specificity was 0.842 at threshold 0.50. These pooled estimates are useful for summary, but the center-wise results remain important because performance varied across sites.

Center-wise performance showed that the model performed better on Cleveland and Hungary than on VA Long Beach. Switzerland had very high disease prevalence, making negative predictive value unstable. These differences indicate that the dataset contains center shift and that random-split-only evaluation would be misleading.

The most influential features included chest pain type, sex, exercise-induced angina, thalassemia/stress-test category, ST slope, oldpeak, cholesterol, fasting blood sugar, number of major vessels, and maximum heart rate. These variables are clinically plausible for a benchmark heart disease prediction task, but the model should not be interpreted causally.

## VI. Ablation and Comparative Studies

This project uses comparative and sensitivity studies rather than only feature-removal ablation, because the dataset is small and center-shifted.

### A. Model Comparison

The model comparison itself is the main comparative study. It shows that the neural-network baseline was not the best final model. The MLP comparator achieved mean AUROC of 0.779 and Brier score of 0.187, while logistic regression achieved mean AUROC of 0.799 and Brier score of 0.168. This supports the conclusion that a simpler calibrated model is more appropriate than a neural network for this dataset.

### B. Duplicate Sensitivity

The duplicate sensitivity study used the supplementary `heart.csv` file. The duplicate-expanded version gave higher apparent performance than the deduplicated version. For example, AUROC was 0.924 before deduplication and 0.909 after deduplication. This confirms that duplicate rows can inflate benchmark results and should not be used as the main evidence.

**Table II. Duplicate Sensitivity on Supplementary Dataset**

| Dataset version | AUROC | AUPRC | Balanced Acc. | Brier |
|---|---:|---:|---:|---:|
| Original duplicate-expanded `heart.csv` | 0.924 | 0.926 | 0.855 | 0.110 |
| Deduplicated `heart.csv` | 0.909 | 0.917 | 0.832 | 0.125 |

### C. Threshold Sensitivity

Threshold analysis showed the trade-off between sensitivity and specificity. At threshold 0.10, sensitivity was 0.988 but specificity was only 0.214. At threshold 0.50, sensitivity was 0.729 and specificity was 0.842. This supports why the deployed app reports probability and risk category rather than claiming a universal clinical decision threshold.

## VII. Localhost Prototype

The final model is served through a FastAPI application. The interface provides one public prediction workflow with a clinical-style input form. It returns predicted probability, calibrated risk category, important feature explanations, model metadata, and a disclaimer. The app does not use a language model in the prediction path. Explanations are generated from the trained model artifact and feature dictionary.

## VIII. Conclusion and Future Work

This project converts a notebook-based heart disease experiment into a reproducible machine-learning research system. The final model was selected through leakage-audited, site-aware validation rather than a single random split. Penalized logistic regression was selected over more complex models because it provided the best balance of calibration, interpretability, and validation performance.

Future work should include prospective validation, larger and more modern datasets, fairness analysis, external validation on independent cohorts, improved missingness modeling, and clinician-centered usability evaluation. The current system should be viewed as an educational benchmark and decision-support prototype, not as clinical deployment evidence.

## References

[1] G. S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," BMJ, vol. 385, e078378, 2024.

[2] G. S. Collins et al., "PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods," BMJ, vol. 388, e082505, 2025.

[3] R. Detrano et al., "International application of a new probability algorithm for the diagnosis of coronary artery disease," American Journal of Cardiology, 1989.

[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.

[5] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," Advances in Neural Information Processing Systems, 2017.
