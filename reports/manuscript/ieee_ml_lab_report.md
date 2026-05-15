# Early CVD Prediction Using Clinical Data: A Leakage-Audited and Explainable Machine Learning Study

Rashik Mahmud Majumder, Ravian Irtisam, Shahariar Hossain, Rumana Yasmin, Md. Sakib Hasan, and Samia Ahmed  
Bangladesh University of Professionals  
Dhaka, Bangladesh

## Abstract

Cardiovascular disease prediction from routine clinical variables is a common machine-learning laboratory problem, but public benchmark studies can overstate performance when duplicate rows, missingness, validation leakage, and calibration are not handled carefully. This project develops Early CVD Prediction as a reproducible and explainable heart-disease prediction workflow using a primary UCI-style multi-center dataset and a secondary heart.csv benchmark. The main binary outcome collapsed num = 1-4 into disease presence and retained num = 0 as disease absence. The primary dataset contained 920 records from Cleveland, Hungary, Switzerland, and VA Long Beach; the secondary file contained 1025 rows but only 302 unique rows after exact deduplication. The main experiment used leave-one-center-out internal-external validation with all preprocessing fitted inside training folds. Ten candidate models were compared, including interpretable baselines, tree ensembles, calibrated boosting, RBF SVM, XGBoost, LightGBM, CatBoost, stacking, and a multilayer perceptron baseline. Calibrated gradient boosting was selected by the composite ranking because it achieved the best overall balance: mean held-out-center AUROC 0.808, AUPRC 0.882, Brier score 0.155, balanced accuracy 0.744, and MCC 0.439. In pooled held-out-center predictions, the selected model achieved AUROC 0.859, AUPRC 0.866, Brier score 0.152, and balanced accuracy 0.784. The work includes a compatible notebook, generated figures and tables, statistical comparisons, tests, and a FastAPI localhost application with probability and explanation output. The system is a research and educational prototype, not a clinically deployed diagnostic device.

**Keywords**: cardiovascular disease prediction, heart disease, machine learning, calibration, explainable AI, site-aware validation

## I. Introduction

Early prediction of cardiovascular disease (CVD) is important because intervention is more useful when risk is identified before severe clinical events. In machine-learning education, heart-disease prediction is also valuable because the predictors are structured, clinically meaningful, and small enough for transparent analysis. Typical variables include age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic findings, maximum heart rate, exercise-induced angina, ST depression measured as oldpeak, ST slope, major vessels, and thalassemia or stress-test category.

The original work for this project began as an exploratory neural-network notebook. A project audit showed that the notebook was useful for motivation and baseline modeling, but it used notebook-only code, a single random split, and a small heart.csv file that did not match the current primary dataset. Therefore, the project was rebuilt around a simpler and more defensible research question: can a leakage-audited, site-aware, explainable pipeline provide a reliable benchmark for binary heart-disease prediction using public tabular clinical data?

The contribution of this work is not a claim of clinical deployment readiness. Instead, the contribution is a complete ML Lab project that connects notebook research with reproducible software: data audit, duplicate analysis, leakage control, fold-contained preprocessing, center-aware validation, model comparison, calibration-aware selection, explainability, a localhost application, and a concise IEEE-style report generated from experiment artifacts.

## II. Literature Review

Recent literature shows strong interest in ML-based CVD prediction, especially with routine electronic health records and tabular clinical features. Systematic reviews and meta-analyses report that random forests, boosting models, support-vector machines, neural networks, and custom ML models can improve discrimination over some conventional scores in selected settings [1]-[3]. However, the same literature repeatedly warns that performance estimates vary substantially across cohorts, outcome definitions, validation strategies, and reporting quality. This is why the present project reports both discrimination and calibration and avoids presenting one random split as external validation.

A second direction uses structured clinical datasets such as Cleveland, Hungarian, Statlog, and merged UCI-style heart-disease data. These studies often combine feature engineering or feature selection with ensemble classifiers and report high accuracy [4], [6], [7]. Such results are useful benchmarks, but public heart-disease datasets are small and sometimes duplicated or reprocessed. This project therefore audits the supplied files directly, treats the UCI-style multi-center file as primary, and treats heart.csv only as a supplementary benchmark after deduplication.

A third direction emphasizes ensembles, boosting, and recent deep tabular methods. Hybrid ensembles, stacking models, CatBoost or XGBoost-style learners, and transformer-inspired tabular models have been proposed for early CVD or heart-disease prediction [8]-[10]. These methods motivate the comparison of several model families rather than assuming that the neural network baseline is best. In this project, the final model is selected by balanced evidence across AUROC, AUPRC, Brier score, calibration slope, center stability, and interpretability.

Explainable AI is increasingly used in CVD prediction, often through SHAP and LIME, to show which clinical variables influence predictions [8], [9], [13]. Explanations are valuable for checking plausibility but should not be confused with causal evidence, especially when clinical variables are correlated. This project therefore uses explanation output as a model-audit and communication tool, not as proof that a variable biologically causes the outcome.

**Table I. Literature Review Synthesis and Project Use**

| Direction | Finding from Literature | Use in This Project |
| --- | --- | --- |
| EHR and cohort evidence | Systematic reviews and meta-analyses report that ML can improve CVD risk discrimination, while validation quality and heterogeneity remain concerns [1]-[3]. | Report discrimination, calibration, and limitations rather than accuracy alone. |
| Tabular clinical datasets | UCI-style studies commonly use age, chest pain, blood pressure, cholesterol, ECG, exercise angina, oldpeak, slope, vessels, and thal variables [4], [6], [7]. | Use a canonical clinical feature dictionary and keep interpretable predictors. |
| Ensembles and boosting | Recent work frequently reports strong performance for random forests, boosting, stacking, and transformer-assisted tabular models [8]-[10]. | Compare stronger tabular learners against the neural-network baseline. |
| Explainability | SHAP and related explanation methods are widely used for clinical ML transparency, but explanations require caution under correlated features [8], [9], [13]. | Provide global and local explanations without causal claims. |
| Reporting standards | TRIPOD+AI and PROBAST+AI emphasize transparent preprocessing, validation, calibration, and risk-of-bias assessment [11], [12]. | Use leakage-safe pipelines, site-aware validation, and explicit dataset limitations. |

## III. Methodology

The primary dataset was heart_disease_uci.csv. The target variable was converted to a binary label: num = 0 represented absence of recorded disease and num > 0 represented disease presence. The center or hospital source was retained for validation grouping but excluded from the predictor set. This design supports internal-external validation: each center is held out once to approximate a new-site test while the remaining centers are used for development.

The secondary dataset was heart.csv. It was not assumed to be clean. Exact duplicate analysis found 723 duplicate rows, leaving 302 unique records. Because this file does not contain a center variable and has heavy row duplication, it was used only for supplementary benchmarking and duplicate sensitivity analysis.

The feature schema was standardized across files. Naming differences such as thalach and thalch, and target and num, were handled through a canonical dictionary. Numeric predictors were median-imputed and standardized. Categorical predictors were imputed with the most frequent value and one-hot encoded. All preprocessing was inside scikit-learn pipelines so imputation, scaling, encoding, and model fitting were learned only from training folds. This is the central leakage-control decision in the project.

Ten model families were evaluated: penalized logistic regression, random forest, gradient boosting, calibrated gradient boosting, RBF SVM, XGBoost, LightGBM, CatBoost, heterogeneous stacking, and a multilayer perceptron baseline preserving the direction of the original neural-network notebook. The final selection principle was methodological rather than cosmetic: accuracy alone could not determine the champion model. A model was preferred only if it showed a balanced profile across discrimination, calibration, stability, interpretability, and suitability for a small tabular clinical dataset.

The localhost application uses FastAPI with Pydantic validation and a lightweight integrated HTML interface. This choice keeps the project simple for submission and presentation while still providing automatic API documentation. The public workflow uses one clinical input form, reports calibrated probability and risk category, displays important feature contributions, and includes a research-prototype disclaimer.

**Figure III.A. Core Methodology Flowchart**

```mermaid
graph TD
    A["DATA ACQUISITION"] --> B["Data Input"]
    B --> B1["Primary: heart_disease_uci.csv<br/>Secondary: heart.csv"]

    B1 --> C["DATA VALIDATION"]
    C --> C1["Duplicate Detection<br/>Missing Value Analysis<br/>Distribution Audit"]

    C1 --> D["PREPROCESSING PIPELINE"]
    D --> D1["Feature Standardization<br/>Canonical Dictionary Mapping"]
    D1 --> D2["Numeric Features:<br/>Median Imputation -> Standardization<br/><br/>Categorical Features:<br/>Mode Imputation -> One-Hot Encoding"]
    D2 --> D3["Leakage Control:<br/>Inside scikit-learn Pipelines<br/>Fit on Training Folds Only"]

    D3 --> E["VALIDATION STRATEGY"]
    E --> E1["Internal-External Validation<br/>Hold-out Each Center"]
    E1 --> E2["Development: Remaining Centers<br/>Test: Held-out Center"]

    E2 --> F["MODEL DEVELOPMENT"]
    F --> F1["Penalized Logistic Regression<br/>Random Forest<br/>Gradient Boosting<br/>Calibrated Gradient Boosting<br/>RBF SVM<br/>XGBoost / LightGBM / CatBoost<br/>Stacking Ensemble<br/>Multilayer Perceptron"]

    F1 --> G["MODEL EVALUATION"]
    G --> G1["Discrimination: AUC/Sensitivity/Specificity<br/>Calibration: Calibration Curves<br/>Stability: Cross-fold Consistency<br/>Interpretability: Feature Importance<br/>Fit: Dataset Suitability"]

    G1 --> H["MODEL SELECTION"]
    H --> H2["Balanced Profile Across:<br/>Discrimination<br/>Calibration<br/>Stability<br/>Interpretability<br/>Clinical Suitability"]

    H2 --> I["DEPLOYMENT"]
    I --> I1["FastAPI Application<br/>Pydantic Validation<br/>HTML Interface"]
    I1 --> I2["Clinical Input Form<br/>Calibrated Risk Prediction<br/>Feature Attribution<br/>Research Prototype Disclaimer"]

    style A fill:#e1f5ff
    style C fill:#f3e5f5
    style D fill:#fff3e0
    style E fill:#e8f5e9
    style F fill:#fce4ec
    style G fill:#f1f8e9
    style H fill:#e0f2f1
    style I fill:#ede7f6
```

The flowchart above visualizes the complete methodology pipeline from data acquisition through deployment. Each stage enforces specific controls: data validation ensures file integrity and duplicate detection; the preprocessing pipeline implements all feature engineering inside training folds to prevent leakage; the validation strategy approximates new-site behavior through leave-one-center-out folds; model evaluation balances multiple criteria rather than optimizing for a single metric; and deployment provides both risk estimates and explainable feature contributions with an appropriate clinical disclaimer.

**Table II. Dataset Audit Summary**

| Role | Dataset | Rows | Grouping | Target/Duplicate Finding | Use |
| --- | --- | --- | --- | --- | --- |
| Primary | heart_disease_uci.csv | 920 | 4 centers | 411 absent / 509 present | Main site-aware experiment |
| Supplementary | heart.csv | 1025 | No center field | 302 unique; 723 exact duplicates | Deduplicated benchmark only |

**Table III. Highest Missingness Variables in the Primary Dataset**

| Variable | Missing Count | Missing Percent |
| --- | --- | --- |
| ca | 611 | 66.4% |
| thal | 486 | 52.8% |
| slope | 309 | 33.6% |
| fbs | 90 | 9.8% |
| oldpeak | 62 | 6.7% |
| trestbps | 59 | 6.4% |
| exang | 55 | 6.0% |
| thalach | 55 | 6.0% |

## IV. Experimentation

The main experiment used leave-one-center-out validation on the primary dataset. In each outer fold, one center was held out. Model selection was performed only on the development centers, and the selected candidate was evaluated on the held-out center. This avoids using the held-out center to choose preprocessing or hyperparameters.

Metrics included AUROC, AUPRC, accuracy, balanced accuracy, sensitivity, specificity, precision or positive predictive value, negative predictive value, F1 score, Matthews correlation coefficient, Brier score, calibration intercept, calibration slope, confusion matrix, and threshold analysis. Bootstrap confidence intervals were calculated for key pooled held-out predictions. Generated artifacts include ROC, precision-recall, calibration, decision-curve, threshold-sensitivity, probability-distribution, confusion-matrix, learning-curve, statistical-test, missingness, distribution, feature-importance, and SHAP-style figures.

Ablation and comparative analyses focused on four practical questions: whether the neural network was better than simpler tabular models, whether threshold choice changed clinical operating behavior, whether center-to-center performance was stable, and whether duplicate-expanded heart.csv results differed from deduplicated benchmark results.

## V. Results and Discussion

The data audit showed that the primary dataset is small and heterogeneous but useful for a site-aware laboratory study. Cleveland and Hungary contain more balanced disease distributions, whereas Switzerland and VA Long Beach have higher disease prevalence. This center imbalance is exactly why a random split would be a weak main result: randomization could mix site-specific patterns into both training and test data.

Missingness was clinically and methodologically important. The most incomplete variables were ca (66.4% missing), thal (52.8%), slope (33.6%), fbs (9.8%), oldpeak (6.7%), trestbps (6.4%), exang (6.0%), and thalach (6.0%). Since these missing values can be center-dependent, imputation was deliberately performed inside each training fold.

The primary model comparison did not support choosing the neural network as the final model. RBF SVM had the highest mean AUROC and AUPRC, but it had weak specificity and balanced accuracy at the selected threshold. Calibrated gradient boosting achieved the best composite ranking after considering discrimination, Brier score, F1, MCC, calibration slope, center stability, and interpretability. This is consistent with the practical lesson that small tabular clinical datasets do not automatically benefit from deeper models.

For pooled held-out-center predictions, the selected calibrated gradient boosting model achieved AUROC 0.859, AUPRC 0.866, Brier score 0.152, and balanced accuracy 0.784. At threshold 0.50, sensitivity was 0.806 and specificity was 0.762. Lowering the threshold to 0.30 increased sensitivity but reduced specificity; raising it to 0.70 did the opposite. Therefore, the application reports a probability and risk category rather than claiming a universal diagnostic threshold.

Center-wise results showed important generalizability differences. Performance was strongest for Cleveland and Hungary, lower for Switzerland and VA Long Beach, and calibration was not uniform across sites. These results are scientifically useful because they show where the public dataset is fragile. The project therefore presents the model as a benchmark and decision-support prototype for education, not as a clinical device.

The explanation analysis identified chest pain type, cholesterol, oldpeak, exercise-induced angina, sex, age, thal category, and maximum heart rate as influential variables. These are plausible for a heart-disease benchmark, but the interpretation remains associational. The explanation layer is meant to answer why the model produced a prediction for the entered profile in terms of learned patterns, not to replace clinical judgment.

**Table IV. Primary Model Comparison**

| Model | AUROC | AUPRC | Brier | Balanced Acc. | MCC |
| --- | --- | --- | --- | --- | --- |
| Calibrated gradient boosting | 0.808 | 0.882 | 0.155 | 0.744 | 0.439 |
| RBF SVM | 0.813 | 0.883 | 0.162 | 0.678 | 0.392 |
| Random forest | 0.803 | 0.875 | 0.168 | 0.729 | 0.409 |
| Penalized logistic regression | 0.780 | 0.871 | 0.180 | 0.734 | 0.421 |
| Stacking ensemble | 0.786 | 0.863 | 0.176 | 0.736 | 0.414 |
| CatBoost | 0.792 | 0.871 | 0.176 | 0.719 | 0.393 |
| Gradient boosting | 0.794 | 0.864 | 0.185 | 0.735 | 0.404 |
| XGBoost | 0.792 | 0.871 | 0.184 | 0.694 | 0.361 |
| LightGBM | 0.772 | 0.863 | 0.182 | 0.699 | 0.370 |
| MLP notebook baseline | 0.754 | 0.855 | 0.184 | 0.704 | 0.381 |

**Table V. Held-Out Center Results for the Selected Model**

| Held-Out Center | AUROC | AUPRC | Brier | Balanced Acc. | Sensitivity | Specificity |
| --- | --- | --- | --- | --- | --- | --- |
| Cleveland | 0.867 | 0.865 | 0.151 | 0.792 | 0.820 | 0.764 |
| Hungary | 0.897 | 0.837 | 0.130 | 0.818 | 0.802 | 0.834 |
| Switzerland | 0.745 | 0.971 | 0.154 | 0.708 | 0.791 | 0.625 |
| VA Long Beach | 0.723 | 0.857 | 0.183 | 0.658 | 0.805 | 0.510 |

**Table VI. Bootstrap Confidence Intervals for Pooled Held-Out Predictions**

| Metric | Estimate | 95% CI |
| --- | --- | --- |
| AUROC | 0.859 | 0.835-0.883 |
| AUPRC | 0.866 | 0.831-0.900 |
| Balanced Accuracy | 0.784 | 0.757-0.810 |
| Brier | 0.152 | 0.139-0.164 |
| Sensitivity | 0.806 | 0.772-0.841 |
| Specificity | 0.762 | 0.722-0.802 |

**Table VII. Threshold Operating Points for the Selected Model**

| Threshold | Sensitivity | Specificity | PPV | NPV | Balanced Acc. | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| 0.300 | 0.925 | 0.550 | 0.718 | 0.856 | 0.738 | 0.522 |
| 0.500 | 0.806 | 0.762 | 0.807 | 0.760 | 0.784 | 0.567 |
| 0.700 | 0.591 | 0.915 | 0.896 | 0.644 | 0.753 | 0.523 |

**Table VIII. Top Global Explanation Features (Permutation Importance)**

| Feature | Importance |
| --- | --- |
| cp | 0.060 |
| chol | 0.047 |
| oldpeak | 0.021 |
| exang | 0.020 |
| sex | 0.012 |
| age | 0.008 |
| thal | 0.007 |
| thalach | 0.007 |

**Fig. 1. Class distribution in the primary UCI-style dataset.**

![Class distribution in the primary UCI-style dataset.](../figures/class_distribution_primary.png)

**Fig. 2. Center distribution used for leave-one-center-out validation.**

![Center distribution used for leave-one-center-out validation.](../figures/site_distribution_primary.png)

**Fig. 3. Missingness pattern across primary dataset predictors.**

![Missingness pattern across primary dataset predictors.](../figures/missingness_heatmap.png)

**Fig. 4. Model comparison across site-aware validation folds.**

![Model comparison across site-aware validation folds.](../figures/model_comparison.png)

**Fig. 5. Multi-metric model comparison across selected performance criteria.**

![Multi-metric model comparison across selected performance criteria.](../figures/model_comparison_multi_metric.png)

**Fig. 6. Model performance radar showing discrimination, calibration, and threshold behavior.**

![Model performance radar showing discrimination, calibration, and threshold behavior.](../figures/radar_chart.png)

**Fig. 7. ROC curves from held-out-center predictions.**

![ROC curves from held-out-center predictions.](../figures/roc_curves.png)

**Fig. 8. Precision-recall curves from held-out-center predictions.**

![Precision-recall curves from held-out-center predictions.](../figures/precision_recall_curves.png)

**Fig. 9. Calibration curves for compared models.**

![Calibration curves for compared models.](../figures/calibration_plots.png)

**Fig. 10. Confusion matrix for the selected model at threshold 0.50.**

![Confusion matrix for the selected model at threshold 0.50.](../figures/confusion_matrix_champion.png)

**Fig. 11. Predicted probability distribution by observed class.**

![Predicted probability distribution by observed class.](../figures/predicted_probability_violin.png)

**Fig. 12. Threshold sensitivity analysis for F1, sensitivity, and specificity.**

![Threshold sensitivity analysis for F1, sensitivity, and specificity.](../figures/threshold_sensitivity.png)

**Fig. 13. Decision-curve analysis for threshold-probability ranges.**

![Decision-curve analysis for threshold-probability ranges.](../figures/decision_curve.png)

**Fig. 14. Global feature importance for the selected model.**

![Global feature importance for the selected model.](../figures/global_feature_importance.png)

**Fig. 15. SHAP-style summary plot for explanation review.**

![SHAP-style summary plot for explanation review.](../figures/shap_summary.png)

**Fig. 16. Learning curve for the selected model.**

![Learning curve for the selected model.](../figures/learning_curves.png)

**Fig. 17. DeLong pairwise AUROC test heatmap.**

![DeLong pairwise AUROC test heatmap.](../figures/delong_heatmap.png)

## VI. Ablation Studies / Comparative Studies

The main comparative study is the model-family comparison. The MLP baseline was preserved from the original notebook direction but underperformed the calibrated gradient boosting, SVM, random-forest, logistic-regression, and stacking alternatives on the composite ranking. This supports the final decision to keep the neural network as a comparator rather than the champion.

The duplicate sensitivity analysis showed why heart.csv is supplementary. The duplicate-expanded version produced stronger apparent performance than the deduplicated version, especially AUROC 0.955 versus 0.903 and Brier score 0.086 versus 0.126. Because duplicate rows can make evaluation easier and less realistic, the deduplicated result is the only defensible supplementary benchmark.

The threshold analysis showed a clear sensitivity-specificity trade-off. A lower threshold is more sensitive and may be useful for screening-oriented analysis, whereas a higher threshold is more specific and reduces false positives. Since the project is not clinically deployed, no single operating threshold is recommended as medically optimal.

The center sensitivity analysis is the most important ablation for scientific honesty. Holding out Switzerland and VA Long Beach produced weaker performance than holding out Cleveland and Hungary. This indicates distribution shift between centers and justifies the report limitation that prospective validation would be required before any clinical use.

The statistical comparison artifacts add another check on the model claims. DeLong pairwise testing on pooled held-out predictions found several significant AUROC differences, including calibrated gradient boosting versus the MLP baseline and XGBoost. However, the per-center Friedman test was not significant at the 0.05 level (chi-square = 15.109, p = 0.088), which is expected given only four held-out centers and supports cautious interpretation rather than overclaiming superiority.

**Table IX. Duplicate Sensitivity on the Supplementary heart.csv Dataset**

| Dataset Version | AUROC | AUPRC | Balanced Acc. | Brier | MCC |
| --- | --- | --- | --- | --- | --- |
| Deduplicated heart.csv | 0.903 | 0.912 | 0.829 | 0.126 | 0.666 |
| Duplicate-expanded heart.csv | 0.955 | 0.955 | 0.870 | 0.086 | 0.742 |

## VII. Conclusion and Future Work

This project satisfies the ML Lab objective by building a complete early CVD prediction workflow from clinical tabular data while correcting the weaknesses of the initial notebook-only approach. The final system contains a reproducible notebook, modular Python code, leakage-safe training and evaluation, generated figures and tables, a FastAPI localhost application, tests, and an IEEE-style report.

The selected model is calibrated gradient boosting because it is competitive across discrimination, has the best mean Brier score, provides strong pooled held-out-center performance, and remains deployable with probability calibration and post-hoc explanation artifacts. Future work should evaluate the model on newer prospective cohorts, add stronger clinical baselines, improve calibration by site, examine fairness across patient subgroups, and validate the user interface with clinical stakeholders. Until then, the project should be described as a reproducible educational benchmark and research prototype, not as a clinical diagnostic system.

## References

[1] T. Liu, A. J. Krentz, L. Lu, and V. Curcin, "Machine learning based prediction models for cardiovascular disease risk using electronic health records data: systematic review and meta-analysis," European Heart Journal - Digital Health, vol. 6, no. 1, pp. 7-22, 2025.

[2] C. Krittanawong et al., "Machine learning prediction in cardiovascular diseases: a meta-analysis," Scientific Reports, vol. 10, Art. no. 16057, 2020.

[3] Y. Cai et al., "Artificial intelligence in the risk prediction models of cardiovascular disease and development of an independent validation screening tool: a systematic review," BMC Medicine, vol. 22, Art. no. 56, 2024.

[4] J. Azmi, M. Arif, M. T. Nafis, M. A. Alam, S. Tanweer, and G. Wang, "A systematic review on machine learning approaches for cardiovascular disease prediction using medical big data," Medical Engineering & Physics, vol. 103, Art. no. 103825, 2022.

[5] A. B. Teshale et al., "A systematic review of artificial intelligence models for time-to-event outcome applied in cardiovascular disease risk prediction," Journal of Medical Systems, 2024.

[6] H. F. El-Sofany, B. Bouallegue, and Y. M. A. El-Latif, "A proposed technique for predicting heart disease using machine learning algorithms and an explainable AI method," Scientific Reports, 2024.

[7] E. Dritsas and M. Trigka, "Efficient data-driven machine learning models for cardiovascular diseases risk prediction," Sensors, vol. 23, no. 3, Art. no. 1161, 2023.

[8] P. Shah, M. Shukla, N. H. Dholakia, and H. Gupta, "Predicting cardiovascular risk with hybrid ensemble learning and explainable AI," Scientific Reports, 2025.

[9] S. M. Ganie, P. K. D. Pramanik, and Z. Zhao, "Ensemble learning with explainable AI for improved heart disease prediction based on multiple datasets," Scientific Reports, 2025.

[10] M. S. I. Sumon et al., "CardioTabNet: a novel hybrid transformer model for heart disease prediction using tabular medical data," Health Information Science and Systems, 2025.

[11] G. S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," BMJ, vol. 385, e078378, 2024.

[12] K. G. M. Moons et al., "PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods," BMJ, vol. 388, e082505, 2025.

[13] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems, 2017.

[14] A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1989, doi: 10.24432/C52P4X.

[15] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.
