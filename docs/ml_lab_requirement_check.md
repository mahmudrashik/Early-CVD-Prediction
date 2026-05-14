# ML Lab Requirement Check

## Requirement Match

The project matches the ML Lab project requirements after the report and notebook cleanup. The codebase, generated results, IEEE-style report draft, final notebook, figures, tables, and localhost app are aligned with the required submission structure.

## Required Sections

| Required section | Current project status | Where it is handled |
|---|---|---|
| Abstract | Present, now expanded for paper use | `reports/manuscript/ieee_ml_lab_report.md` |
| Introduction | Present but needed expansion | `reports/manuscript/ieee_ml_lab_report.md` |
| Literature Review | Needed a dedicated section | `reports/manuscript/ieee_ml_lab_report.md` |
| Methodology | Present in code/docs, now written as report text | `src/application/train.py`, `src/pipelines/preprocessing.py`, report draft |
| Experimentation | Present in generated tables, now summarized | `reports/tables/`, report draft |
| Results and Discussion | Present in tables/figures, now interpreted | `reports/tables/model_selection_ranking.csv`, `reports/figures/` |
| Ablation / Comparative Studies | Present as model comparison, duplicate sensitivity, threshold sensitivity | `reports/tables/model_selection_ranking.csv`, `duplicate_sensitivity.csv`, `threshold_analysis.csv` |
| Conclusion and Future Work | Present in new paper draft | `reports/manuscript/ieee_ml_lab_report.md` |
| References | Added in IEEE numeric style | `reports/manuscript/ieee_ml_lab_report.md` |

## Does the Project Match the Requirement?

Yes, after the paper rewrite, the project matches the lab requirement well. It includes a complete ML workflow, reproducible experiments, model comparison, generated figures/tables, a localhost app, and a research-style report. The strongest part is that the project avoids a naive random split as the main result and instead uses center-aware validation.

## Where Is Ablation / Comparative Study?

Use the term **Comparative and Sensitivity Studies** in the report. It is supported by:

1. **Model comparison**: logistic regression, random forest, gradient boosting, calibrated gradient boosting, and MLP baseline.
2. **Duplicate sensitivity**: original duplicate-expanded `heart.csv` versus deduplicated `heart.csv`.
3. **Threshold sensitivity**: performance from thresholds 0.10 to 0.90.
4. **Notebook baseline comparison**: original neural-network work is retained as a comparator, but not treated as final evidence.

This is acceptable for the lab section titled **Ablation Studies / Comparative Studies**. Strict feature-removal ablation was not the main design because the dataset is small and center-shifted; model and data sensitivity studies are more scientifically honest here.

## What Should Be Added in Results?

The Results section should include:

- Primary model comparison table.
- Final selected model and why it was selected.
- Center-wise validation result discussion.
- ROC, PR, calibration, confusion matrix, and model comparison figures.
- Duplicate sensitivity finding for `heart.csv`.
- Threshold analysis at clinically meaningful thresholds.
- A clear caution that results are not clinical deployment evidence.

Recommended results wording:

> Penalized logistic regression was selected as the final model because it had the best overall selection score after considering AUROC, AUPRC, Brier score, calibration slope, MCC, stability, and interpretability. Although calibrated gradient boosting had slightly higher mean AUROC, logistic regression had better calibration and interpretability, making it more appropriate for this small public-dataset clinical prediction task.

## What Is the Methodology?

The methodology is:

1. Audit the original notebook and datasets.
2. Use `heart_disease_uci.csv` as the primary dataset.
3. Define binary target: `num = 0` as no disease, `num > 0` as disease.
4. Use `dataset` as the hospital/center grouping variable.
5. Standardize schema and feature names.
6. Keep preprocessing inside pipelines:
   - median imputation for numeric variables,
   - most-frequent imputation for categorical variables,
   - standard scaling for numeric variables,
   - one-hot encoding for categorical variables.
7. Run leave-one-center-out internal-external validation.
8. Tune models only on development centers.
9. Compare multiple models.
10. Select final model using discrimination, calibration, stability, and interpretability.
11. Generate explanations and deploy through FastAPI.

## Notebook Compatibility

The exploratory notebook is compatible as historical baseline work, not as the final scientific result.

Main issue:

- The notebook used a 303-row `heart.csv` from `./Desktop/Hrt_Dis_Pred/heart.csv`.
- The uploaded `heart.csv` has 1,025 rows and 723 exact duplicates.
- The notebook tuned a neural network using the test set as validation data.

How the final project relates to the notebook:

- The exploratory notebook was audited in `docs/current_work_audit.md`.
- The neural-network idea is retained as the `mlp_notebook_baseline` comparator.
- The final project notebook is `notebooks/Early_CVD_Prediction_ML_Lab.ipynb`.
- The final project improves the notebook by using leakage-safe pipelines, site-aware validation, calibration metrics, reproducible reports, and a deployable app.

## Final Recommendation

For the ML Lab submission, use:

- `reports/manuscript/early_cvd_ieee_lab_report.docx` as the editable IEEE-style report draft.
- `reports/manuscript/ieee_ml_lab_report.md` as the source text.
- `reports/figures/` for figures.
- `reports/tables/` for tables.

Before final submission, fill in the exact student name, ID, course code, section, and teacher information required by your department.
