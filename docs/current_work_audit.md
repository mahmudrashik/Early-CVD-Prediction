# Current Work Audit

Project: Early CVD Prediction

Date of audit: 2026-05-13, Asia/Dhaka

Audited source files:

- `C:\Users\mahmu\OneDrive\Documents\Semester 8\ML Lab\Heart_Disease_Prediction_03_SmallSet_Feat_Eng_NeuralNet_ModDef_Training-Copy1.ipynb`
- `C:\Users\mahmu\OneDrive\Documents\Semester 8\ML Lab\Project\heart.csv`
- `C:\Users\mahmu\OneDrive\Documents\Semester 8\ML Lab\Project\heart_disease_uci.csv`

## Executive Finding

The existing notebook is a useful neural-network exploratory analysis, but it is not yet a publication-grade prediction-model workflow. The main reasons are: the notebook used a different 303-row `heart.csv` than the currently uploaded 1,025-row file; the uploaded benchmark file contains extensive duplicate rows; target coding between the Kaggle-style `heart.csv` and the UCI-style `heart_disease_uci.csv` is not safely aligned; model selection used the test set during neural-network tuning; and the primary evaluation is a single random split rather than site-aware validation.

The notebook should therefore be treated as historical baseline work, not as final scientific evidence.

## Notebook Summary

The notebook is titled `Heart Disease Prediction - Neural Network Model Definition and Training`. It contains 167 cells and uses a Python 3 kernel. Its stated dataset is the Cleveland heart disease dataset from UCI/Kaggle.

Main workflow:

1. Imports a large scientific stack including pandas, NumPy, imbalanced-learn, scikit-learn, statsmodels, XGBoost, TensorFlow/Keras, seaborn, matplotlib, scikit-plot, and sktools.
2. Sets random seeds with `numpy.random.seed(42)` and `tf.random.set_seed(42)`.
3. Loads data using `pd.read_csv('./Desktop/Hrt_Dis_Pred/heart.csv')`.
4. Reports shape `(303, 14)` in the notebook output.
5. Performs exploratory histogram plotting.
6. Separates `target` from 13 predictors.
7. Runs chi-squared univariate feature scoring.
8. Runs ExtraTrees feature importance.
9. Performs low-variance review using a threshold of `0.006`; no features are removed.
10. One-hot encodes `cp`, `slope`, `thal`, and `restecg`, increasing predictors from 13 to 23.
11. Uses `LabelEncoder` on the target.
12. Notes SMOTE as optional; the SMOTE cell is raw/unexecuted and the classes are described as fairly balanced.
13. Splits the data into an 80/20 train/test split using `train_test_split(..., train_size=0.8, random_state=42)`.
14. Fits `MinMaxScaler` on `X_train` and transforms train/test.
15. Trains multiple Keras feedforward neural-network models.
16. Saves best-weight and tuned-model artifacts as HDF5 files.
17. Evaluates using accuracy, MCC, classification report, confusion matrix, ROC-AUC, Jaccard score, and a binomial confidence interval for accuracy.
18. Produces a sample prediction and probabilities from a reconstructed tuned model.

## Dataset Used by the Notebook

The notebook used a file at `./Desktop/Hrt_Dis_Pred/heart.csv` and reports:

- Rows: 303
- Columns: 14
- Columns shown: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`
- Class distribution after label encoding:
  - Class 1: 165 records, 54.455%
  - Class 0: 138 records, 45.545%

This is not the same physical file as the currently uploaded `heart.csv`, which has 1,025 rows and 723 exact duplicates. A known notebook row (`63,1,3,145,233,1,0,150,0,2.3,0,0,1,1`) appears three times in the current uploaded `heart.csv`, supporting the conclusion that the uploaded file is a duplicate-expanded derivative rather than the exact notebook file.

## Reusable Work

The following parts are reusable after refactoring:

- Basic feature list and clinical variable descriptions.
- Initial exploratory logic for feature distributions.
- Chi-squared and tree-based feature-importance ideas, if reimplemented inside appropriate training-only or explanatory contexts.
- One-hot encoding choice for categorical predictors, provided encoding is fit inside cross-validation folds.
- Min-max or standard scaling for neural-network and penalized linear models, provided scalers are inside the pipeline.
- The original feedforward neural-network design as a comparator.
- Reporting of confusion matrix, classification report, MCC, and ROC-AUC as partial evaluation ingredients.
- Saved-model concept, but not the specific notebook artifacts as final deployable models.

## Technical Debt

The notebook contains several issues that prevent it from serving as the final research pipeline:

- Hard-coded local path: `./Desktop/Hrt_Dis_Pred/heart.csv`.
- Notebook-only state: multiple cells depend on implicit variables from earlier cells.
- Mixed exploratory, training, tuning, evaluation, and artifact-saving logic in one document.
- No data-version manifest or source hash tracking.
- No package structure, CLI, tests, CI, or deployment layer.
- No external-style validation.
- No model registry or experiment tracking.
- No reproducible configuration files.
- No typed input schema for the prediction app.
- Uses deprecated Keras methods such as `predict_classes` and `predict_proba` on Sequential models.
- Uses a raw/unexecuted SMOTE cell, leaving ambiguity about whether oversampling was used.
- Some imported models and libraries are unused.
- No calibration metrics or calibration plots.
- No decision-curve analysis.
- No bootstrap confidence intervals for key metrics beyond a simple binomial interval for accuracy.
- No site-stratified performance analysis.
- No formal leakage audit.

## Current Notebook Models and Artifacts

### Baseline Neural Network

Architecture:

- Input dimension: 23 encoded/scaled predictors
- Dense layer: 72 units, ReLU activation
- Output layer: 2 units, softmax activation
- Loss: sparse categorical cross-entropy
- Optimizer: Adam
- Metric: accuracy
- Trainable parameters: 1,874

Callbacks:

- `EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)` was defined but not used in the baseline fit call.
- `ModelCheckpoint(filepath='model.weights.bestLoss.hdf5', monitor='val_loss', save_best_only=True, mode='min')`
- `ModelCheckpoint(filepath='model.weights.bestAcc.hdf5', monitor='val_accuracy', save_best_only=True, mode='max')`
- `ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=100)`

Reported results:

- Best-loss loaded model test accuracy: 86.89%
- Best-accuracy loaded model test accuracy: 85.25%
- Baseline summary cell reported:
  - Test loss: 0.3619
  - Test accuracy: 0.8689
  - MCC: 0.754
- Classification report from one baseline evaluation:
  - Class 0 precision/recall/F1: 0.79 / 0.93 / 0.86
  - Class 1 precision/recall/F1: 0.93 / 0.78 / 0.85
  - Accuracy: 0.85 on 61 test records
- ROC-AUC reported in one baseline ROC cell: 0.947

Saved artifacts:

- `model.weights.bestLoss.hdf5`
- `model.weights.bestAcc.hdf5`
- `HeartDis_bestLossSNN_model.0.868852436542511.h5`
- `HeartDis_bestAccSNN_model.0.868852436542511.h5`

### Larger Neural Network

Architecture:

- Dense 72
- Dense 64
- Dense 24
- Dense 2 softmax
- Total trainable parameters: 8,010

Reported results:

- Test loss: 0.5162
- Test accuracy: 0.8689
- MCC: 0.737
- Classification report:
  - Class 0 precision/recall/F1: 0.86 / 0.86 / 0.86
  - Class 1 precision/recall/F1: 0.88 / 0.88 / 0.88
- ROC-AUC reported: 0.932

### Tuned Neural Network

Tuning approach:

- Grid search over activation functions `selu`, `relu`, `elu`, `tanh` for three hidden layers.
- Grid search over optimizers `RMSprop`, `adam`, `nadam`, `sgd`.
- Model form:
  - Dense 72 with L1/L2 regularization
  - Dense 64 with L1/L2 regularization
  - Dense 24 with L1/L2 regularization
  - Dense 2 softmax
- Uses early stopping with `patience=10`.

Critical issue:

- The grid-search loop uses `validation_data=(X_test, y_test)`. This makes the held-out test set part of model selection and creates leakage into the reported final tuned-model result.

Reported chosen tuned model:

- File loaded: `kerSM_Test_tuned_HeartDis_model.relu.selu.elu.RMSprop.0.902.h5`
- Architecture: 72 -> 64 -> 24 -> 2
- Total trainable parameters: 8,010
- Test loss: 0.6072
- Test accuracy: 0.9016
- MCC: 0.804
- Classification report:
  - Class 0 precision/recall/F1: 0.93 / 0.86 / 0.89
  - Class 1 precision/recall/F1: 0.88 / 0.94 / 0.91
- ROC-AUC cell reported: 0.949
- Saved final model: `HeartDisSM_tuned.h5`

## Dataset Discrepancy Analysis

### Uploaded `heart.csv`

File hash:

- SHA-256: `ddb2996b2f4db2e00aad13f4518200179ff69f79093838e3c21ffa672ebec0f1`

Observed structure:

- Shape: 1,025 rows x 14 columns
- Columns: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`
- Missing values: none detected
- Exact duplicate rows: 723
- Unique rows after exact deduplication: 302
- Target counts before deduplication:
  - 0: 499
  - 1: 526
- Target counts after deduplication:
  - 0: 138
  - 1: 164

Interpretation:

The currently uploaded `heart.csv` is not the same file used by the notebook. It appears to be a duplicate-expanded Kaggle-style heart dataset with only 302 unique records. It should only be used as a supplementary benchmark after exact-row deduplication, and all results from the non-deduplicated 1,025-row version should be treated as at high risk of duplicate leakage under random splitting.

### Uploaded `heart_disease_uci.csv`

File hash:

- SHA-256: `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae`

Observed structure:

- Shape: 920 rows x 16 columns
- Columns: `id`, `age`, `sex`, `dataset`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalch`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `num`
- Centers:
  - Cleveland: 304
  - Hungary: 293
  - Switzerland: 123
  - VA Long Beach: 200
- Binary target after collapsing `num > 0` to disease presence:
  - Absence: 411
  - Presence: 509
- Multiclass `num` distribution:
  - 0: 411
  - 1: 265
  - 2: 109
  - 3: 107
  - 4: 28
- Missingness:
  - `ca`: 611
  - `thal`: 486
  - `slope`: 309
  - `fbs`: 90
  - `oldpeak`: 62
  - `trestbps`: 59
  - `thalch`: 55
  - `exang`: 55
  - `chol`: 30
  - `restecg`: 2
- Exact full-row duplicates: 0
- Duplicates ignoring `id`: 2

Interpretation:

This is the correct primary research dataset because it preserves the hospital/center variable required for internal-external validation. Missingness is substantial and center-dependent, especially for `ca`, `thal`, and `slope`, so imputation must be performed inside training folds only and center-specific missingness must be discussed.

### Schema Differences

Important schema differences:

- `heart.csv` uses `thalach`; `heart_disease_uci.csv` uses `thalch`.
- `heart.csv` uses `target`; `heart_disease_uci.csv` uses `num`.
- `heart.csv` uses numeric-coded categorical variables; `heart_disease_uci.csv` uses human-readable categorical strings.
- `heart_disease_uci.csv` contains `id` and `dataset`; `heart.csv` does not.
- `heart_disease_uci.csv` contains multiclass disease severity in `num`; `heart.csv` contains a binary target.

The canonical project schema should use:

- `age`
- `sex`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalach`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`
- `center`
- `target_binary`
- `target_multiclass`, available only where supported

### Target-Coding Warning

The `heart.csv` target cannot be assumed to mean the same thing as `num > 0` in `heart_disease_uci.csv`. Under a canonical mapping of Cleveland-style features, the uploaded `heart.csv` overlaps with Cleveland records but target alignment is ambiguous and appears inconsistent with the UCI `num > 0` definition for at least part of the overlap.

For the primary study, the safe target definition is:

- UCI primary outcome: `target_binary = 1 if num in {1,2,3,4}, else 0`.

For the supplementary `heart.csv` benchmark:

- Use the file's binary `target` only after clearly documenting that it is a dataset-specific benchmark label.
- Do not pool it with `heart_disease_uci.csv`.
- Do not use it as external validation for UCI unless target semantics are independently verified.

## Leakage-Risk Analysis

Identified leakage risks:

1. Duplicate leakage in uploaded `heart.csv`.
   - The file contains 723 exact duplicate rows. Random splitting before deduplication can put identical patients/records in train and test.

2. Test-set leakage during neural-network tuning.
   - The notebook grid-search loop uses `validation_data=(X_test, y_test)`, so the reported tuned-model test result is not a clean held-out estimate.

3. Feature selection outside resampling.
   - Chi-squared feature scoring and tree importances are computed on the full dataset before the split. Even if no features were removed, this pattern is unsafe if later used for selection.

4. One-hot encoding before train/test split.
   - The notebook encodes the whole dataset before splitting. For this dataset it mainly fixes category levels, but the safer approach is to fit encoders inside folds.

5. Label encoding and schema transformation outside a tracked pipeline.
   - Target encoding itself is not leakage, but transformations are not versioned or enforced.

6. Validation split inside Keras training after an external train/test split.
   - Acceptable for training control, but not enough for model selection. It also does not account for centers.

7. Site/center leakage.
   - The notebook does not use center identifiers and cannot estimate generalization across hospitals.

8. Possible target-coding leakage or semantic inversion.
   - `heart.csv` target semantics may not match the UCI `num > 0` disease definition.

9. Repeated manual model loading by filename.
   - Filename-encoded accuracies can encourage selecting artifacts based on test performance.

10. Calibration omitted from selection.
   - A model can show good accuracy/AUROC while providing poorly calibrated probabilities.

## Migration Plan

The notebook should be migrated as follows:

1. Treat `heart_disease_uci.csv` as the primary dataset and `heart.csv` as a supplementary benchmark only.
2. Copy raw files into `data/raw/` with source hashes recorded in a manifest.
3. Define a canonical feature dictionary and schema in `src/domain/`.
4. Implement CSV readers and schema standardization in `src/infrastructure/`.
5. Build sklearn-compatible preprocessing pipelines with imputation, encoding, and scaling inside cross-validation folds.
6. Implement site-aware leave-one-center-out internal-external validation using `dataset` as the grouping variable.
7. Implement inner group-aware model selection on development centers.
8. Compare penalized logistic regression, random forest, gradient boosting, calibrated gradient boosting, and MLP baselines.
9. Select the final model using AUROC, AUPRC, Brier score, calibration slope/intercept, site stability, and interpretability.
10. Generate all tables, figures, and manuscript numbers from versioned code.
11. Add an XAI layer that gives global and local explanations in clinically understandable language while avoiding causal claims.
12. Deploy the final selected model through FastAPI with one public prediction workflow.
13. Track experiments with MLflow and preserve serialized model/preprocessing artifacts.
14. Add tests, CI, Docker, DVC metadata, README, model card, manuscript draft, and supplementary materials.

## Recommended Research Plan

Primary research question:

Can a leakage-audited, calibration-aware machine-learning pipeline predict binary heart disease presence using public UCI heart disease data under center-aware internal-external validation?

Primary dataset:

- `heart_disease_uci.csv`

Primary validation:

- Leave-one-center-out internal-external validation:
  - Hold out Cleveland, train/select on remaining centers.
  - Hold out Hungary, train/select on remaining centers.
  - Hold out Switzerland, train/select on remaining centers.
  - Hold out VA Long Beach, train/select on remaining centers.

Secondary benchmark:

- Deduplicated `heart.csv`, reported only as supplementary.

Candidate models:

- Penalized logistic regression.
- Random forest.
- Gradient boosting.
- Calibrated gradient boosting.
- MLP baseline preserving the notebook's neural-network comparator role.

Primary selection criteria:

- AUROC and AUPRC for discrimination.
- Brier score, calibration intercept, and calibration slope for probabilistic reliability.
- Sensitivity, specificity, PPV, NPV, MCC, balanced accuracy, and F1 at prespecified thresholds.
- Performance stability across held-out centers.
- Interpretability and clinical plausibility.

Scientific caution:

This project should be framed as a reproducible public-dataset benchmark and educational decision-support prototype. It must not be framed as clinically deployment-ready or superior to clinicians without prospective validation.
