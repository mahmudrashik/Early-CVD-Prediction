# Leakage Audit Report

## Major Risks and Controls

1. **Duplicate leakage** in `heart.csv`.
   - Exact duplicate rows detected: 723
   - Control: supplementary benchmark uses exact-row deduplication before split.

2. **Test-set leakage** in the original notebook.
   - The notebook tuned neural-network variants using `validation_data=(X_test, y_test)`.
   - Control: the package uses inner group-aware cross-validation on development centres only.

3. **Preprocessing leakage**.
   - Control: imputation, scaling, one-hot encoding, and SMOTE are contained in
     sklearn/imblearn pipelines and fitted only inside training folds.

4. **Site leakage**.
   - Control: the centre variable is used for validation grouping and is not used
     as a model predictor.

5. **Target-definition leakage** or semantic mismatch.
   - Control: the primary target is defined only from UCI `num > 0`. The benchmark
     `heart.csv` is not pooled with UCI.

6. **Artifact traceability**.
   - Control: source file hashes, configuration, selected model metadata, and
     validation tables are written to `reports/` and `artifacts/`.

7. **SMOTE leakage**.
   - Control: SMOTE is applied via `imblearn.pipeline.Pipeline` inside training
     folds only. Test-fold samples are never synthesised.
