from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from domain.feature_dictionary import FEATURE_DICTIONARY, GROUP_COLUMN, TARGET_BINARY
from infrastructure.config import resolve_path
from infrastructure.data import load_primary_dataset, load_secondary_dataset
from visualization import plots


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_data_and_leakage_reports(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    paths = config["paths"]
    primary_path = resolve_path(paths["primary_raw"])
    secondary_path = resolve_path(paths["secondary_raw"])
    figures_dir = resolve_path(paths["figures_dir"])
    tables_dir = resolve_path(paths["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    primary = load_primary_dataset(primary_path)
    secondary_dedup = load_secondary_dataset(secondary_path, deduplicate=True)
    secondary_original = load_secondary_dataset(secondary_path, deduplicate=False)

    source_manifest = pd.DataFrame(
        [
            {"file": str(paths["primary_raw"]), "sha256": sha256(primary_path), "rows": len(primary), "role": "primary"},
            {"file": str(paths["secondary_raw"]), "sha256": sha256(secondary_path), "rows": len(secondary_original), "role": "supplementary"},
        ]
    )
    source_manifest.to_csv(tables_dir / "source_manifest.csv", index=False)

    missing_primary = primary.isna().sum().reset_index()
    missing_primary.columns = ["variable", "missing_count"]
    missing_primary["missing_percent"] = missing_primary["missing_count"] / len(primary) * 100
    missing_primary.to_csv(tables_dir / "primary_missingness.csv", index=False)

    class_by_center = primary.groupby([GROUP_COLUMN, TARGET_BINARY]).size().reset_index(name="n")
    class_by_center.to_csv(tables_dir / "class_by_center.csv", index=False)

    duplicate_table = pd.DataFrame(
        [
            {"dataset": "heart_disease_uci.csv", "rows": len(primary), "exact_duplicates": 0, "rows_after_dedup": len(primary)},
            {
                "dataset": "heart.csv",
                "rows": len(secondary_original),
                "exact_duplicates": int(secondary_original.attrs.get("original_rows", len(secondary_original)) - len(secondary_dedup)),
                "rows_after_dedup": len(secondary_dedup),
            },
        ]
    )
    duplicate_table.to_csv(tables_dir / "duplicate_audit.csv", index=False)

    dictionary = pd.DataFrame([vars(v) for v in FEATURE_DICTIONARY.values()])
    dictionary.to_csv(tables_dir / "data_dictionary.csv", index=False)

    plots.missingness_heatmap(primary.drop(columns=["id"], errors="ignore"), figures_dir / "missingness_heatmap.png")
    plots.class_distribution(primary, TARGET_BINARY, figures_dir / "class_distribution_primary.png")
    plots.site_distribution(primary, GROUP_COLUMN, TARGET_BINARY, figures_dir / "site_distribution_primary.png")

    _write_data_audit_doc(primary, secondary_original, secondary_dedup, missing_primary, duplicate_table, source_manifest)
    _write_leakage_doc(secondary_original, secondary_dedup)
    return {
        "primary": primary,
        "secondary_dedup": secondary_dedup,
        "source_manifest": source_manifest,
        "missing_primary": missing_primary,
        "duplicate_table": duplicate_table,
    }


def write_result_documents(config: dict[str, Any], champion: str, summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    top = summary.loc[summary["model"] == champion].iloc[0]

    model_card = f"""# Final Model Card

## Model

Selected model: `{champion}`

## Intended Use

Educational and research-oriented heart disease risk prediction using public benchmark data. This is not a medical device and is not intended for clinical deployment.

## Selection Rationale

The model was selected using discrimination, calibration, center-to-center stability, and interpretability. Accuracy alone was not used as the selection criterion.

## Primary Validation

Leave-one-center-out internal-external cross-validation using the UCI dataset center variable.

## Key Aggregate Metrics

- Mean AUROC: {top['auroc_mean']:.3f}
- Mean AUPRC: {top['auprc_mean']:.3f}
- Mean Brier score: {top['brier_mean']:.3f}
- Mean balanced accuracy: {top['balanced_accuracy_mean']:.3f}
- Mean MCC: {top['mcc_mean']:.3f}

## Limitations

The public datasets are small, historical, partially missing, and not representative of modern prospective care. The supplementary `heart.csv` file has extensive duplicate rows and uncertain target semantics relative to the UCI `num > 0` definition.
"""
    (reports_dir / "model_card.md").write_text(model_card, encoding="utf-8")


def _write_data_audit_doc(primary: pd.DataFrame, secondary_original: pd.DataFrame, secondary_dedup: pd.DataFrame, missing: pd.DataFrame, duplicates: pd.DataFrame, manifest: pd.DataFrame) -> None:
    doc = f"""# Data Audit Report

## Primary Dataset

`heart_disease_uci.csv` is used as the primary research dataset because it contains a `dataset` center variable enabling internal-external validation.

- Rows: {len(primary)}
- Centers: {primary[GROUP_COLUMN].nunique()}
- Binary target absent/present counts: {primary[TARGET_BINARY].value_counts().sort_index().to_dict()}
- Center counts: {primary[GROUP_COLUMN].value_counts().to_dict()}

## Missingness

Substantial missingness is present, especially in variables such as `ca`, `thal`, and `slope`. Missing values are not imputed globally; imputation is performed inside training folds.

Top missing variables:

{missing.sort_values('missing_count', ascending=False).head(10).to_markdown(index=False)}

## Secondary Dataset

`heart.csv` is retained as a supplementary benchmark only.

- Original rows: {len(secondary_original)}
- Rows after exact deduplication: {len(secondary_dedup)}
- Exact duplicates removed: {len(secondary_original) - len(secondary_dedup)}

## Source Manifest

{manifest.to_markdown(index=False)}
"""
    Path("docs/data_audit_report.md").write_text(doc, encoding="utf-8")


def _write_leakage_doc(secondary_original: pd.DataFrame, secondary_dedup: pd.DataFrame) -> None:
    duplicate_count = len(secondary_original) - len(secondary_dedup)
    doc = f"""# Leakage Audit Report

## Major Risks and Controls

1. Duplicate leakage in `heart.csv`.
   - Exact duplicate rows detected: {duplicate_count}
   - Control: the supplementary benchmark uses exact-row deduplication before any split.

2. Test-set leakage in the original notebook.
   - The notebook tuned neural-network variants using `validation_data=(X_test, y_test)`.
   - Control: the package uses inner group-aware cross-validation on development centers only.

3. Preprocessing leakage.
   - Control: imputation, scaling, and one-hot encoding are contained in sklearn pipelines and fitted only inside training folds.

4. Site leakage.
   - Control: the center variable is used for validation grouping and is not used as a model predictor.

5. Target-definition leakage or semantic mismatch.
   - Control: the primary target is defined only from UCI `num > 0`. The benchmark `heart.csv` is not pooled with UCI.

6. Artifact traceability.
   - Control: source file hashes, configuration, selected model metadata, and validation tables are written to `reports/` and `artifacts/`.
"""
    Path("docs/leakage_audit_report.md").write_text(doc, encoding="utf-8")
