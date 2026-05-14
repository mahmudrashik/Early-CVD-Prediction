from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold

from domain.feature_dictionary import FEATURE_COLUMNS, GROUP_COLUMN, TARGET_BINARY
from evaluation.decision_curve import decision_curve
from evaluation.explain import global_importance
from evaluation.metrics import bootstrap_ci, classification_metrics, threshold_table
from evaluation.selection import select_champion
from infrastructure.config import ensure_directories, resolve_path
from infrastructure.data import feature_target_group, load_primary_dataset, load_secondary_dataset
from infrastructure.persistence import save_bundle
from infrastructure.tracking import log_metrics, mlflow_run
from models.factory import candidate_models, make_pipeline
from pipelines.preprocessing import build_preprocessor
from visualization import plots
from application.reporting import write_data_and_leakage_reports, write_result_documents


def train_and_evaluate(config: dict[str, Any]) -> dict[str, Any]:
    ensure_directories(config)
    seed = int(config["random_seed"])
    threshold = float(config["validation"]["threshold"])
    primary = load_primary_dataset(resolve_path(config["paths"]["primary_raw"]))

    write_data_and_leakage_reports(config)

    outer_metrics, predictions, best_params = _site_aware_validation(primary, config)
    tables_dir = resolve_path(config["paths"]["tables_dir"])
    figures_dir = resolve_path(config["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    center_metrics = pd.DataFrame(outer_metrics)
    center_metrics.to_csv(tables_dir / "center_metrics.csv", index=False)
    prediction_df = pd.DataFrame(predictions)
    prediction_df.to_csv(tables_dir / "outer_predictions.csv", index=False)
    best_params_df = pd.DataFrame(best_params)
    best_params_df.to_csv(tables_dir / "inner_model_selection.csv", index=False)

    summary = _summarize_metrics(center_metrics)
    champion, ranked = select_champion(summary)
    ranked.to_csv(tables_dir / "model_selection_ranking.csv", index=False)
    summary.to_csv(tables_dir / "model_comparison.csv", index=False)

    champion_predictions = prediction_df[prediction_df["model"] == champion].copy()
    y_true = champion_predictions["y_true"].to_numpy(dtype=int)
    y_prob = champion_predictions["y_prob"].to_numpy(dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    boot = bootstrap_ci(
        y_true,
        y_prob,
        threshold=threshold,
        iterations=int(config["validation"]["bootstrap_iterations"]),
        seed=seed,
    )
    boot.to_csv(tables_dir / "bootstrap_confidence_intervals.csv", index=False)
    thresholds = [float(x) for x in config["validation"]["thresholds"]]
    threshold_df = threshold_table(y_true, y_prob, thresholds)
    threshold_df.to_csv(tables_dir / "threshold_analysis.csv", index=False)
    dca = decision_curve(y_true, y_prob, thresholds)
    dca.to_csv(tables_dir / "decision_curve.csv", index=False)

    plots.model_comparison(summary, figures_dir / "model_comparison.png")
    plots.roc_pr_curves(prediction_df, figures_dir / "roc_curves.png", figures_dir / "precision_recall_curves.png")
    plots.calibration_plot(prediction_df, figures_dir / "calibration_plots.png")
    plots.confusion_matrix_plot(y_true, y_pred, figures_dir / "confusion_matrix_champion.png")
    plots.decision_curve_plot(dca, figures_dir / "decision_curve.png")

    final_model, final_params = _fit_final_model(primary, champion, config)
    X, y, _ = feature_target_group(primary)
    importance = global_importance(final_model, X, y, seed)
    importance.to_csv(tables_dir / "global_feature_importance.csv", index=False)
    if not plots.shap_summary_plot(final_model, X, figures_dir / "shap_summary.png"):
        plots.importance_plot(importance, figures_dir / "shap_summary.png")
    plots.importance_plot(importance, figures_dir / "global_feature_importance.png")

    metadata = {
        "project_name": config["project_name"],
        "model_name": champion,
        "final_params": final_params,
        "selection_summary": ranked.to_dict(orient="records"),
        "validation_summary": summary.to_dict(orient="records"),
        "bootstrap_ci": boot.to_dict(orient="records"),
        "feature_columns": FEATURE_COLUMNS,
        "global_importance": importance.to_dict(orient="records"),
        "target_definition": "target_binary = 1 if UCI num > 0 else 0",
        "validation_design": "leave-one-center-out internal-external cross-validation",
        "threshold": threshold,
        "disclaimer": config["reporting"]["caution"],
    }
    bundle = {"model": final_model, "metadata": metadata}
    model_bundle_path = resolve_path(config["paths"]["model_bundle"])
    save_bundle(bundle, model_bundle_path)
    (resolve_path(config["paths"]["artifacts_dir"]) / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_result_documents(config, champion, ranked, boot)
    _write_supplementary_benchmark(primary, champion, config)

    with mlflow_run(config["project_name"], f"final_{champion}", resolve_path(config["paths"]["mlruns_dir"])):
        champion_row = ranked[ranked["model"] == champion].iloc[0]
        log_metrics({k: champion_row[k] for k in champion_row.index if k.endswith("_mean") and isinstance(champion_row[k], (int, float, np.floating))})

    return {"champion": champion, "model_bundle": str(model_bundle_path), "summary": ranked}


def _site_aware_validation(df: pd.DataFrame, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(config["random_seed"])
    threshold = float(config["validation"]["threshold"])
    included = set(config["models"]["include"])
    specs = {k: v for k, v in candidate_models(seed).items() if k in included}
    centers = sorted(df[GROUP_COLUMN].dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    best_params: list[dict[str, Any]] = []

    for holdout_center in centers:
        train_df = df[df[GROUP_COLUMN].astype(str) != holdout_center].copy()
        test_df = df[df[GROUP_COLUMN].astype(str) == holdout_center].copy()
        X_train, y_train, groups_train = feature_target_group(train_df)
        X_test, y_test, _ = feature_target_group(test_df)
        n_groups = groups_train.nunique()
        inner_cv = GroupKFold(n_splits=max(2, min(3, n_groups)))

        for model_name, spec in specs.items():
            pipeline = make_pipeline(build_preprocessor(), spec.estimator)
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=spec.param_grid if spec.param_grid else [{}],
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=1,
                refit=True,
                error_score=np.nan,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_train, y_train, groups=groups_train)
            best = search.best_estimator_
            y_prob = best.predict_proba(X_test)[:, 1]
            metric_row = classification_metrics(y_test.to_numpy(), y_prob, threshold)
            metric_row.update({"model": model_name, "held_out_center": holdout_center, "inner_best_score": float(search.best_score_) if search.best_score_ == search.best_score_ else np.nan})
            rows.append(metric_row)
            best_params.append({"model": model_name, "held_out_center": holdout_center, "best_params": json.dumps(search.best_params_), "inner_best_score": metric_row["inner_best_score"]})
            for idx, truth, prob in zip(test_df.index, y_test, y_prob):
                predictions.append({"model": model_name, "held_out_center": holdout_center, "row_index": int(idx), "y_true": int(truth), "y_prob": float(prob)})
    return rows, predictions, best_params


def _summarize_metrics(center_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "auroc",
        "auprc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision_ppv",
        "npv",
        "f1",
        "mcc",
        "brier",
        "calibration_intercept",
        "calibration_slope",
    ]
    summary = center_metrics.groupby("model")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["model"] + [f"{metric}_{stat}" for metric, stat in summary.columns[1:]]
    return summary


def _fit_final_model(df: pd.DataFrame, model_name: str, config: dict[str, Any]) -> tuple[object, dict[str, Any]]:
    seed = int(config["random_seed"])
    spec = candidate_models(seed)[model_name]
    X, y, groups = feature_target_group(df)
    inner_cv = GroupKFold(n_splits=max(2, min(4, groups.nunique())))
    pipeline = make_pipeline(build_preprocessor(), spec.estimator)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=spec.param_grid if spec.param_grid else [{}],
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        error_score=np.nan,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


def _write_supplementary_benchmark(primary: pd.DataFrame, champion: str, config: dict[str, Any]) -> None:
    secondary = load_secondary_dataset(resolve_path(config["paths"]["secondary_raw"]), deduplicate=True)
    secondary_original = load_secondary_dataset(resolve_path(config["paths"]["secondary_raw"]), deduplicate=False)
    tables_dir = resolve_path(config["paths"]["tables_dir"])
    rows = [
        {"dataset": "primary_uci", "rows": len(primary), "target_absent": int((primary[TARGET_BINARY] == 0).sum()), "target_present": int((primary[TARGET_BINARY] == 1).sum()), "note": "primary, site-aware"},
        {
            "dataset": "heart_csv_deduplicated",
            "rows": len(secondary),
            "target_absent": int((secondary[TARGET_BINARY] == 0).sum()),
            "target_present": int((secondary[TARGET_BINARY] == 1).sum()),
            "note": "supplementary only; target semantics not pooled with UCI",
        },
    ]
    pd.DataFrame(rows).to_csv(tables_dir / "sensitivity_dataset_summary.csv", index=False)
    sensitivity = []
    for label, dataset in [("heart_csv_original_with_duplicates", secondary_original), ("heart_csv_deduplicated", secondary)]:
        fold_rows = _supplementary_cv(dataset, champion, config)
        for row in fold_rows:
            row["dataset_version"] = label
        sensitivity.extend(fold_rows)
    sensitivity_df = pd.DataFrame(sensitivity)
    sensitivity_df.to_csv(tables_dir / "duplicate_sensitivity.csv", index=False)
    summary = sensitivity_df.groupby("dataset_version")[["auroc", "auprc", "balanced_accuracy", "brier", "mcc"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["dataset_version"] + [f"{metric}_{stat}" for metric, stat in summary.columns[1:]]
    summary["note"] = "Supplementary only; target semantics are not pooled with UCI."
    summary.to_csv(tables_dir / "supplementary_heart_csv_benchmark.csv", index=False)


def _supplementary_cv(df: pd.DataFrame, model_name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(config["random_seed"])
    threshold = float(config["validation"]["threshold"])
    spec = candidate_models(seed)[model_name]
    X, y, _ = feature_target_group(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model = make_pipeline(build_preprocessor(), clone(spec.estimator))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        row = classification_metrics(y.iloc[test_idx].to_numpy(), y_prob, threshold)
        row.update({"model": model_name, "fold": fold})
        rows.append(row)
    return rows
