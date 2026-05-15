"""Training & evaluation pipeline — site-aware cross-validation.

Orchestrates the complete experiment lifecycle:

1. Data loading & audit reports.
2. Leave-one-centre-out internal–external cross-validation with all
   candidate models (including XGBoost, LightGBM, CatBoost, SVM,
   Stacking Ensemble).
3. Champion selection via multi-criteria ranking.
4. Statistical significance testing (DeLong, Friedman, Nemenyi, McNemar).
5. IEEE-standard figures and tables.
6. Final model fitting with the full dataset.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV, StratifiedKFold

from domain.feature_dictionary import FEATURE_COLUMNS, GROUP_COLUMN, NUMERIC_FEATURES, TARGET_BINARY
from evaluation.decision_curve import decision_curve
from evaluation.explain import global_importance
from evaluation.metrics import bootstrap_ci, classification_metrics, threshold_table
from evaluation.selection import select_champion
from evaluation.statistical_tests import run_all_statistical_tests
from infrastructure.config import ensure_directories, resolve_path
from infrastructure.data import feature_target_group, load_primary_dataset, load_secondary_dataset
from infrastructure.persistence import save_bundle
from infrastructure.tracking import log_metrics, mlflow_run
from models.factory import candidate_models, make_pipeline
from pipelines.preprocessing import build_preprocessor
from visualization import plots
from application.reporting import write_data_and_leakage_reports, write_result_documents


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full training, evaluation, and reporting pipeline."""
    ensure_directories(config)
    seed = int(config["random_seed"])
    threshold = float(config["validation"]["threshold"])
    use_smote = config.get("preprocessing", {}).get("smote", False)
    feat_eng = config.get("preprocessing", {}).get("feature_engineering", False)
    smote_k = config.get("preprocessing", {}).get("smote_k_neighbors", 3)
    primary = load_primary_dataset(resolve_path(config["paths"]["primary_raw"]))

    write_data_and_leakage_reports(config)

    tables_dir = resolve_path(config["paths"]["tables_dir"])
    figures_dir = resolve_path(config["paths"]["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Data exploration figures ──────────────────────────────────────────
    X_full, y_full, _ = feature_target_group(primary)
    plots.correlation_heatmap(X_full.select_dtypes(include=[np.number]).assign(target=y_full),
                              figures_dir / "correlation_heatmap.png")
    plots.feature_distributions(primary, TARGET_BINARY, NUMERIC_FEATURES,
                                figures_dir / "feature_distributions.png")

    # ── Site-aware cross-validation ──────────────────────────────────────
    outer_metrics, predictions, best_params = _site_aware_validation(
        primary, config, use_smote=use_smote, feat_eng=feat_eng, smote_k=smote_k,
    )

    center_metrics = pd.DataFrame(outer_metrics)
    center_metrics.to_csv(tables_dir / "center_metrics.csv", index=False)
    prediction_df = pd.DataFrame(predictions)
    prediction_df.to_csv(tables_dir / "outer_predictions.csv", index=False)
    best_params_df = pd.DataFrame(best_params)
    best_params_df.to_csv(tables_dir / "inner_model_selection.csv", index=False)

    # ── Model comparison & champion selection ────────────────────────────
    summary = _summarize_metrics(center_metrics)
    champion, ranked = select_champion(summary)
    ranked.to_csv(tables_dir / "model_selection_ranking.csv", index=False)
    summary.to_csv(tables_dir / "model_comparison.csv", index=False)

    # Detailed per-fold table
    center_metrics.to_csv(tables_dir / "cross_validation_detailed.csv", index=False)

    # ── Champion predictions ─────────────────────────────────────────────
    champ_preds = prediction_df[prediction_df["model"] == champion].copy()
    y_true = champ_preds["y_true"].to_numpy(dtype=int)
    y_prob = champ_preds["y_prob"].to_numpy(dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # ── Bootstrap CI ─────────────────────────────────────────────────────
    boot = bootstrap_ci(y_true, y_prob, threshold=threshold,
                        iterations=int(config["validation"]["bootstrap_iterations"]), seed=seed)
    boot.to_csv(tables_dir / "bootstrap_confidence_intervals.csv", index=False)

    # ── Threshold analysis ───────────────────────────────────────────────
    thresholds = [float(x) for x in config["validation"]["thresholds"]]
    threshold_df = threshold_table(y_true, y_prob, thresholds)
    threshold_df.to_csv(tables_dir / "threshold_analysis.csv", index=False)

    # ── Decision curve ───────────────────────────────────────────────────
    dca = decision_curve(y_true, y_prob, thresholds)
    dca.to_csv(tables_dir / "decision_curve.csv", index=False)

    # ── Statistical significance tests ───────────────────────────────────
    stat_results = _run_statistical_tests(prediction_df, center_metrics, threshold, tables_dir)

    # ── Comprehensive model comparison table with 95% CI ─────────────────
    _write_comprehensive_comparison(prediction_df, threshold, config, tables_dir)

    # ── Hyperparameter summary ───────────────────────────────────────────
    best_params_df.to_csv(tables_dir / "hyperparameter_summary.csv", index=False)

    # ── Figures ──────────────────────────────────────────────────────────
    plots.model_comparison(summary, figures_dir / "model_comparison.png")
    plots.model_comparison_multi_metric(summary, figures_dir / "model_comparison_multi_metric.png")
    plots.radar_chart(summary, figures_dir / "radar_chart.png")
    plots.roc_pr_curves(prediction_df, figures_dir / "roc_curves.png",
                        figures_dir / "precision_recall_curves.png")
    plots.calibration_plot(prediction_df, figures_dir / "calibration_plots.png")
    plots.confusion_matrix_plot(y_true, y_pred, figures_dir / "confusion_matrix_champion.png")
    plots.decision_curve_plot(dca, figures_dir / "decision_curve.png")
    plots.threshold_sensitivity_plot(threshold_df, figures_dir / "threshold_sensitivity.png")
    plots.predicted_probability_violin(y_true, y_prob, figures_dir / "predicted_probability_violin.png")

    # Box plots for key metrics
    for metric in ["auroc", "auprc", "f1", "mcc", "balanced_accuracy"]:
        if metric in center_metrics.columns:
            plots.boxplot_comparison(center_metrics, metric,
                                    figures_dir / f"boxplot_{metric}.png")

    # Confusion matrix grid (all models)
    model_preds_dict = {}
    for model_name, grp in prediction_df.groupby("model"):
        model_preds_dict[model_name] = (grp["y_prob"].values >= threshold).astype(int)
    all_y_true = prediction_df.groupby("model").first().reset_index()
    y_true_for_grid = prediction_df[prediction_df["model"] == champion]["y_true"].values.astype(int)
    plots.confusion_matrix_grid(y_true_for_grid, {
        m: prediction_df[prediction_df["model"] == m]["y_prob"].values >= threshold
        for m in prediction_df["model"].unique()
        if len(prediction_df[prediction_df["model"] == m]) == len(y_true_for_grid)
    }, figures_dir / "confusion_matrix_grid.png")

    # Statistical figures
    if "delong" in stat_results and not stat_results["delong"].empty:
        plots.delong_heatmap(stat_results["delong"], figures_dir / "delong_heatmap.png")
    if "nemenyi" in stat_results and isinstance(stat_results["nemenyi"], pd.DataFrame) and not stat_results["nemenyi"].empty:
        plots.critical_difference_diagram(stat_results["nemenyi"], figures_dir / "critical_difference_diagram.png")

    # ── Final model fitting ──────────────────────────────────────────────
    final_model, final_params = _fit_final_model(primary, champion, config,
                                                  use_smote=use_smote, feat_eng=feat_eng, smote_k=smote_k)
    X, y, _ = feature_target_group(primary)
    importance = global_importance(final_model, X, y, seed)
    importance.to_csv(tables_dir / "global_feature_importance.csv", index=False)

    if not plots.shap_summary_plot(final_model, X, figures_dir / "shap_summary.png"):
        plots.importance_plot(importance, figures_dir / "shap_summary.png")
    plots.importance_plot(importance, figures_dir / "global_feature_importance.png")
    plots.learning_curve_plot(final_model, X, y, figures_dir / "learning_curves.png")

    # Feature correlation matrix table
    X.select_dtypes(include=[np.number]).corr(method="spearman").to_csv(
        tables_dir / "feature_correlation_matrix.csv")

    # ── Metadata & bundle ────────────────────────────────────────────────
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
        "preprocessing": {"smote": use_smote, "feature_engineering": feat_eng},
        "disclaimer": config["reporting"]["caution"],
    }
    bundle = {"model": final_model, "metadata": metadata}
    model_bundle_path = resolve_path(config["paths"]["model_bundle"])
    save_bundle(bundle, model_bundle_path)
    (resolve_path(config["paths"]["artifacts_dir"]) / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    write_result_documents(config, champion, ranked, boot)
    _write_supplementary_benchmark(primary, champion, config,
                                    use_smote=use_smote, feat_eng=feat_eng, smote_k=smote_k)

    with mlflow_run(config["project_name"], f"final_{champion}", resolve_path(config["paths"]["mlruns_dir"])):
        champion_row = ranked[ranked["model"] == champion].iloc[0]
        log_metrics({k: champion_row[k] for k in champion_row.index
                     if k.endswith("_mean") and isinstance(champion_row[k], (int, float, np.floating))})

    return {"champion": champion, "model_bundle": str(model_bundle_path), "summary": ranked}


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _site_aware_validation(
    df: pd.DataFrame, config: dict[str, Any], *,
    use_smote: bool = False, feat_eng: bool = False, smote_k: int = 3,
) -> tuple[list, list, list]:
    """Leave-one-centre-out cross-validation with all candidate models."""
    seed = int(config["random_seed"])
    threshold = float(config["validation"]["threshold"])
    opt_cfg = config.get("optimization", {})
    n_iter = int(opt_cfg.get("randomized_n_iter", 60))
    included = set(config["models"]["include"])
    specs = {k: v for k, v in candidate_models(seed).items() if k in included}
    centers = sorted(df[GROUP_COLUMN].dropna().astype(str).unique())

    rows, predictions, best_params = [], [], []

    for holdout_center in centers:
        train_df = df[df[GROUP_COLUMN].astype(str) != holdout_center].copy()
        test_df = df[df[GROUP_COLUMN].astype(str) == holdout_center].copy()
        X_train, y_train, groups_train = feature_target_group(train_df)
        X_test, y_test, _ = feature_target_group(test_df)
        n_groups = groups_train.nunique()
        inner_cv = GroupKFold(n_splits=max(2, min(3, n_groups)))

        for model_name, spec in specs.items():
            pipeline = make_pipeline(
                build_preprocessor(feature_engineering=feat_eng),
                spec.estimator,
                use_smote=use_smote, smote_k=smote_k, seed=seed,
            )
            # Choose search strategy based on param grid size
            total_combos = 1
            for v in spec.param_grid.values():
                total_combos *= len(v)

            if not spec.param_grid or total_combos <= 1:
                search = GridSearchCV(
                    estimator=pipeline, param_grid=[{}],
                    scoring="roc_auc", cv=inner_cv, n_jobs=1,
                    refit=True, error_score=np.nan,
                )
            elif total_combos <= 30:
                search = GridSearchCV(
                    estimator=pipeline, param_grid=spec.param_grid,
                    scoring="roc_auc", cv=inner_cv, n_jobs=1,
                    refit=True, error_score=np.nan,
                )
            else:
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=spec.param_grid,
                    n_iter=min(n_iter, spec.search_budget),
                    scoring="roc_auc", cv=inner_cv, n_jobs=1,
                    refit=True, error_score=np.nan, random_state=seed,
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_train, y_train, groups=groups_train)

            best = search.best_estimator_
            y_prob = best.predict_proba(X_test)[:, 1]
            metric_row = classification_metrics(y_test.to_numpy(), y_prob, threshold)
            metric_row.update({
                "model": model_name,
                "held_out_center": holdout_center,
                "inner_best_score": float(search.best_score_) if search.best_score_ == search.best_score_ else np.nan,
            })
            rows.append(metric_row)
            best_params.append({
                "model": model_name, "held_out_center": holdout_center,
                "best_params": json.dumps(search.best_params_),
                "inner_best_score": metric_row["inner_best_score"],
            })
            for idx, truth, prob in zip(test_df.index, y_test, y_prob):
                predictions.append({
                    "model": model_name, "held_out_center": holdout_center,
                    "row_index": int(idx), "y_true": int(truth), "y_prob": float(prob),
                })
    return rows, predictions, best_params


def _summarize_metrics(center_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "auroc", "auprc", "accuracy", "balanced_accuracy", "sensitivity",
        "specificity", "precision_ppv", "npv", "f1", "mcc", "brier",
        "calibration_intercept", "calibration_slope",
    ]
    available = [c for c in metric_cols if c in center_metrics.columns]
    summary = center_metrics.groupby("model")[available].agg(["mean", "std"]).reset_index()
    summary.columns = ["model"] + [f"{m}_{s}" for m, s in summary.columns[1:]]
    return summary


def _fit_final_model(
    df: pd.DataFrame, model_name: str, config: dict[str, Any], *,
    use_smote: bool = False, feat_eng: bool = False, smote_k: int = 3,
) -> tuple[object, dict[str, Any]]:
    seed = int(config["random_seed"])
    opt_cfg = config.get("optimization", {})
    n_iter = int(opt_cfg.get("randomized_n_iter", 60))
    spec = candidate_models(seed)[model_name]
    X, y, groups = feature_target_group(df)
    inner_cv = GroupKFold(n_splits=max(2, min(4, groups.nunique())))
    pipeline = make_pipeline(
        build_preprocessor(feature_engineering=feat_eng),
        spec.estimator, use_smote=use_smote, smote_k=smote_k, seed=seed,
    )

    total_combos = 1
    for v in spec.param_grid.values():
        total_combos *= len(v)

    if not spec.param_grid or total_combos <= 1:
        search = GridSearchCV(
            estimator=pipeline, param_grid=[{}], scoring="roc_auc",
            cv=inner_cv, n_jobs=1, refit=True, error_score=np.nan,
        )
    elif total_combos <= 30:
        search = GridSearchCV(
            estimator=pipeline, param_grid=spec.param_grid,
            scoring="roc_auc", cv=inner_cv, n_jobs=1, refit=True, error_score=np.nan,
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipeline, param_distributions=spec.param_grid,
            n_iter=min(n_iter, spec.search_budget),
            scoring="roc_auc", cv=inner_cv, n_jobs=1,
            refit=True, error_score=np.nan, random_state=seed,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


def _run_statistical_tests(
    prediction_df: pd.DataFrame,
    center_metrics: pd.DataFrame,
    threshold: float,
    tables_dir: Path,
) -> dict[str, Any]:
    """Run DeLong, Friedman, Nemenyi, and McNemar tests."""
    models = prediction_df["model"].unique()
    # Check all models have same y_true
    ref_model = models[0]
    ref_y = prediction_df[prediction_df["model"] == ref_model].sort_values("row_index")["y_true"].values

    model_probs, model_preds = {}, {}
    valid_models = []
    for m in models:
        grp = prediction_df[prediction_df["model"] == m].sort_values("row_index")
        if len(grp) == len(ref_y):
            model_probs[m] = grp["y_prob"].values.astype(float)
            model_preds[m] = (model_probs[m] >= threshold).astype(int)
            valid_models.append(m)

    if len(valid_models) < 2:
        return {}

    y_true = ref_y.astype(int)

    # Build fold metric matrix for Friedman test
    fold_pivot = center_metrics.pivot_table(index="held_out_center", columns="model", values="auroc")
    fold_pivot = fold_pivot[[c for c in valid_models if c in fold_pivot.columns]].dropna()

    results = run_all_statistical_tests(y_true, model_probs, fold_pivot, threshold)

    # Save tables
    if isinstance(results.get("delong"), pd.DataFrame):
        results["delong"].to_csv(tables_dir / "delong_pairwise.csv", index=False)
    if isinstance(results.get("friedman"), dict):
        pd.DataFrame([results["friedman"]]).to_csv(tables_dir / "friedman_test.csv", index=False)
    if isinstance(results.get("nemenyi"), pd.DataFrame) and not results["nemenyi"].empty:
        results["nemenyi"].to_csv(tables_dir / "nemenyi_posthoc.csv", index=False)
    if isinstance(results.get("mcnemar"), pd.DataFrame):
        results["mcnemar"].to_csv(tables_dir / "mcnemar_pairwise.csv", index=False)

    # Combined statistical tests summary
    stat_summary_rows = []
    if isinstance(results.get("friedman"), dict):
        stat_summary_rows.append({"test": "Friedman", **results["friedman"]})
    stat_df = pd.DataFrame(stat_summary_rows) if stat_summary_rows else pd.DataFrame()
    stat_df.to_csv(tables_dir / "statistical_tests.csv", index=False)

    return results


def _write_comprehensive_comparison(
    prediction_df: pd.DataFrame, threshold: float,
    config: dict[str, Any], tables_dir: Path,
) -> None:
    """Write a comprehensive model comparison table with 95% CI for all models."""
    seed = int(config["random_seed"])
    iters = int(config["validation"]["bootstrap_iterations"])
    rows = []
    for model_name, grp in prediction_df.groupby("model"):
        y_t = grp["y_true"].values.astype(int)
        y_p = grp["y_prob"].values.astype(float)
        ci = bootstrap_ci(y_t, y_p, threshold=threshold, iterations=iters, seed=seed)
        row = {"model": model_name}
        for _, ci_row in ci.iterrows():
            m = ci_row["metric"]
            row[f"{m}"] = ci_row["estimate"]
            row[f"{m}_ci_lower"] = ci_row["ci_lower"]
            row[f"{m}_ci_upper"] = ci_row["ci_upper"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables_dir / "comprehensive_model_comparison.csv", index=False)


def _write_supplementary_benchmark(
    primary: pd.DataFrame, champion: str, config: dict[str, Any], *,
    use_smote: bool = False, feat_eng: bool = False, smote_k: int = 3,
) -> None:
    secondary = load_secondary_dataset(resolve_path(config["paths"]["secondary_raw"]), deduplicate=True)
    secondary_original = load_secondary_dataset(resolve_path(config["paths"]["secondary_raw"]), deduplicate=False)
    tables_dir = resolve_path(config["paths"]["tables_dir"])
    rows = [
        {"dataset": "primary_uci", "rows": len(primary),
         "target_absent": int((primary[TARGET_BINARY] == 0).sum()),
         "target_present": int((primary[TARGET_BINARY] == 1).sum()),
         "note": "primary, site-aware"},
        {"dataset": "heart_csv_deduplicated", "rows": len(secondary),
         "target_absent": int((secondary[TARGET_BINARY] == 0).sum()),
         "target_present": int((secondary[TARGET_BINARY] == 1).sum()),
         "note": "supplementary only; target semantics not pooled with UCI"},
    ]
    pd.DataFrame(rows).to_csv(tables_dir / "sensitivity_dataset_summary.csv", index=False)
    sensitivity = []
    for label, dataset in [("heart_csv_original_with_duplicates", secondary_original),
                           ("heart_csv_deduplicated", secondary)]:
        fold_rows = _supplementary_cv(dataset, champion, config,
                                       use_smote=use_smote, feat_eng=feat_eng, smote_k=smote_k)
        for row in fold_rows:
            row["dataset_version"] = label
        sensitivity.extend(fold_rows)
    sensitivity_df = pd.DataFrame(sensitivity)
    sensitivity_df.to_csv(tables_dir / "duplicate_sensitivity.csv", index=False)
    summary = sensitivity_df.groupby("dataset_version")[
        ["auroc", "auprc", "balanced_accuracy", "brier", "mcc"]
    ].agg(["mean", "std"]).reset_index()
    summary.columns = ["dataset_version"] + [f"{m}_{s}" for m, s in summary.columns[1:]]
    summary["note"] = "Supplementary only; target semantics are not pooled with UCI."
    summary.to_csv(tables_dir / "supplementary_heart_csv_benchmark.csv", index=False)


def _supplementary_cv(
    df: pd.DataFrame, model_name: str, config: dict[str, Any], *,
    use_smote: bool = False, feat_eng: bool = False, smote_k: int = 3,
) -> list[dict[str, Any]]:
    seed = int(config["random_seed"])
    threshold = float(config["validation"]["threshold"])
    spec = candidate_models(seed)[model_name]
    X, y, _ = feature_target_group(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model = make_pipeline(
            build_preprocessor(feature_engineering=feat_eng),
            clone(spec.estimator),
            use_smote=use_smote, smote_k=smote_k, seed=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        row = classification_metrics(y.iloc[test_idx].to_numpy(), y_prob, threshold)
        row.update({"model": model_name, "fold": fold})
        rows.append(row)
    return rows
