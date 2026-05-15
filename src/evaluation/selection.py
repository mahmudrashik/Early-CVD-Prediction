"""Champion model selection via multi-criteria ranking.

The selection considers discrimination (AUROC, AUPRC, MCC), calibration
(Brier score, calibration slope), stability (AUROC std across centres),
and an interpretability bonus for clinically transparent models.
"""
from __future__ import annotations

import pandas as pd


def select_champion(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Select a champion with calibration and stability considered alongside discrimination."""
    ranked = summary.copy()
    ranked["calibration_slope_error"] = (ranked["calibration_slope_mean"] - 1.0).abs()

    # Rank each criterion (lower rank = better)
    ranked["auroc_rank"] = ranked["auroc_mean"].rank(ascending=False, method="min")
    ranked["auprc_rank"] = ranked["auprc_mean"].rank(ascending=False, method="min")
    ranked["brier_rank"] = ranked["brier_mean"].rank(ascending=True, method="min")
    ranked["mcc_rank"] = ranked["mcc_mean"].rank(ascending=False, method="min")
    ranked["f1_rank"] = ranked["f1_mean"].rank(ascending=False, method="min")
    ranked["calibration_rank"] = ranked["calibration_slope_error"].rank(ascending=True, method="min")
    ranked["stability_rank"] = ranked["auroc_std"].fillna(0).rank(ascending=True, method="min")

    # Interpretability bonus (negative = preferred)
    interpretability_map = {
        "logistic_l2": -0.75,
        "calibrated_gradient_boosting": -0.25,
        "gradient_boosting": 0.0,
        "random_forest": 0.1,
        "xgboost": 0.1,
        "lightgbm": 0.1,
        "catboost": 0.1,
        "svm_rbf": 0.2,
        "mlp_notebook_baseline": 0.5,
        "stacking_ensemble": 0.3,
    }
    ranked["interpretability_bonus"] = ranked["model"].map(interpretability_map).fillna(0.0)

    # Composite score (lower = better)
    ranked["selection_score"] = (
        ranked["auroc_rank"] * 2.0          # double weight on discrimination
        + ranked["auprc_rank"]
        + ranked["brier_rank"]
        + ranked["mcc_rank"]
        + ranked["f1_rank"]
        + ranked["calibration_rank"]
        + ranked["stability_rank"]
        + ranked["interpretability_bonus"]
    )
    ranked = ranked.sort_values(
        ["selection_score", "brier_mean", "auroc_mean"],
        ascending=[True, True, False],
    )
    return str(ranked.iloc[0]["model"]), ranked
