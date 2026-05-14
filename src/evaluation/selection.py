from __future__ import annotations

import pandas as pd


def select_champion(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Select a champion with calibration and stability considered alongside discrimination."""
    ranked = summary.copy()
    ranked["calibration_slope_error"] = (ranked["calibration_slope_mean"] - 1.0).abs()
    ranked["auroc_rank"] = ranked["auroc_mean"].rank(ascending=False, method="min")
    ranked["auprc_rank"] = ranked["auprc_mean"].rank(ascending=False, method="min")
    ranked["brier_rank"] = ranked["brier_mean"].rank(ascending=True, method="min")
    ranked["mcc_rank"] = ranked["mcc_mean"].rank(ascending=False, method="min")
    ranked["calibration_rank"] = ranked["calibration_slope_error"].rank(ascending=True, method="min")
    ranked["stability_rank"] = ranked["auroc_std"].fillna(0).rank(ascending=True, method="min")
    ranked["interpretability_bonus"] = ranked["model"].map(
        {
            "logistic_l2": -0.75,
            "calibrated_gradient_boosting": -0.25,
            "gradient_boosting": 0.0,
            "random_forest": 0.1,
            "mlp_notebook_baseline": 0.5,
        }
    ).fillna(0.0)
    ranked["selection_score"] = (
        ranked["auroc_rank"]
        + ranked["auprc_rank"]
        + ranked["brier_rank"]
        + ranked["mcc_rank"]
        + ranked["calibration_rank"]
        + ranked["stability_rank"]
        + ranked["interpretability_bonus"]
    )
    ranked = ranked.sort_values(["selection_score", "brier_mean", "auroc_mean"], ascending=[True, True, False])
    return str(ranked.iloc[0]["model"]), ranked

