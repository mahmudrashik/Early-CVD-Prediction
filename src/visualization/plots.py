from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def _save(fig: plt.Figure, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def missingness_heatmap(df: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False, ax=ax)
    ax.set_title("Missingness Pattern")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Records")
    _save(fig, path)


def class_distribution(df: pd.DataFrame, target: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#4c78a8")
    ax.set_title("Binary Outcome Distribution")
    ax.set_xlabel("Target: 0 absent, 1 present")
    ax.set_ylabel("Records")
    _save(fig, path)


def site_distribution(df: pd.DataFrame, group_col: str, target: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    table = df.groupby([group_col, target]).size().reset_index(name="n")
    sns.barplot(data=table, x=group_col, y="n", hue=target, ax=ax)
    ax.set_title("Center Distribution by Binary Outcome")
    ax.set_xlabel("Center")
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def model_comparison(summary: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    order = summary.sort_values("auroc_mean", ascending=False)["model"]
    sns.barplot(data=summary, x="auroc_mean", y="model", order=order, ax=ax, color="#4c78a8")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Site-Aware Model Comparison by Mean AUROC")
    ax.set_xlabel("Mean AUROC Across Held-Out Centers")
    ax.set_ylabel("")
    _save(fig, path)


def roc_pr_curves(predictions: pd.DataFrame, path_roc: str | Path, path_pr: str | Path) -> None:
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    for model, group in predictions.groupby("model"):
        y = group["y_true"].astype(int)
        p = group["y_prob"].astype(float)
        if y.nunique() < 2:
            continue
        RocCurveDisplay.from_predictions(y, p, ax=ax_roc, name=model)
        PrecisionRecallDisplay.from_predictions(y, p, ax=ax_pr, name=model)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_title("ROC Curves from Held-Out Centers")
    ax_pr.set_title("Precision-Recall Curves from Held-Out Centers")
    _save(fig_roc, path_roc)
    _save(fig_pr, path_pr)


def calibration_plot(predictions: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Ideal")
    for model, group in predictions.groupby("model"):
        y = group["y_true"].astype(int)
        p = group["y_prob"].astype(float)
        if y.nunique() < 2:
            continue
        prob_true, prob_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=model)
    ax.set_title("Calibration Curves from Held-Out Centers")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.legend(fontsize=7)
    _save(fig, path)


def confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix at Selected Threshold")
    _save(fig, path)


def decision_curve_plot(dca: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dca["threshold"], dca["model_net_benefit"], label="Model", color="#4c78a8")
    ax.plot(dca["threshold"], dca["treat_all_net_benefit"], label="Treat all", color="#f58518")
    ax.plot(dca["threshold"], dca["treat_none_net_benefit"], label="Treat none", color="gray")
    ax.set_title("Decision-Curve Analysis")
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.legend()
    _save(fig, path)


def importance_plot(importance: pd.DataFrame, path: str | Path) -> None:
    top = importance.sort_values("importance", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=top, x="importance", y="feature", color="#4c78a8", ax=ax)
    ax.set_title("Global Model Explanation Summary")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    _save(fig, path)


def shap_summary_plot(model: object, X: pd.DataFrame, path: str | Path) -> bool:
    try:
        import shap

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        transformed = preprocessor.transform(X)
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.Explainer(classifier, transformed, feature_names=feature_names)
        values = explainer(transformed)
        shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=12)
        fig = plt.gcf()
        _save(fig, path)
        return True
    except Exception:
        return False
