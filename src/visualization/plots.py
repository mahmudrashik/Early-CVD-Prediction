"""Publication-quality visualisations for IEEE-standard CVD prediction paper.

Every function saves a 300 DPI figure and returns silently.  Figures cover
data exploration, model comparison, calibration, explainability, and
statistical significance.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    roc_auc_score,
)
from sklearn.model_selection import learning_curve

# ── Style ────────────────────────────────────────────────────────────────────
_PALETTE = sns.color_palette("colorblind", 12)
sns.set_theme(style="whitegrid", font_scale=1.05, palette=_PALETTE)
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.family": "sans-serif"})


def _save(fig: plt.Figure, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# DATA EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

def missingness_heatmap(df: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False, ax=ax, cmap="YlOrRd")
    ax.set_title("Missingness Pattern"); ax.set_xlabel("Variables"); ax.set_ylabel("Records")
    _save(fig, path)

def class_distribution(df: pd.DataFrame, target: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette=_PALETTE[:2])
    for i, v in enumerate(counts.values): ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    ax.set_title("Binary Outcome Distribution"); ax.set_xlabel("Target"); ax.set_ylabel("Records")
    _save(fig, path)

def site_distribution(df: pd.DataFrame, group_col: str, target: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    table = df.groupby([group_col, target]).size().reset_index(name="n")
    sns.barplot(data=table, x=group_col, y="n", hue=target, ax=ax)
    ax.set_title("Centre Distribution by Outcome"); ax.set_xlabel("Centre"); ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=20); _save(fig, path)

def correlation_heatmap(df: pd.DataFrame, path: str | Path) -> None:
    """Feature correlation matrix (numeric columns only)."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Spearman Feature Correlation Matrix"); _save(fig, path)

def feature_distributions(df: pd.DataFrame, target: str, numeric_cols: list[str], path: str | Path) -> None:
    """Violin plots of numeric features split by target class."""
    n = len(numeric_cols); cols = min(3, n); rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4 * rows_n))
    axes = np.atleast_1d(axes).flatten()
    for i, col in enumerate(numeric_cols):
        sns.violinplot(data=df, x=target, y=col, ax=axes[i], palette=_PALETTE[:2], inner="quartile")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Feature Distributions by Class", fontsize=14, y=1.02); _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def model_comparison(summary: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    order = summary.sort_values("auroc_mean", ascending=False)["model"]
    bars = sns.barplot(data=summary, x="auroc_mean", y="model", order=order, ax=ax, palette=_PALETTE)
    if "auroc_std" in summary.columns:
        ordered = summary.set_index("model").loc[order]
        for i, (_, row) in enumerate(ordered.iterrows()):
            ax.errorbar(row["auroc_mean"], i, xerr=row["auroc_std"], color="black", capsize=3, linewidth=1)
    ax.set_xlim(0.5, 1.0); ax.set_title("Model Comparison — Mean AUROC ± SD")
    ax.set_xlabel("Mean AUROC"); ax.set_ylabel(""); _save(fig, path)

def model_comparison_multi_metric(summary: pd.DataFrame, path: str | Path) -> None:
    """Grouped bar chart comparing multiple metrics across models."""
    metrics = ["auroc_mean", "auprc_mean", "f1_mean", "mcc_mean", "balanced_accuracy_mean"]
    available = [m for m in metrics if m in summary.columns]
    melted = summary.melt(id_vars=["model"], value_vars=available, var_name="metric", value_name="score")
    melted["metric"] = melted["metric"].str.replace("_mean", "").str.upper()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="model", y="score", hue="metric", ax=ax, palette="Set2")
    ax.set_ylim(0, 1.05); ax.tick_params(axis="x", rotation=35)
    ax.set_title("Multi-Metric Model Comparison"); ax.set_ylabel("Score"); ax.set_xlabel("")
    ax.legend(loc="lower right", fontsize=8); _save(fig, path)

def radar_chart(summary: pd.DataFrame, path: str | Path) -> None:
    """Spider / radar chart comparing models across key metrics."""
    metrics = ["auroc_mean", "auprc_mean", "sensitivity_mean", "specificity_mean", "f1_mean", "mcc_mean"]
    available = [m for m in metrics if m in summary.columns]
    if len(available) < 3: return
    labels = [m.replace("_mean", "").replace("_", " ").title() for m in available]
    n_metrics = len(available)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(summary.iterrows()):
        values = [float(row[m]) for m in available]; values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=row["model"], color=_PALETTE[i % len(_PALETTE)])
        ax.fill(angles, values, alpha=0.08, color=_PALETTE[i % len(_PALETTE)])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_title("Model Performance Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7); _save(fig, path)

def boxplot_comparison(center_metrics: pd.DataFrame, metric: str, path: str | Path) -> None:
    """Box plot showing per-fold metric distributions across models."""
    if metric not in center_metrics.columns: return
    fig, ax = plt.subplots(figsize=(10, 5))
    order = center_metrics.groupby("model")[metric].median().sort_values(ascending=False).index
    sns.boxplot(data=center_metrics, x="model", y=metric, order=order, ax=ax, palette=_PALETTE)
    sns.stripplot(data=center_metrics, x="model", y=metric, order=order, ax=ax, color="black", alpha=0.4, size=4)
    ax.tick_params(axis="x", rotation=30); ax.set_title(f"Per-Centre {metric.upper()} Distribution")
    ax.set_xlabel(""); ax.set_ylabel(metric.upper()); _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# ROC / PR / CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def roc_pr_curves(predictions: pd.DataFrame, path_roc: str | Path, path_pr: str | Path) -> None:
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    for i, (model, group) in enumerate(predictions.groupby("model")):
        y, p = group["y_true"].astype(int), group["y_prob"].astype(float)
        if y.nunique() < 2: continue
        auc = roc_auc_score(y, p)
        RocCurveDisplay.from_predictions(y, p, ax=ax_roc, name=f"{model} ({auc:.3f})", color=_PALETTE[i % len(_PALETTE)])
        PrecisionRecallDisplay.from_predictions(y, p, ax=ax_pr, name=model, color=_PALETTE[i % len(_PALETTE)])
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax_roc.set_title("ROC Curves — Held-Out Centres"); _save(fig_roc, path_roc)
    ax_pr.set_title("Precision-Recall Curves"); _save(fig_pr, path_pr)

def calibration_plot(predictions: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Ideal")
    for i, (model, group) in enumerate(predictions.groupby("model")):
        y, p = group["y_true"].astype(int), group["y_prob"].astype(float)
        if y.nunique() < 2: continue
        prob_true, prob_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-", label=model, color=_PALETTE[i % len(_PALETTE)])
    ax.set_title("Calibration Curves"); ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate"); ax.legend(fontsize=7); _save(fig, path)

def confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Champion Confusion Matrix"); _save(fig, path)

def confusion_matrix_grid(y_true: np.ndarray, model_preds: dict[str, np.ndarray], path: str | Path) -> None:
    """Side-by-side confusion matrices for all models."""
    n = len(model_preds); cols = min(4, n); rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.5 * rows_n))
    axes = np.atleast_1d(axes).flatten()
    for i, (name, preds) in enumerate(sorted(model_preds.items())):
        ConfusionMatrixDisplay.from_predictions(y_true, preds, labels=[0, 1], ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(name, fontsize=9)
    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("Confusion Matrices — All Models", fontsize=13, y=1.02); _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def decision_curve_plot(dca: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dca["threshold"], dca["model_net_benefit"], label="Model", color=_PALETTE[0], lw=2)
    ax.plot(dca["threshold"], dca["treat_all_net_benefit"], label="Treat all", color=_PALETTE[1], ls="--")
    ax.plot(dca["threshold"], dca["treat_none_net_benefit"], label="Treat none", color="gray", ls=":")
    ax.set_title("Decision-Curve Analysis"); ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit"); ax.legend(); _save(fig, path)

def threshold_sensitivity_plot(threshold_df: pd.DataFrame, path: str | Path) -> None:
    """F1, Sensitivity, Specificity vs. threshold for champion model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, color in [("f1", _PALETTE[0]), ("sensitivity", _PALETTE[1]), ("specificity", _PALETTE[2])]:
        if metric in threshold_df.columns:
            ax.plot(threshold_df["threshold"], threshold_df[metric], "o-", label=metric.title(), color=color, lw=2)
    ax.set_title("Threshold Sensitivity Analysis"); ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score"); ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); _save(fig, path)

def predicted_probability_violin(y_true: np.ndarray, y_prob: np.ndarray, path: str | Path) -> None:
    """Violin plots of predicted probability distributions per class."""
    fig, ax = plt.subplots(figsize=(7, 5))
    df = pd.DataFrame({"Predicted Probability": y_prob, "True Class": y_true.astype(str)})
    sns.violinplot(data=df, x="True Class", y="Predicted Probability", ax=ax, palette=_PALETTE[:2], inner="quartile")
    ax.set_title("Predicted Probability Distribution by Class"); _save(fig, path)


# ══════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def importance_plot(importance: pd.DataFrame, path: str | Path) -> None:
    top = importance.sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title("Global Feature Importance"); ax.set_xlabel("Importance"); ax.set_ylabel(""); _save(fig, path)

def shap_summary_plot(model: object, X: pd.DataFrame, path: str | Path) -> bool:
    try:
        import shap
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        transformed = preprocessor.transform(X)
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.Explainer(classifier, transformed, feature_names=feature_names)
        values = explainer(transformed)
        shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=15)
        fig = plt.gcf(); _save(fig, path); return True
    except Exception:
        return False

def learning_curve_plot(model: object, X: pd.DataFrame, y: pd.Series, path: str | Path, cv: int = 5) -> None:
    """Training vs validation learning curves for the champion model."""
    try:
        from sklearn.base import clone
        fig, ax = plt.subplots(figsize=(8, 5))
        train_sizes, train_scores, val_scores = learning_curve(
            clone(model), X, y, cv=cv, scoring="roc_auc",
            train_sizes=np.linspace(0.2, 1.0, 8), n_jobs=-1, random_state=42,
        )
        ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training", color=_PALETTE[0])
        ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                        train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color=_PALETTE[0])
        ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation", color=_PALETTE[1])
        ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                        val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color=_PALETTE[1])
        ax.set_title("Learning Curves — Champion Model"); ax.set_xlabel("Training Set Size")
        ax.set_ylabel("AUROC"); ax.legend(); _save(fig, path)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════

def critical_difference_diagram(nemenyi_df: pd.DataFrame, path: str | Path) -> None:
    """Simplified CD diagram from Nemenyi post-hoc results."""
    if nemenyi_df.empty: return
    models_a = set(nemenyi_df["model_a"]); models_b = set(nemenyi_df["model_b"])
    all_models = sorted(models_a | models_b)
    ranks = {}
    for m in all_models:
        mask_a = nemenyi_df["model_a"] == m; mask_b = nemenyi_df["model_b"] == m
        if mask_a.any(): ranks[m] = nemenyi_df.loc[mask_a, "rank_a"].iloc[0]
        elif mask_b.any(): ranks[m] = nemenyi_df.loc[mask_b, "rank_b"].iloc[0]
    sorted_models = sorted(ranks, key=lambda x: ranks[x])
    cd = nemenyi_df["critical_difference"].iloc[0] if not nemenyi_df.empty else 0
    fig, ax = plt.subplots(figsize=(10, max(3, len(sorted_models) * 0.5)))
    for i, m in enumerate(sorted_models):
        ax.barh(i, ranks[m], color=_PALETTE[i % len(_PALETTE)], alpha=0.7, height=0.6)
        ax.text(ranks[m] + 0.05, i, f"{m} ({ranks[m]:.2f})", va="center", fontsize=9)
    ax.axvline(x=cd, color="red", ls="--", lw=1.5, label=f"CD = {cd:.2f}")
    ax.set_yticks(range(len(sorted_models))); ax.set_yticklabels([""] * len(sorted_models))
    ax.set_xlabel("Average Rank"); ax.set_title("Critical Difference Diagram")
    ax.legend(); ax.invert_yaxis(); _save(fig, path)

def delong_heatmap(delong_df: pd.DataFrame, path: str | Path) -> None:
    """Heatmap of DeLong p-values for pairwise AUROC comparisons."""
    if delong_df.empty: return
    all_models = sorted(set(delong_df["model_a"]) | set(delong_df["model_b"]))
    matrix = pd.DataFrame(1.0, index=all_models, columns=all_models)
    for _, row in delong_df.iterrows():
        matrix.loc[row["model_a"], row["model_b"]] = row["p_value"]
        matrix.loc[row["model_b"], row["model_a"]] = row["p_value"]
    np.fill_diagonal(matrix.values, np.nan)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
                center=0.05, vmin=0, vmax=0.2, ax=ax, square=True, linewidths=0.5)
    ax.set_title("DeLong Test p-values (AUROC Pairwise Comparison)"); _save(fig, path)
