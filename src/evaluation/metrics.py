from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(fn, *args, default: float = float("nan"), **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logit, y_true)
        return float(model.intercept_[0]), float(model.coef_[0][0])
    except Exception:
        return float("nan"), float("nan")


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    npv = tn / (tn + fn) if (tn + fn) else math.nan
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    return {
        "auroc": _safe_metric(roc_auc_score, y_true, y_prob),
        "auprc": _safe_metric(average_precision_score, y_true, y_prob),
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "npv": float(npv),
        "f1": _safe_metric(f1_score, y_true, y_pred, zero_division=0),
        "mcc": _safe_metric(matthews_corrcoef, y_true, y_pred),
        "brier": _safe_metric(brier_score_loss, y_true, y_prob),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "threshold": float(threshold),
        "n": float(len(y_true)),
        "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
    }


def threshold_table(y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        row = classification_metrics(y_true, y_prob, threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
    metrics: tuple[str, ...] = ("auroc", "auprc", "balanced_accuracy", "brier", "sensitivity", "specificity"),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    values: dict[str, list[float]] = {m: [] for m in metrics}
    n = len(y_true)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        row = classification_metrics(y_true[idx], y_prob[idx], threshold)
        for metric in metrics:
            values[metric].append(row[metric])
    rows = []
    point = classification_metrics(y_true, y_prob, threshold)
    for metric, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "metric": metric,
                "estimate": point[metric],
                "ci_lower": float(np.nanpercentile(arr, 2.5)) if len(arr) else float("nan"),
                "ci_upper": float(np.nanpercentile(arr, 97.5)) if len(arr) else float("nan"),
                "bootstrap_iterations": int(len(arr)),
            }
        )
    return pd.DataFrame(rows)

