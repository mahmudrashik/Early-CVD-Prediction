import numpy as np

from evaluation.metrics import bootstrap_ci, classification_metrics


def test_classification_metrics_keys():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = classification_metrics(y, p, threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["brier"] < 0.2


def test_bootstrap_ci_returns_rows():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    ci = bootstrap_ci(y, p, threshold=0.5, iterations=10, seed=42)
    assert {"metric", "estimate", "ci_lower", "ci_upper"}.issubset(ci.columns)
    assert len(ci) > 0

