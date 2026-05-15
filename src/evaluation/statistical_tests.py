"""Statistical significance tests for model comparison.

Implements the three standard tests expected in IEEE-level ML papers:

1. **DeLong test** — pairwise comparison of two correlated AUROCs.
2. **Friedman test** — non-parametric test for differences among
   multiple classifiers across multiple folds/centres.
3. **Nemenyi post-hoc test** — pairwise post-hoc after Friedman.
4. **McNemar's test** — pairwise comparison of classification decisions.

References
----------
* DeLong et al., *Biometrics* 44(3), 1988.
* Demšar, JMLR 7, 2006 (Friedman + Nemenyi with CD diagrams).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# 1. DeLong Test
# ══════════════════════════════════════════════════════════════════════════════


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute mid-ranks for tied observations."""
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j + 1)
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _fast_delong(y_true: np.ndarray, y_score_a: np.ndarray, y_score_b: np.ndarray) -> tuple[float, float]:
    """Compute DeLong's test statistic and p-value for two AUROCs.

    Returns
    -------
    z_stat : float
        Signed Z-statistic.
    p_value : float
        Two-sided p-value.
    """
    y_true = np.asarray(y_true, dtype=int)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")

    # Structural components
    scores = np.column_stack([y_score_a, y_score_b])  # (N, 2)
    aucs = np.zeros(2)
    v_10 = np.zeros((2, m))
    v_01 = np.zeros((2, n))

    for k in range(2):
        s = scores[:, k]
        for i, pi in enumerate(pos):
            v_10[k, i] = np.mean(s[pi] > s[neg]) + 0.5 * np.mean(s[pi] == s[neg])
        for j, nj in enumerate(neg):
            v_01[k, j] = np.mean(s[pos] > s[nj]) + 0.5 * np.mean(s[pos] == s[nj])
        aucs[k] = np.mean(v_10[k])

    # Covariance matrix
    s10 = np.cov(v_10) if m > 1 else np.zeros((2, 2))
    s01 = np.cov(v_01) if n > 1 else np.zeros((2, 2))
    S = s10 / m + s01 / n

    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return float("nan"), float("nan")
    z = diff / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(z), float(p)


def delong_pairwise(
    y_true: np.ndarray,
    model_probs: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Run DeLong tests for every pair of models.

    Parameters
    ----------
    y_true : array of shape (n_samples,)
    model_probs : dict mapping model name → predicted probabilities

    Returns
    -------
    DataFrame with columns: model_a, model_b, auc_a, auc_b, z_stat, p_value, significant_005
    """
    from sklearn.metrics import roc_auc_score

    names = sorted(model_probs.keys())
    rows: list[dict[str, Any]] = []
    for a, b in itertools.combinations(names, 2):
        pa, pb = model_probs[a], model_probs[b]
        z, p = _fast_delong(y_true, pa, pb)
        rows.append({
            "model_a": a,
            "model_b": b,
            "auc_a": roc_auc_score(y_true, pa) if len(np.unique(y_true)) > 1 else float("nan"),
            "auc_b": roc_auc_score(y_true, pb) if len(np.unique(y_true)) > 1 else float("nan"),
            "z_stat": z,
            "p_value": p,
            "significant_005": p < 0.05 if not np.isnan(p) else False,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Friedman Test + Nemenyi Post-Hoc
# ══════════════════════════════════════════════════════════════════════════════


def friedman_test(metric_matrix: pd.DataFrame) -> dict[str, float]:
    """Non-parametric Friedman test across models.

    Parameters
    ----------
    metric_matrix : DataFrame
        Rows = folds/centres, columns = model names, values = metric.

    Returns
    -------
    dict with keys: chi2, p_value, df, significant_005
    """
    k = metric_matrix.shape[1]  # number of models
    n = metric_matrix.shape[0]  # number of folds / centres
    if k < 3 or n < 2:
        return {"chi2": float("nan"), "p_value": float("nan"), "df": k - 1, "significant_005": False}

    stat, p = stats.friedmanchisquare(
        *[metric_matrix.iloc[:, i].values for i in range(k)]
    )
    return {"chi2": float(stat), "p_value": float(p), "df": k - 1, "significant_005": p < 0.05}


def nemenyi_posthoc(metric_matrix: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Nemenyi post-hoc pairwise comparison.

    Uses the critical difference approach from Demšar (JMLR 2006).

    Returns
    -------
    DataFrame with columns: model_a, model_b, rank_a, rank_b, rank_diff, cd, significant
    """
    k = metric_matrix.shape[1]
    n = metric_matrix.shape[0]

    # Compute average ranks (higher metric = better = lower rank number)
    ranks = metric_matrix.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0)

    # Critical difference (Nemenyi, two-tailed)
    q_alpha = _nemenyi_q(k, alpha)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    names = metric_matrix.columns.tolist()
    rows: list[dict[str, Any]] = []
    for a, b in itertools.combinations(names, 2):
        diff = abs(avg_ranks[a] - avg_ranks[b])
        rows.append({
            "model_a": a,
            "model_b": b,
            "rank_a": avg_ranks[a],
            "rank_b": avg_ranks[b],
            "rank_diff": diff,
            "critical_difference": cd,
            "significant": diff > cd,
        })
    return pd.DataFrame(rows)


def _nemenyi_q(k: int, alpha: float = 0.05) -> float:
    """Approximate critical value for Nemenyi test.

    Uses the Studentised range distribution quantile.
    For common k values at α = 0.05.
    """
    # Table of q_α values for k groups at α = 0.05  (Demšar, 2006)
    table_005 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
    }
    if alpha == 0.05 and k in table_005:
        return table_005[k]
    # Fallback: approximate via normal quantile (less precise)
    return stats.norm.ppf(1 - alpha / (k * (k - 1)))


# ══════════════════════════════════════════════════════════════════════════════
# 3. McNemar's Test
# ══════════════════════════════════════════════════════════════════════════════


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict[str, float]:
    """McNemar's test comparing two classifiers' binary decisions.

    Returns
    -------
    dict with keys: chi2, p_value, b, c  (b = A-wrong/B-right, c = A-right/B-wrong)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    b = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    c = int(np.sum(correct_a & ~correct_b))   # A right, B wrong

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c, "significant_005": False}

    # Continuity-corrected McNemar
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.chi2.sf(chi2, 1))
    return {"chi2": float(chi2), "p_value": p, "b": b, "c": c, "significant_005": p < 0.05}


def mcnemar_pairwise(
    y_true: np.ndarray,
    model_preds: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Run McNemar test for every pair of models."""
    names = sorted(model_preds.keys())
    rows: list[dict[str, Any]] = []
    for a, b in itertools.combinations(names, 2):
        result = mcnemar_test(y_true, model_preds[a], model_preds[b])
        result["model_a"] = a
        result["model_b"] = b
        rows.append(result)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Orchestrator — run all tests
# ══════════════════════════════════════════════════════════════════════════════


def run_all_statistical_tests(
    y_true: np.ndarray,
    model_probs: dict[str, np.ndarray],
    fold_metrics: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, pd.DataFrame | dict]:
    """Run the full suite of statistical significance tests.

    Parameters
    ----------
    y_true : array
        Ground-truth labels (pooled across folds).
    model_probs : dict
        Model name → predicted probabilities (pooled).
    fold_metrics : DataFrame
        Columns = model names, rows = fold/centre, values = AUROC.
    threshold : float
        Decision threshold for McNemar test.

    Returns
    -------
    dict with keys: delong, friedman, nemenyi, mcnemar
    """
    # DeLong pairwise
    delong = delong_pairwise(y_true, model_probs)

    # Friedman
    friedman = friedman_test(fold_metrics)

    # Nemenyi (only if Friedman is significant)
    nemenyi = nemenyi_posthoc(fold_metrics) if friedman.get("significant_005", False) else pd.DataFrame()

    # McNemar pairwise
    model_preds = {
        name: (probs >= threshold).astype(int)
        for name, probs in model_probs.items()
    }
    mcnemar = mcnemar_pairwise(y_true, model_preds)

    return {
        "delong": delong,
        "friedman": friedman,
        "nemenyi": nemenyi,
        "mcnemar": mcnemar,
    }
