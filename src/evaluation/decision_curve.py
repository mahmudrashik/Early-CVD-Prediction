from __future__ import annotations

import numpy as np
import pandas as pd


def decision_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    prevalence = y_true.mean() if n else 0.0
    rows = []
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            continue
        pred = y_prob >= pt
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        net_benefit = (tp / n) - (fp / n) * (pt / (1 - pt))
        treat_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        rows.append({"threshold": pt, "model_net_benefit": net_benefit, "treat_all_net_benefit": treat_all, "treat_none_net_benefit": 0.0})
    return pd.DataFrame(rows)

