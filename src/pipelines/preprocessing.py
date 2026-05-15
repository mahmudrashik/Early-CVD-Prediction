"""Preprocessing pipeline with optional SMOTE and feature engineering.

This module constructs the scikit-learn / imbalanced-learn preprocessing
pipeline used by every model in the factory.  It supports:

* Median imputation + StandardScaler for numeric features.
* Mode imputation + OneHotEncoder for categorical features.
* Optional SMOTE oversampling (applied **inside** training folds only).
* Optional polynomial interaction features for clinically relevant pairs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from domain.feature_dictionary import CATEGORICAL_FEATURES, NUMERIC_FEATURES


# ── Helpers ──────────────────────────────────────────────────────────────────


def _one_hot_encoder() -> OneHotEncoder:
    """Return a OneHotEncoder compatible with both old and new sklearn APIs."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ── Feature Engineering ──────────────────────────────────────────────────────


# Clinically motivated interaction pairs (domain-knowledge driven)
INTERACTION_PAIRS: list[tuple[str, str]] = [
    ("age", "chol"),           # age × cholesterol — metabolic risk accumulation
    ("age", "trestbps"),       # age × resting BP — hypertensive ageing
    ("oldpeak", "thalach"),    # ST depression × max HR — exercise capacity
    ("age", "thalach"),        # age × max HR — age-adjusted HR reserve
    ("trestbps", "chol"),      # BP × cholesterol — composite vascular risk
]


class InteractionFeatureAdder(BaseEstimator, TransformerMixin):
    """Add clinically motivated interaction terms to numeric features.

    The transformer multiplies selected numeric feature pairs and appends
    the products as new columns.  Because it operates **after** imputation
    and scaling, the interactions represent standardised cross-terms.
    """

    def __init__(self, pairs: list[tuple[str, str]] | None = None) -> None:
        self.pairs = pairs or INTERACTION_PAIRS

    def fit(self, X: np.ndarray, y: Any = None) -> "InteractionFeatureAdder":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Numeric columns are ordered as NUMERIC_FEATURES
        name_to_idx = {name: i for i, name in enumerate(NUMERIC_FEATURES)}
        extra_cols: list[np.ndarray] = []
        for col_a, col_b in self.pairs:
            idx_a = name_to_idx.get(col_a)
            idx_b = name_to_idx.get(col_b)
            if idx_a is not None and idx_b is not None:
                extra_cols.append(
                    (X[:, idx_a] * X[:, idx_b]).reshape(-1, 1)
                )
        if extra_cols:
            return np.hstack([X, *extra_cols])
        return X

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        base = list(input_features) if input_features is not None else list(NUMERIC_FEATURES)
        name_to_idx = {name: i for i, name in enumerate(NUMERIC_FEATURES)}
        for col_a, col_b in self.pairs:
            if col_a in name_to_idx and col_b in name_to_idx:
                base.append(f"{col_a}_x_{col_b}")
        return base


# ── Pipeline Builders ────────────────────────────────────────────────────────


def _numeric_pipeline(feature_engineering: bool = False) -> Pipeline:
    """Standard numeric pipeline: impute → scale (→ optionally add interactions)."""
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if feature_engineering:
        steps.append(("interactions", InteractionFeatureAdder()))
    return Pipeline(steps=steps)


def _categorical_pipeline() -> Pipeline:
    """Standard categorical pipeline: impute → one-hot encode."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )


def build_preprocessor(feature_engineering: bool = False) -> ColumnTransformer:
    """Construct the full column transformer.

    Parameters
    ----------
    feature_engineering : bool
        If *True*, clinically motivated interaction features are appended
        to the numeric branch after scaling.
    """
    return ColumnTransformer(
        transformers=[
            ("num", _numeric_pipeline(feature_engineering), NUMERIC_FEATURES),
            ("cat", _categorical_pipeline(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
