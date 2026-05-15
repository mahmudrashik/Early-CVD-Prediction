"""Model factory — all candidate classifiers and pipeline constructors.

Provides ten candidate classifiers spanning interpretable baselines,
modern gradient-boosting frameworks, and a heterogeneous stacking
ensemble.  Every model is wrapped in a pipeline that includes
preprocessing (imputation + scaling + optional feature engineering)
and optional SMOTE oversampling **inside** the training fold.

Design decisions
----------------
* XGBoost / LightGBM / CatBoost are included because they consistently
  achieve top-tier AUROC (>0.92) on tabular clinical data (IEEE 2023–24).
* The stacking ensemble uses diverse Level-0 learners combined through a
  calibrated Logistic Regression meta-learner — the gold-standard
  approach in published heart-disease prediction research.
* SMOTE is injected via ``imblearn.pipeline.Pipeline`` so it is applied
  only to training data; test-fold samples are never synthesised.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# ── Guard imports for optional heavy dependencies ────────────────────────────
try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    _HAS_LGB = True
except ImportError:  # pragma: no cover
    _HAS_LGB = False

try:
    from catboost import CatBoostClassifier

    _HAS_CAT = True
except ImportError:  # pragma: no cover
    _HAS_CAT = False

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    _HAS_SMOTE = True
except ImportError:  # pragma: no cover
    _HAS_SMOTE = False

from sklearn.pipeline import Pipeline


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single candidate model."""

    name: str
    estimator: object
    param_grid: dict[str, list[Any]]
    description: str
    search_budget: int = 30  # max iterations for RandomizedSearchCV


# ── Individual model builders ────────────────────────────────────────────────


def _calibrated_gradient_boosting(seed: int) -> CalibratedClassifierCV:
    base = GradientBoostingClassifier(
        random_state=seed,
        n_estimators=80,
        learning_rate=0.05,
        max_depth=2,
    )
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def _xgboost(seed: int) -> "XGBClassifier":
    return XGBClassifier(
        random_state=seed,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0,
    )


def _lightgbm(seed: int) -> "LGBMClassifier":
    return LGBMClassifier(
        random_state=seed,
        is_unbalance=True,
        verbose=-1,
        n_jobs=-1,
    )


def _catboost(seed: int) -> "CatBoostClassifier":
    return CatBoostClassifier(
        random_seed=seed,
        auto_class_weights="Balanced",
        verbose=0,
        allow_writing_files=False,
    )


def _svm_rbf(seed: int) -> CalibratedClassifierCV:
    base = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=seed,
    )
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def _stacking_ensemble(seed: int) -> StackingClassifier:
    """Heterogeneous stacking: diverse Level-0 + LR meta-learner."""
    estimators: list[tuple[str, Any]] = [
        ("rf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.05, max_depth=3,
            random_state=seed,
        )),
        ("lr", LogisticRegression(
            max_iter=3000, class_weight="balanced",
            solver="liblinear", random_state=seed, C=1.0,
        )),
    ]
    if _HAS_XGB:
        estimators.append((
            "xgb",
            XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                eval_metric="logloss", use_label_encoder=False,
                random_state=seed, n_jobs=-1, verbosity=0,
                scale_pos_weight=1.0, subsample=0.8,
                colsample_bytree=0.8,
            ),
        ))
    if _HAS_LGB:
        estimators.append((
            "lgb",
            LGBMClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=-1,
                num_leaves=31, is_unbalance=True, verbose=-1,
                random_state=seed, n_jobs=-1,
            ),
        ))
    if _HAS_CAT:
        estimators.append((
            "cat",
            CatBoostClassifier(
                iterations=200, learning_rate=0.05, depth=6,
                auto_class_weights="Balanced", verbose=0,
                random_seed=seed, allow_writing_files=False,
            ),
        ))

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            max_iter=3000, solver="lbfgs", random_state=seed,
        ),
        cv=3,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )


# ── Master catalogue ─────────────────────────────────────────────────────────


def candidate_models(seed: int) -> dict[str, ModelSpec]:
    """Return all candidate model specifications keyed by name."""
    models: dict[str, ModelSpec] = {
        # ── Interpretable baselines ──────────────────────────────────────
        "logistic_l2": ModelSpec(
            name="logistic_l2",
            estimator=LogisticRegression(
                max_iter=3000, class_weight="balanced",
                solver="liblinear", random_state=seed,
            ),
            param_grid={"classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
            description="Penalized logistic regression with balanced class weights.",
            search_budget=6,
        ),

        # ── Ensemble baselines ───────────────────────────────────────────
        "random_forest": ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
            param_grid={
                "classifier__max_depth": [4, 6, 8, None],
                "classifier__min_samples_leaf": [1, 3, 5],
                "classifier__min_samples_split": [2, 5],
                "classifier__max_features": ["sqrt", "log2"],
            },
            description="Random forest with class-balanced bootstrap trees.",
            search_budget=40,
        ),
        "gradient_boosting": ModelSpec(
            name="gradient_boosting",
            estimator=GradientBoostingClassifier(random_state=seed),
            param_grid={
                "classifier__n_estimators": [60, 120, 200],
                "classifier__learning_rate": [0.01, 0.03, 0.08, 0.1],
                "classifier__max_depth": [2, 3, 4],
                "classifier__subsample": [0.8, 1.0],
                "classifier__min_samples_leaf": [1, 5],
            },
            description="Gradient boosting classifier.",
            search_budget=50,
        ),
        "calibrated_gradient_boosting": ModelSpec(
            name="calibrated_gradient_boosting",
            estimator=_calibrated_gradient_boosting(seed),
            param_grid={},
            description="Gradient boosting with sigmoid probability calibration.",
            search_budget=1,
        ),
        "mlp_notebook_baseline": ModelSpec(
            name="mlp_notebook_baseline",
            estimator=MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size=64,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=seed,
            ),
            param_grid={
                "classifier__alpha": [0.0001, 0.001, 0.01],
                "classifier__hidden_layer_sizes": [
                    (128, 64),
                    (128, 64, 32),
                    (256, 128, 64),
                ],
                "classifier__learning_rate_init": [0.0005, 0.001],
            },
            description="Scikit-learn MLP comparator with deeper architecture.",
            search_budget=18,
        ),
    }

    # ── Modern gradient boosting ─────────────────────────────────────────
    if _HAS_XGB:
        models["xgboost"] = ModelSpec(
            name="xgboost",
            estimator=_xgboost(seed),
            param_grid={
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [3, 4, 5, 6],
                "classifier__subsample": [0.7, 0.8, 1.0],
                "classifier__colsample_bytree": [0.7, 0.8, 1.0],
                "classifier__min_child_weight": [1, 3, 5],
                "classifier__gamma": [0, 0.1, 0.2],
                "classifier__reg_alpha": [0, 0.01, 0.1],
                "classifier__reg_lambda": [0.5, 1.0, 1.5],
                "classifier__scale_pos_weight": [1.0, 1.5, 2.0],
            },
            description="XGBoost gradient-boosted trees with extensive regularisation.",
            search_budget=60,
        )

    if _HAS_LGB:
        models["lightgbm"] = ModelSpec(
            name="lightgbm",
            estimator=_lightgbm(seed),
            param_grid={
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [-1, 4, 6, 8],
                "classifier__num_leaves": [15, 31, 63],
                "classifier__min_child_samples": [5, 10, 20],
                "classifier__subsample": [0.7, 0.8, 1.0],
                "classifier__colsample_bytree": [0.7, 0.8, 1.0],
                "classifier__reg_alpha": [0, 0.01, 0.1],
                "classifier__reg_lambda": [0, 0.1, 1.0],
            },
            description="LightGBM histogram-based gradient boosting.",
            search_budget=60,
        )

    if _HAS_CAT:
        models["catboost"] = ModelSpec(
            name="catboost",
            estimator=_catboost(seed),
            param_grid={
                "classifier__iterations": [100, 200, 300, 500],
                "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "classifier__depth": [4, 6, 8],
                "classifier__l2_leaf_reg": [1, 3, 5, 7],
                "classifier__border_count": [32, 64, 128],
            },
            description="CatBoost with ordered boosting and balanced class weights.",
            search_budget=60,
        )

    # ── Kernel-based ─────────────────────────────────────────────────────
    models["svm_rbf"] = ModelSpec(
        name="svm_rbf",
        estimator=_svm_rbf(seed),
        param_grid={},  # tuning happens inside CalibratedClassifierCV
        description="Radial-basis-function SVM with probability calibration.",
        search_budget=1,
    )

    # ── Stacking ensemble ────────────────────────────────────────────────
    models["stacking_ensemble"] = ModelSpec(
        name="stacking_ensemble",
        estimator=_stacking_ensemble(seed),
        param_grid={},  # base learners are pre-configured
        description="Heterogeneous stacking: RF + GB + LR + XGB + LGB + CAT → LR meta-learner.",
        search_budget=1,
    )

    return models


# ── Pipeline constructors ────────────────────────────────────────────────────


def make_pipeline(
    preprocessor: object,
    estimator: object,
    use_smote: bool = False,
    smote_k: int = 3,
    seed: int = 42,
) -> Pipeline:
    """Build a sklearn/imblearn pipeline with optional SMOTE.

    When *use_smote* is True and ``imbalanced-learn`` is available, SMOTE
    is inserted **between** the preprocessor and the classifier so that
    synthetic samples are generated only from training data.  The
    ``imblearn.pipeline.Pipeline`` ensures SMOTE is skipped during
    ``predict`` / ``predict_proba``.
    """
    if use_smote and _HAS_SMOTE:
        return ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(k_neighbors=smote_k, random_state=seed)),
                ("classifier", estimator),
            ]
        )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
