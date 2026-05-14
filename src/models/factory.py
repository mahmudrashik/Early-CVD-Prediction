from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict[str, list[Any]]
    description: str


def _calibrated_gradient_boosting(seed: int) -> CalibratedClassifierCV:
    base = GradientBoostingClassifier(random_state=seed, n_estimators=80, learning_rate=0.05, max_depth=2)
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def candidate_models(seed: int) -> dict[str, ModelSpec]:
    return {
        "logistic_l2": ModelSpec(
            name="logistic_l2",
            estimator=LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=seed),
            param_grid={"classifier__C": [0.1, 1.0, 10.0]},
            description="Penalized logistic regression with balanced class weights.",
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=180,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
            param_grid={
                "classifier__max_depth": [4, None],
                "classifier__min_samples_leaf": [1, 5],
            },
            description="Random forest with class-balanced bootstrap trees.",
        ),
        "gradient_boosting": ModelSpec(
            name="gradient_boosting",
            estimator=GradientBoostingClassifier(random_state=seed),
            param_grid={
                "classifier__n_estimators": [60, 120],
                "classifier__learning_rate": [0.03, 0.08],
                "classifier__max_depth": [2],
            },
            description="Gradient boosting classifier.",
        ),
        "calibrated_gradient_boosting": ModelSpec(
            name="calibrated_gradient_boosting",
            estimator=_calibrated_gradient_boosting(seed),
            param_grid={},
            description="Gradient boosting with sigmoid probability calibration fitted only on development data.",
        ),
        "mlp_notebook_baseline": ModelSpec(
            name="mlp_notebook_baseline",
            estimator=MLPClassifier(
                hidden_layer_sizes=(72,),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size=256,
                learning_rate_init=0.001,
                max_iter=700,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20,
                random_state=seed,
            ),
            param_grid={
                "classifier__alpha": [0.0001, 0.01],
                "classifier__hidden_layer_sizes": [(72,), (72, 64, 24)],
            },
            description="Scikit-learn MLP comparator preserving the notebook feedforward neural-network role.",
        ),
    }


def make_pipeline(preprocessor: object, estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

