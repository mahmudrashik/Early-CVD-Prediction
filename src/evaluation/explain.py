from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from domain.feature_dictionary import CLINICAL_DIRECTION_NOTES, FEATURE_COLUMNS


def global_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    classifier = model.named_steps.get("classifier")
    preprocessor = model.named_steps.get("preprocessor")
    if classifier is not None and preprocessor is not None and hasattr(classifier, "coef_"):
        names = preprocessor.get_feature_names_out()
        coefs = classifier.coef_[0]
        for name, coef in zip(names, coefs):
            original = _original_feature_name(name)
            rows.append({"feature": original, "transformed_feature": name, "importance": abs(float(coef)), "signed_effect": float(coef)})
        out = pd.DataFrame(rows)
        return out.groupby("feature", as_index=False).agg(importance=("importance", "sum"), signed_effect=("signed_effect", "sum")).sort_values("importance", ascending=False)

    result = permutation_importance(model, X, y, n_repeats=8, random_state=seed, scoring="roc_auc")
    for feature, mean, std in zip(FEATURE_COLUMNS, result.importances_mean, result.importances_std):
        rows.append({"feature": feature, "importance": float(mean), "importance_std": float(std), "signed_effect": np.nan})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def local_explanation(bundle: dict[str, object], X_row: pd.DataFrame, top_n: int = 6) -> list[dict[str, object]]:
    model = bundle["model"]
    metadata = bundle.get("metadata", {})
    importance = pd.DataFrame(metadata.get("global_importance", []))
    if importance.empty:
        importance = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": np.ones(len(FEATURE_COLUMNS))})

    classifier = model.named_steps.get("classifier")
    preprocessor = model.named_steps.get("preprocessor")
    if classifier is not None and preprocessor is not None and hasattr(classifier, "coef_"):
        transformed = preprocessor.transform(X_row)
        names = preprocessor.get_feature_names_out()
        contributions = transformed[0] * classifier.coef_[0]
        rows = []
        for name, value in zip(names, contributions):
            feature = _original_feature_name(name)
            rows.append({"feature": feature, "contribution": float(value)})
        contrib = pd.DataFrame(rows).groupby("feature", as_index=False)["contribution"].sum()
        contrib["magnitude"] = contrib["contribution"].abs()
        contrib = contrib.sort_values("magnitude", ascending=False).head(top_n)
        return [_explanation_record(row["feature"], X_row.iloc[0][row["feature"]], row["contribution"]) for _, row in contrib.iterrows()]

    top = importance.sort_values("importance", ascending=False).head(top_n)
    return [_explanation_record(row["feature"], X_row.iloc[0][row["feature"]], None) for _, row in top.iterrows()]


def _original_feature_name(transformed_name: str) -> str:
    cleaned = transformed_name.split("__", 1)[-1]
    for feature in FEATURE_COLUMNS:
        if cleaned == feature or cleaned.startswith(f"{feature}_"):
            return feature
    return cleaned


def _explanation_record(feature: str, value: object, contribution: float | None) -> dict[str, object]:
    if contribution is None or np.isnan(contribution):
        direction = "important for this model"
    elif contribution > 0:
        direction = "increased the estimated probability"
    else:
        direction = "decreased the estimated probability"
    return {
        "feature": feature,
        "value": None if pd.isna(value) else value,
        "direction": direction,
        "contribution": None if contribution is None or np.isnan(contribution) else round(float(contribution), 4),
        "clinical_note": CLINICAL_DIRECTION_NOTES.get(feature, "This feature contributed to the model estimate in the fitted benchmark model."),
    }

