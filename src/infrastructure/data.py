from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from domain.feature_dictionary import FEATURE_COLUMNS, GROUP_COLUMN, TARGET_BINARY, TARGET_MULTICLASS


CP_FROM_NUMERIC = {
    0: "asymptomatic",
    1: "non-anginal",
    2: "atypical angina",
    3: "typical angina",
}
RESTECG_FROM_NUMERIC = {
    0: "lv hypertrophy",
    1: "normal",
    2: "st-t abnormality",
}
SLOPE_FROM_NUMERIC = {
    0: "downsloping",
    1: "flat",
    2: "upsloping",
}
THAL_FROM_NUMERIC = {
    0: "unknown",
    1: "fixed defect",
    2: "normal",
    3: "reversable defect",
}
SEX_FROM_NUMERIC = {0: "Female", 1: "Male"}
BOOL_FROM_NUMERIC = {0: "false", 1: "true"}


def _bool_to_label(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return pd.NA


def _map_int_category(series: pd.Series, mapping: dict[int, str]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(lambda x: mapping.get(int(x), "unknown") if pd.notna(x) else pd.NA)


def load_primary_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"dataset": GROUP_COLUMN, "thalch": "thalach"})
    df[TARGET_MULTICLASS] = pd.to_numeric(df["num"], errors="coerce").astype("Int64")
    df[TARGET_BINARY] = (df[TARGET_MULTICLASS] > 0).astype(int)
    df["fbs"] = df["fbs"].map(_bool_to_label)
    df["exang"] = df["exang"].map(_bool_to_label)
    df["sex"] = df["sex"].astype(object)
    for col in ["cp", "restecg", "slope", "thal", GROUP_COLUMN, "fbs", "exang"]:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["id", GROUP_COLUMN, *FEATURE_COLUMNS, TARGET_BINARY, TARGET_MULTICLASS]].copy()


def load_secondary_dataset(path: str | Path, deduplicate: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    duplicate_count = int(df.duplicated().sum())
    if deduplicate:
        df = df.drop_duplicates().reset_index(drop=True)
    df = df.rename(columns={"target": TARGET_BINARY})
    df[GROUP_COLUMN] = "heart_csv"
    df[TARGET_MULTICLASS] = pd.NA
    df["sex"] = _map_int_category(df["sex"], SEX_FROM_NUMERIC)
    df["cp"] = _map_int_category(df["cp"], CP_FROM_NUMERIC)
    df["fbs"] = _map_int_category(df["fbs"], BOOL_FROM_NUMERIC)
    df["restecg"] = _map_int_category(df["restecg"], RESTECG_FROM_NUMERIC)
    df["exang"] = _map_int_category(df["exang"], BOOL_FROM_NUMERIC)
    df["slope"] = _map_int_category(df["slope"], SLOPE_FROM_NUMERIC)
    df["thal"] = _map_int_category(df["thal"], THAL_FROM_NUMERIC)
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_BINARY] = pd.to_numeric(df[TARGET_BINARY], errors="coerce").astype(int)
    out = df[[GROUP_COLUMN, *FEATURE_COLUMNS, TARGET_BINARY, TARGET_MULTICLASS]].copy()
    out.attrs["exact_duplicate_rows_removed"] = duplicate_count if deduplicate else 0
    out.attrs["original_rows"] = len(pd.read_csv(path))
    return out


def feature_target_group(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_BINARY].astype(int).copy()
    groups = df[GROUP_COLUMN].astype(str).copy()
    return X, y, groups
