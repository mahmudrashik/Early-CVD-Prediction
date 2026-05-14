from __future__ import annotations

from dataclasses import dataclass


FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

TARGET_BINARY = "target_binary"
TARGET_MULTICLASS = "target_multiclass"
GROUP_COLUMN = "center"


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    label: str
    kind: str
    description: str
    clinical_note: str


FEATURE_DICTIONARY: dict[str, FeatureDefinition] = {
    "age": FeatureDefinition("age", "Age", "numeric", "Age in years.", "Older age is generally associated with higher cardiovascular risk."),
    "sex": FeatureDefinition("sex", "Sex", "categorical", "Recorded biological sex in the source dataset.", "Sex can reflect baseline risk differences and historical referral patterns."),
    "cp": FeatureDefinition("cp", "Chest pain type", "categorical", "Chest pain category.", "Asymptomatic or atypical presentations may carry different risk patterns in these benchmark datasets."),
    "trestbps": FeatureDefinition("trestbps", "Resting blood pressure", "numeric", "Resting systolic blood pressure in mm Hg.", "Higher resting blood pressure is a recognized cardiovascular risk marker."),
    "chol": FeatureDefinition("chol", "Serum cholesterol", "numeric", "Serum cholesterol in mg/dL.", "Elevated cholesterol may contribute to cardiovascular risk, although this dataset has measurement and provenance limitations."),
    "fbs": FeatureDefinition("fbs", "Fasting blood sugar > 120 mg/dL", "categorical", "Boolean fasting blood sugar indicator.", "Hyperglycemia can co-occur with metabolic cardiovascular risk."),
    "restecg": FeatureDefinition("restecg", "Resting ECG", "categorical", "Resting electrocardiographic result.", "Abnormal ECG findings may influence predicted risk."),
    "thalach": FeatureDefinition("thalach", "Maximum heart rate achieved", "numeric", "Maximum heart rate achieved.", "Lower achieved heart rate can indicate reduced exercise tolerance in this dataset."),
    "exang": FeatureDefinition("exang", "Exercise-induced angina", "categorical", "Exercise-induced angina indicator.", "Exercise-induced angina is often a strong risk signal in these data."),
    "oldpeak": FeatureDefinition("oldpeak", "ST depression", "numeric", "ST depression induced by exercise relative to rest.", "Higher ST depression can indicate ischemic changes."),
    "slope": FeatureDefinition("slope", "ST segment slope", "categorical", "Slope of the peak exercise ST segment.", "Flat or downsloping ST segments may be associated with higher risk."),
    "ca": FeatureDefinition("ca", "Major vessels", "numeric", "Number of major vessels colored by fluoroscopy.", "More affected vessels can be associated with disease, but missingness is substantial outside Cleveland."),
    "thal": FeatureDefinition("thal", "Thalassemia / stress test result", "categorical", "Thalassemia/stress-test result coding in the source data.", "Reversible defect often contributes strongly in Cleveland-style datasets."),
}


ALLOWED_CATEGORIES = {
    "sex": ["Female", "Male"],
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "fbs": ["false", "true"],
    "restecg": ["normal", "st-t abnormality", "lv hypertrophy"],
    "exang": ["false", "true"],
    "slope": ["upsloping", "flat", "downsloping"],
    "thal": ["normal", "fixed defect", "reversable defect", "unknown"],
}


CLINICAL_DIRECTION_NOTES = {
    "age": "Age is part of the model because cardiovascular disease prevalence rises with age in the source cohorts.",
    "cp": "Chest pain category is influential because symptom presentation differs between patients with and without recorded disease.",
    "thalach": "Lower maximum heart rate achieved can increase model-estimated risk in this benchmark context.",
    "exang": "Exercise-induced angina commonly increases model-estimated risk.",
    "oldpeak": "Higher exercise-induced ST depression commonly increases model-estimated risk.",
    "slope": "Flat or downsloping ST segment patterns often increase model-estimated risk.",
    "ca": "A larger number of major vessels can increase model-estimated risk, but this variable has important missingness limitations.",
    "thal": "Reversible or fixed defect categories can strongly influence predictions in Cleveland-style data.",
}

