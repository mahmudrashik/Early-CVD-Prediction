from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: float = Field(..., ge=18, le=100)
    sex: Literal["Female", "Male"]
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    trestbps: float = Field(..., ge=60, le=260)
    chol: float = Field(..., ge=80, le=700)
    fbs: Literal["false", "true"]
    restecg: Literal["normal", "st-t abnormality", "lv hypertrophy"]
    thalach: float = Field(..., ge=40, le=230)
    exang: Literal["false", "true"]
    oldpeak: float = Field(..., ge=-5, le=10)
    slope: Literal["upsloping", "flat", "downsloping"]
    ca: float = Field(..., ge=0, le=4)
    thal: Literal["normal", "fixed defect", "reversable defect", "unknown"]


class ExplanationItem(BaseModel):
    feature: str
    value: object | None
    direction: str
    contribution: float | None
    clinical_note: str


class PredictionResponse(BaseModel):
    probability: float
    risk_category: str
    message: str
    threshold: float
    model_name: str
    explanation: list[ExplanationItem]
    disclaimer: str

