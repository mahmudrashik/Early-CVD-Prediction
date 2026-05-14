from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.schemas import PredictionRequest, PredictionResponse
from domain.feature_dictionary import FEATURE_COLUMNS
from domain.risk import risk_category, risk_message
from evaluation.explain import local_explanation
from infrastructure.config import resolve_path
from infrastructure.persistence import load_bundle


APP_ROOT = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(APP_ROOT / "webapp" / "templates"))

app = FastAPI(
    title="Early CVD Prediction API",
    description="Research prototype for public-dataset heart disease prediction. Not for clinical deployment.",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory=str(APP_ROOT / "webapp" / "static")), name="static")


def _bundle() -> dict[str, object]:
    path = resolve_path("artifacts/model_bundle.joblib")
    if not path.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found. Run `early-cvd train` first.")
    return load_bundle(path)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    metadata = {}
    try:
        metadata = _bundle().get("metadata", {})
    except HTTPException:
        metadata = {"model_name": "not trained", "disclaimer": "Run training before using predictions."}
    return templates.TemplateResponse(request, "index.html", {"metadata": metadata})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model-metadata")
def model_metadata() -> dict[str, object]:
    return _bundle().get("metadata", {})


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    bundle = _bundle()
    model = bundle["model"]
    metadata = bundle.get("metadata", {})
    X = pd.DataFrame([request.model_dump()])[FEATURE_COLUMNS]
    probability = float(model.predict_proba(X)[:, 1][0])
    explanation = local_explanation(bundle, X)
    threshold = float(metadata.get("threshold", 0.5))
    return PredictionResponse(
        probability=round(probability, 4),
        risk_category=risk_category(probability),
        message=risk_message(probability),
        threshold=threshold,
        model_name=str(metadata.get("model_name", "unknown")),
        explanation=explanation,
        disclaimer=str(metadata.get("disclaimer", "Research prototype; not for clinical deployment.")),
    )
