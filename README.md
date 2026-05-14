# Early CVD Prediction

Research-grade heart disease prediction system built from an original notebook and two public benchmark datasets.  
The project turns notebook experiments into a reproducible ML pipeline, evaluation package, and localhost web app.

> This is an educational and research prototype. It is not a clinical diagnostic device.

## What I Built

- Audited the original neural-network notebook before rewriting anything.
- Found that the uploaded `heart.csv` has `1,025` rows but only `302` unique rows.
- Used `heart_disease_uci.csv` as the primary research dataset because it contains hospital/center information.
- Standardized schema differences such as `thalach` vs `thalch` and `target` vs `num`.
- Built leakage-safe preprocessing with imputation, encoding, and scaling inside model pipelines.
- Used site-aware internal-external validation by holding out one center at a time.
- Compared logistic regression, random forest, gradient boosting, calibrated boosting, and an MLP baseline.
- Selected the final model using discrimination, calibration, stability, and interpretability, not accuracy alone.
- Added explainable prediction output for the frontend.
- Built a FastAPI localhost app with a clinical-style input form, probability, risk category, explanation panel, metadata, and disclaimer.
- Generated research reports, figures, tables, model card, manuscript draft, tests, Docker files, and CI workflow.

## Final Model

The selected model is **penalized logistic regression**.

It was chosen because it gave the best overall balance of:

- stable site-aware validation performance
- good calibration behavior
- interpretability
- simple deployment

Primary site-aware mean results:

| Metric | Value |
|---|---:|
| AUROC | 0.799 |
| AUPRC | 0.880 |
| Brier score | 0.168 |
| Balanced accuracy | 0.745 |

## What Is Inside

```text
configs/              Reproducible project configuration
data/raw/             Provided datasets
docs/                 Current work, data, leakage, and architecture audits
src/                  Clean Python package
src/domain/           Feature definitions and risk rules
src/application/      Training, evaluation, and reporting use cases
src/infrastructure/   Data loading, config, tracking, persistence
src/models/           Candidate model definitions
src/evaluation/       Metrics, calibration, bootstrap CI, decision curve, XAI
src/api/              FastAPI backend
src/webapp/           Frontend templates and styling
reports/              Generated figures, tables, manuscript, model card
artifacts/            Saved model bundle and metadata
tests/                Unit and API tests
docker/               Docker setup
.github/workflows/    GitHub Actions CI
```

## Quick Run on Windows

Double-click:

```text
run_app.bat
```

Then open:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Other helpers:

```text
train_project.bat       Retrain and evaluate all models
generate_reports.bat    Regenerate audit and report files
run_tests.bat           Run the test suite
```

## Manual Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run the app:

```powershell
early-cvd serve --host 127.0.0.1 --port 8000
```

Train models:

```powershell
early-cvd train --config configs/default.yaml
```

Generate reports:

```powershell
early-cvd generate-report --config configs/default.yaml
```

Run tests:

```powershell
pytest
```

## Docker

```powershell
docker compose -f docker/docker-compose.yml up --build
```

## Main Reports

- `docs/current_work_audit.md`
- `docs/data_audit_report.md`
- `docs/leakage_audit_report.md`
- `docs/architecture.md`
- `reports/model_card.md`
- `reports/final_executive_summary.md`
- `reports/manuscript/manuscript_draft.md`

## Methodology Summary

The project predicts binary heart disease presence.  
For the UCI-style dataset, `num = 0` means absence and `num > 0` means presence.

The main validation is center-aware:

1. Hold out one hospital/center.
2. Train and tune models on the remaining centers.
3. Evaluate on the held-out center.
4. Repeat for each center.

This gives a more honest estimate than a single random split.

## Important Limitation

The data are small, historical, public benchmark datasets.  
The results are useful for ML research, education, and portfolio presentation, but not for clinical deployment.

## Tech Stack

Python, pandas, scikit-learn, FastAPI, Pydantic, Jinja2, MLflow, DVC, pytest, Docker, GitHub Actions.
