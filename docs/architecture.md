# Final Project Architecture

## Layers

- `src/domain`: feature definitions, target rules, and risk-category rules.
- `src/application`: training, evaluation, calibration, and reporting use cases.
- `src/infrastructure`: CSV readers, configuration loading, persistence, and MLflow adapter.
- `src/interfaces`: command-line interface.
- `src/api`: FastAPI controllers and Pydantic request/response schemas.
- `src/webapp`: localhost frontend templates and static assets.
- `tests`: unit, integration, and API contract tests.

## Design Decisions

The web application uses FastAPI with integrated Jinja templates. This is cleaner than adding a second Streamlit service because the project needs one public prediction workflow, automatic API documentation, typed request validation, and a single deployable localhost process.

The prediction path does not use an LLM framework. Explanations are derived from the fitted model artifact and feature dictionary.

The hospital/center field is used for internal-external validation and excluded from model predictors to avoid learning center prevalence shortcuts.
