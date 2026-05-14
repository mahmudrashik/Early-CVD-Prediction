@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%\src

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)

echo Installing/updating project dependencies...
".venv\Scripts\python.exe" -m pip install -e ".[dev]"

if not exist "artifacts\model_bundle.joblib" (
  echo Model artifact not found. Training the model first...
  ".venv\Scripts\early-cvd.exe" train --config configs/default.yaml
)

echo.
echo Starting Early CVD Prediction on http://127.0.0.1:8000
echo API docs: http://127.0.0.1:8000/docs
echo Press Ctrl+C to stop the server.
echo.
".venv\Scripts\early-cvd.exe" serve --host 127.0.0.1 --port 8000
