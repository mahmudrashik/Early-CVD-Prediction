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

echo Generating audit and documentation reports...
".venv\Scripts\early-cvd.exe" generate-report --config configs/default.yaml

echo.
echo Done. Open docs\ and reports\ to view the generated files.
pause
