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

echo Running site-aware training and evaluation...
".venv\Scripts\early-cvd.exe" train --config configs/default.yaml

echo.
echo Done. Results are in reports\ and artifacts\.
pause
