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

echo Running tests...
".venv\Scripts\python.exe" -m pytest
pause
