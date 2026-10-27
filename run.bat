@echo off
REM Start the Arjuna web app: http://localhost:8000
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment (.venv) and installing requirements...
  python -m venv .venv
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\pip install -q -r requirements-web.txt
)
.venv\Scripts\python -m web.backend.app
