#!/usr/bin/env bash
# Start the Arjuna web app: http://localhost:8000
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "Creating virtual environment (.venv) and installing requirements…"
  $PY -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements-web.txt
fi
exec .venv/bin/python -m web.backend.app "$@"
