#!/usr/bin/env bash
# One-time setup on Mac/Linux: creates venv/ and installs dependencies.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python) || { echo "Python 3 not found; install it first."; exit 1; }
[ -d venv ] || "$PY" -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt pytest
echo "Setup done. Now run: ./run.sh   (or ./run.sh --data yfinance)"
