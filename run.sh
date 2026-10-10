#!/usr/bin/env bash
# Run the alpha report on Mac/Linux. Extra args go to run.py, e.g. ./run.sh --data yfinance
set -e
cd "$(dirname "$0")"
[ -x venv/bin/python ] || ./setup.sh
venv/bin/python run.py "$@"
echo "Results: reports/summary.csv and reports/pnl.png"
