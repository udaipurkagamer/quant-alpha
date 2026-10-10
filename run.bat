@echo off
REM Run the alpha report on Windows. Extra args go to run.py, e.g. run.bat --data yfinance
cd /d "%~dp0"
if not exist venv\Scripts\python.exe call setup.bat
venv\Scripts\python run.py %*
echo.
echo Results: reports\summary.csv and reports\pnl.png
pause
