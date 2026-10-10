@echo off
REM One-time setup on Windows: creates venv\ and installs dependencies.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version >nul 2>nul || (
    echo Python not found. Install it from https://www.python.org/downloads/
    echo and tick "Add Python to PATH" during install.
    pause & exit /b 1
)
if not exist venv %PY% -m venv venv || (pause & exit /b 1)
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt pytest || (pause & exit /b 1)
echo.
echo Setup done. Now double-click run.bat (or: run.bat --data yfinance)
pause
