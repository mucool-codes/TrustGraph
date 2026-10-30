@echo off
REM Rebuild demo\data.js from the saved outputs, using the project's .venv.
REM With PyTorch available this also re-runs the frozen C head for exact per-step trust.
cd /d "%~dp0.."
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" demo\export_demo_data.py
echo.
pause
