@echo off
REM TrustGraph demo launcher: double-click to open the demo in your browser.
REM Builds demo\data.js from saved outputs on first use. Never changes project code.
cd /d "%~dp0"
if exist "demo\data.js" goto open
echo Building demo data - first run only, about a minute...
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" demo\export_demo_data.py
if errorlevel 1 goto fail
:open
start "" "%~dp0demo\index.html"
exit /b 0
:fail
echo.
echo Could not build the demo data. See the messages above.
pause
exit /b 1
