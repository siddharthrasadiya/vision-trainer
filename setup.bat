@echo off
REM One-time setup: creates a virtual environment and installs the libraries.
cd /d "%~dp0"

REM Find Python: try the "py" launcher first, then "python"
set PY=
py -3.12 --version >nul 2>&1 && set PY=py -3.12
if not defined PY py -3.11 --version >nul 2>&1 && set PY=py -3.11
if not defined PY py --version >nul 2>&1 && set PY=py
if not defined PY python --version >nul 2>&1 && set PY=python

if not defined PY (
    echo.
    echo ERROR: Python is not installed.
    echo 1. Download Python 3.12 from https://www.python.org/downloads/
    echo 2. In the installer, TICK "Add python.exe to PATH"
    echo 3. Close this window and double-click setup.bat again
    echo.
    pause
    exit /b 1
)

echo Using: %PY%
%PY% --version
%PY% -m venv venv || (echo ERROR: could not create venv & pause & exit /b 1)
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo ERROR: install failed - copy the message above & pause & exit /b 1)

echo.
echo Setup done! Start the app with: run_app.bat
echo   venv\Scripts\activate
echo   python collect.py pen
echo   python collect.py mobile
echo   python collect.py hand
echo   python collect.py nothing
echo   python train.py
echo   python predict.py
pause
