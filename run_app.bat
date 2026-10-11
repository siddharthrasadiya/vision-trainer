@echo off
REM Starts the Vision Trainer web app
cd /d "%~dp0"
if not exist venv (echo Run setup.bat first & pause & exit /b 1)
call venv\Scripts\activate
python -c "import flask" 2>nul || python -m pip install flask
python app.py
pause
