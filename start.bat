@echo off
chcp 65001 >nul
echo ✧ Starting Velpur AI Streamer ✧...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.9+.
    pause
    exit /b
)

:: Install requirements if requirements.txt exists
if exist requirements.txt (
    echo ❀ Installing dependencies...
    pip install -r requirements.txt
)

:: Run the application
echo ⋆ Launching Velpur...
python avatar_main.py

pause
