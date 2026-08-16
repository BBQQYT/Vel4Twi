@echo off
setlocal enabledelayedexpansion
title Vel4Twi - AI Streamer (Velpur)

echo =======================================================
echo          Vel4Twi - AI Streamer Velpur
echo =======================================================
echo.

REM 1. Check Python
echo [*] Checking Python installation...
set "PYTHON_CMD="
for %%P in (python3 python py) do (
    if not defined PYTHON_CMD (
        %%P --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=%%P"
        )
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] Python was not found in your PATH!
    echo Please install Python 3.9+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [+] Python detected:
%PYTHON_CMD% --version
echo.

REM 2. Check FFmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] FFmpeg was not found in PATH.
    echo Audio processing and mic might not work properly.
    echo.
)

REM 3. Virtual Environment
set "VENV_DIR=venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [*] Creating virtual environment (.%VENV_DIR%)...
    %PYTHON_CMD% -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [+] Virtual environment created successfully.
    echo.
)

call "%VENV_DIR%\Scripts\activate.bat"

REM 4. Dependencies
echo [*] Checking pip and dependencies...
python -m pip install --upgrade pip >nul 2>&1

if exist "requirements.txt" (
    python -c "import torch" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [*] Installing PyTorch with CUDA support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
    echo [*] Installing requirements...
    pip install -r requirements.txt
    echo [+] Dependencies ready.
    echo.
)

REM 5. Config files
if not exist ".env" (
    if exist ".env.example" (
        echo [*] Creating .env from template...
        copy .env.example .env >nul
    )
)
if not exist "config.json" (
    if exist "config.example.json" (
        echo [*] Creating config.json from template...
        copy config.example.json config.json >nul
    )
)

REM 6. Run Application
echo =======================================================
echo          Starting AI Streamer Velpur...
echo =======================================================
echo.

if exist "main.py" (
    python main.py
) else if exist "app.py" (
    python app.py
) else if exist "run.py" (
    python run.py
) else (
    echo [ERROR] Entry point (main.py, app.py, or run.py) not found!
)

if %errorlevel% neq 0 (
    echo.
    echo [!] Process exited with code: %errorlevel%
)

echo.
pause
