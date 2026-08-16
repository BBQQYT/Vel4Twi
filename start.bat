@echo off
setlocal enabledelayedexpansion
title Vel4Twi - AI Streamer (Velpur)

echo =======================================================
echo          Vel4Twi - AI Streamer Velpur
echo =======================================================
echo.

REM 1. Поиск и активация Conda
set "CONDA_ACTIVATE="
if exist "F:\miniconda\Scripts\activate.bat" set "CONDA_ACTIVATE=F:\miniconda\Scripts\activate.bat"
if exist "F:\miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=F:\miniconda3\Scripts\activate.bat"
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%USERPROFILE%\anaconda3\Scripts\activate.bat"
if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%LOCALAPPDATA%\miniconda3\Scripts\activate.bat"
if exist "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=%LOCALAPPDATA%\anaconda3\Scripts\activate.bat"
if exist "C:\miniconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=C:\miniconda3\Scripts\activate.bat"
if exist "C:\anaconda3\Scripts\activate.bat" set "CONDA_ACTIVATE=C:\anaconda3\Scripts\activate.bat"

if defined CONDA_ACTIVATE goto CONDA_SETUP

for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
    if not defined CONDA_ACTIVATE if exist "%%~dpI..\Scripts\activate.bat" set "CONDA_ACTIVATE=%%~dpI..\Scripts\activate.bat"
    if not defined CONDA_ACTIVATE if exist "%%~dpIactivate.bat" set "CONDA_ACTIVATE=%%~dpIactivate.bat"
)

if defined CONDA_ACTIVATE goto CONDA_SETUP

goto SYSTEM_PYTHON_SETUP

:CONDA_SETUP
echo [+] Conda detected: "%CONDA_ACTIVATE%"
call "%CONDA_ACTIVATE%" base

call conda env list > "%TEMP%\conda_envs.txt" 2>nul
findstr /C:"vel4twi" "%TEMP%\conda_envs.txt" >nul 2>&1
if errorlevel 1 (
    echo [*] Creating isolated conda env 'vel4twi' with Python 3.10...
    call conda create -n vel4twi python=3.10 -y
    echo [*] Installing FFmpeg from conda-forge...
    call conda install -n vel4twi -c conda-forge ffmpeg -y
)
if exist "%TEMP%\conda_envs.txt" del "%TEMP%\conda_envs.txt" 2>nul

echo [*] Activating conda environment 'vel4twi'...
call "%CONDA_ACTIVATE%" vel4twi
goto RUN_BOOTSTRAP

:SYSTEM_PYTHON_SETUP
echo [-] Conda not found. Using system Python...
set "PYTHON_CMD="
for %%P in (python3 python py) do (
    if not defined PYTHON_CMD (
        %%P --version >nul 2>&1
        if !errorlevel! equ 0 set "PYTHON_CMD=%%P"
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] Python not found in PATH!
    echo Please install Python 3.10+ or Miniconda.
    pause
    exit /b 1
)

echo [+] Python detected:
%PYTHON_CMD% --version

set "VENV_DIR=venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [*] Creating venv...
    %PYTHON_CMD% -m venv %VENV_DIR%
)
call "%VENV_DIR%\Scripts\activate.bat"

:RUN_BOOTSTRAP
echo [*] Initializing Python bootstrap...
(
echo import subprocess, sys, os
echo def get_gpu_info^(^):
echo     paths = ["nvidia-smi", r"C:\Windows\System32\nvidia-smi.exe", r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]
echo     for p in paths:
echo         try:
echo             res = subprocess.run([p, "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True^)
echo             if res.returncode == 0 and res.stdout.strip^(^):
echo                 lines = [l.strip^(^) for l in res.stdout.strip^(^).splitlines^(^) if l.strip^(^)]
echo                 if lines:
echo                     parts = lines[0].split(","^)
echo                     return parts[0].strip^(^), parts[1].strip^(^) if len^(parts^) ^> 1 else ""
echo         except Exception:
echo             continue
echo     return None, None
echo.
echo # 1. Fix setuptools for librosa ^& TTS (requires pkg_resources^)
echo subprocess.run([sys.executable, "-m", "pip", "install", "setuptools<70", "wheel"], check=False^)
echo.
echo # 2. PyTorch ^& CUDA check
echo need_torch = True
echo try:
echo     import torch
echo     if torch.cuda.is_available^(^):
echo         print^(f"[+] PyTorch CUDA is ACTIVE on: {torch.cuda.get_device_name(0)}"^)
echo         need_torch = False
echo     else:
echo         print^("[-] Installed PyTorch has no CUDA support. Reinstalling..."^)
echo except ImportError:
echo     pass
echo.
echo if need_torch:
echo     gpu_name, driver = get_gpu_info^(^)
echo     if gpu_name:
echo         print^(f"[+] GPU Detected: {gpu_name} (Driver: {driver})"^)
echo         print^("[*] Installing PyTorch with CUDA 12.1 support..."^)
echo         subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"], check=True^)
echo     else:
echo         print^("[-] No NVIDIA GPU detected. Installing CPU PyTorch..."^)
echo         subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True^)
echo.
echo # 3. Project Requirements
echo extra_pkgs = ["webrtcvad-wheels", "pyaudio", "mss", "pytesseract", "pillow", "pytchat"]
echo subprocess.run([sys.executable, "-m", "pip", "install", *extra_pkgs], check=False^)
echo.
echo if os.path.exists^("requirements.txt"^):
echo     print^("[*] Installing requirements.txt..."^)
echo     subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True^)
echo.
echo # Final verification of pkg_resources
echo subprocess.run([sys.executable, "-m", "pip", "install", "setuptools<70"], check=False^)
echo print^("[+] Environment is fully ready!"^)
) > "%TEMP%\vel4twi_bootstrap.py"

python "%TEMP%\vel4twi_bootstrap.py"
if exist "%TEMP%\vel4twi_bootstrap.py" del "%TEMP%\vel4twi_bootstrap.py" 2>nul

REM 4. Конфигурационные файлы
if not exist ".env" if exist ".env.example" copy .env.example .env >nul
if not exist "config.json" if exist "config.example.json" copy config.example.json config.json >nul

echo.
echo =======================================================
echo          Starting Vel4Twi (Velpur)...
echo =======================================================
echo.

REM 5. Запуск точки входа
set "APP_ENTRY="
if exist "main.py" set "APP_ENTRY=main.py"
if not defined APP_ENTRY if exist "src\main.py" set "APP_ENTRY=src\main.py"
if not defined APP_ENTRY if exist "app.py" set "APP_ENTRY=app.py"
if not defined APP_ENTRY if exist "src\app.py" set "APP_ENTRY=src\app.py"
if not defined APP_ENTRY if exist "run.py" set "APP_ENTRY=run.py"

if defined APP_ENTRY goto RUN_APP

echo [ERROR] Entry point (main.py / app.py / run.py) not found!
goto FINISH

:RUN_APP
echo [*] Running: %APP_ENTRY%
python "%APP_ENTRY%"

:FINISH
if errorlevel 1 (
    echo.
    echo [!] Program exited with code: %errorlevel%
)

echo.
pause
