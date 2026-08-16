@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Vel4Twi - AI Streamer (Velpur)

echo =======================================================
echo          ✦ Vel4Twi - AI Streamer Velpur ✦
echo =======================================================
echo.

:: 1. Проверка наличия Python
echo [*] Проверка наличия Python...
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
    echo [ERROR] Python не найден в системе!
    echo Пожалуйста, установите Python 3.9+ с сайта https://www.python.org/
    echo Обязательно отметьте галочку "Add Python to PATH" при установке.
    echo.
    pause
    exit /b 1
)

echo [+] Python обнаружен:
%PYTHON_CMD% --version
echo.

:: 2. Проверка FFmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] FFmpeg не найден в переменной PATH.
    echo Обработка аудио и микрофона может работать с ошибками.
    echo Рекомендуется установить FFmpeg и добавить его в PATH.
    echo.
)

:: 3. Создание и активация виртуального окружения
set "VENV_DIR=venv"
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [*] Создание виртуального окружения (.%VENV_DIR%)...
    %PYTHON_CMD% -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo [ERROR] Не удалось создать виртуальное окружение.
        pause
        exit /b 1
    )
    echo [+] Виртуальное окружение успешно создано.
    echo.
)

call "%VENV_DIR%\Scripts\activate.bat"

:: 4. Установка зависимостей
echo [*] Проверка pip и зависимостей...
python -m pip install --upgrade pip >nul 2>&1

if exist "requirements.txt" (
    python -c "import torch" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [*] Установка PyTorch с поддержкой CUDA...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    )
    echo [*] Проверка и установка пакетов из requirements.txt...
    pip install -r requirements.txt
    echo [+] Все зависимости готовы.
    echo.
)

:: 5. Проверка файлов конфигурации
if not exist ".env" (
    if exist ".env.example" (
        echo [*] Создание .env из шаблона .env.example...
        copy .env.example .env >nul
    )
)
if not exist "config.json" (
    if exist "config.example.json" (
        echo [*] Создание config.json из шаблона config.example.json...
        copy config.example.json config.json >nul
    )
)

:: 6. Запуск проекта
echo =======================================================
echo          Запуск AI-стримерши Velpur...
echo =======================================================
echo.

if exist "main.py" (
    python main.py
) else if exist "app.py" (
    python app.py
) else if exist "run.py" (
    python run.py
) else (
    echo [ERROR] Не найден исполняемый файл (main.py, app.py или run.py)!
)

if %errorlevel% neq 0 (
    echo.
    echo [!] Программа завершилась с кодом ошибки: %errorlevel%
)

echo.
pause
