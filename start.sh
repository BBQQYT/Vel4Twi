#!/usr/bin/env bash
set -e

# Цветовая подсветка вывода
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # Без цвета

echo -e "${CYAN}=======================================================${NC}"
echo -e "${CYAN}         ✦ Vel4Twi - AI Streamer Velpur ✦${NC}"
echo -e "${CYAN}=======================================================${NC}\n"

# 1. Проверка наличия Python
echo -e "[*] Проверка Python..."
PYTHON_CMD=""

for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}[ERROR] Python не найден! Пожалуйста, установите Python 3.9+${NC}"
    exit 1
fi

echo -e "${GREEN}[+] Найден Python:${NC} $($PYTHON_CMD --version)"

# 2. Проверка FFmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo -e "${YELLOW}[WARN] FFmpeg не найден! Для работы со звуком установите его (например: sudo apt install ffmpeg)${NC}"
fi

# 3. Создание и активация виртуального окружения
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "[*] Создание виртуального окружения (${VENV_DIR})..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}[+] Виртуальное окружение создано.${NC}"
fi

source "${VENV_DIR}/bin/activate"

# 4. Проверка и установка зависимостей
echo -e "[*] Проверка pip..."
pip install --upgrade pip >/dev/null 2>&1

if [ -f "requirements.txt" ]; then
    echo -e "[*] Проверка PyTorch..."
    if ! python -c "import torch" >/dev/null 2>&1; then
        echo -e "[*] Установка PyTorch (CUDA)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || pip install torch torchvision torchaudio
    fi
    echo -e "[*] Установка остальных зависимостей из requirements.txt..."
    pip install -r requirements.txt
    echo -e "${GREEN}[+] Все зависимости установлены.${NC}"
fi

# 5. Проверка файлов конфигурации
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo -e "[*] Создание .env из .env.example..."
    cp .env.example .env
fi

if [ ! -f "config.json" ] && [ -f "config.example.json" ]; then
    echo -e "[*] Создание config.json из config.example.json..."
    cp config.example.json config.json
fi

# 6. Запуск проекта
echo -e "\n${CYAN}=======================================================${NC}"
echo -e "${CYAN}         Запуск AI-стримерши Velpur...${NC}"
echo -e "${CYAN}=======================================================${NC}\n"

if [ -f "main.py" ]; then
    python main.py
elif [ -f "app.py" ]; then
    python app.py
elif [ -f "run.py" ]; then
    python run.py
else
    echo -e "${RED}[ERROR] Файл запуска (main.py / app.py / run.py) не найден!${NC}"
    exit 1
fi
