<div align="center">
  
  <!-- Анимированный заголовок -->
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12&height=200&section=header&text=Vel4Twi%20🎆&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=35" />
  
  <h1>🎥 Локальный AI-стример Velpur</h1>
  
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&multiline=true&width=600&height=80&lines=🤖+Полнофункциональная+AI-стримерша;🎨+VTube+Studio+%7C+Discord+%7C+Twitch;🧠+Локальные+LLM+%7C+Whisper+%7C+Coqui+TTS" alt="Typing SVG" />
  
  <p>
    <img src="https://img.shields.io/github/stars/BBQQYT/Vel4Twi?style=for-the-badge&logo=github&logoColor=white&color=ff6b6b" alt="Stars" />
    <img src="https://img.shields.io/github/forks/BBQQYT/Vel4Twi?style=for-the-badge&logo=github&logoColor=white&color=6366f1" alt="Forks" />
    <img src="https://img.shields.io/github/license/BBQQYT/Vel4Twi?style=for-the-badge&color=10b981" alt="License" />
    <img src="https://img.shields.io/github/last-commit/BBQQYT/Vel4Twi?style=for-the-badge&color=f59e0b" alt="Last Commit" />
  </p>
  
</div>

---

## 🎯 О проекте

**Vel4Twi** — это революционный проект с открытым исходным кодом, позволяющий запустить полнофункциональную AI-стримершу по имени **Velpur** прямо на вашем компьютере. 

🚀 **Velpur** — это не просто чат-бот. Это виртуальная личность с собственным характером, памятью и способностью взаимодействовать со зрителями.

---

## ✨ Ключевые возможности

<table>
  <tr>
    <td width="33%" align="center">
      <h3>🤖 Продвинутый AI</h3>
      <p>Локальная LLM через LM Studio</p>
      <p>Характер Velpur через детальный промпт</p>
    </td>
    <td width="33%" align="center">
      <h3>🗣️ Обработка речи</h3>
      <p>STT: Whisper для распознавания</p>
      <p>TTS: Coqui TTS + XTTS v2</p>
    </td>
    <td width="33%" align="center">
      <h3>🎭 VTube Studio</h3>
      <p>Автоматические анимации</p>
      <p>Lip Sync и Idle-анимации</p>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td width="50%" align="center">
      <h3>💬 Мультиплатформенность</h3>
      <p>• Discord (текст + голос)</p>
      <p>• Twitch Chat</p>
      <p>• Командная строка</p>
    </td>
    <td width="50%" align="center">
      <h3>🧠 Долгосрочная память</h3>
      <p>• SQLite база данных</p>
      <p>• Автоизвлечение фактов</p>
      <p>• История диалогов</p>
    </td>
  </tr>
</table>

---

## 🔧 Технологический стек

<div align="center">
  
  <table>
    <tr>
      <td align="center" width="20%">
        <h4>🚀 Основа</h4>
        <img src="https://skillicons.dev/icons?i=python,pytorch&theme=dark" /><br/>
        <strong>Python + PyTorch</strong>
      </td>
      <td align="center" width="20%">
        <h4>🤖 AI/ML</h4>
        <img src="https://img.shields.io/badge/Whisper-25D366?style=for-the-badge&logo=openai&logoColor=white" /><br/>
        <img src="https://img.shields.io/badge/LM_Studio-000000?style=for-the-badge&logo=microsoft&logoColor=white" /><br/>
        <img src="https://img.shields.io/badge/Coqui_TTS-FF6B35?style=for-the-badge&logo=python&logoColor=white" />
      </td>
      <td align="center" width="20%">
        <h4>🎮 Интеграции</h4>
        <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" /><br/>
        <img src="https://img.shields.io/badge/Twitch-9146FF?style=for-the-badge&logo=twitch&logoColor=white" /><br/>
        <img src="https://img.shields.io/badge/VTube_Studio-FF69B4?style=for-the-badge&logo=youtube&logoColor=white" />
      </td>
      <td align="center" width="20%">
        <h4>💾 Данные</h4>
        <img src="https://skillicons.dev/icons?i=sqlite&theme=dark" /><br/>
        <strong>SQLite + JSON</strong>
      </td>
      <td align="center" width="20%">
        <h4>🔊 Аудио</h4>
        <img src="https://img.shields.io/badge/FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white" /><br/>
        <strong>VB-CABLE</strong>
      </td>
    </tr>
  </table>
  
</div>

---

## 🛠️ Архитектура

```mermaid
graph TB
    A[💬 Discord/Twitch Input] --> B[🧠 Orchestrator]
    C[🎤 Voice Input] --> D[🗣️ STT Module]
    D --> B
    B --> E[💾 Memory Module]
    E --> F[🤖 LLM Module]
    F --> G[🔊 TTS Module]
    G --> H[🎭 VTube Studio]
    G --> I[🔈 Audio Output]
    B --> H
```

### 📋 Принцип работы

1. **📲 Ввод:** Получение сообщений из Discord, Twitch или голосовых сообщений
2. **🔍 Обработка:** Whisper преобразует голос в текст
3. **🧠 Мышление:** Локальная LLM генерирует ответ с учетом контекста
4. **🗣️ Ответ:** Coqui TTS синтезирует голос Velpur
5. **🎭 Анимация:** VTube Studio оживляет аватар

---

## 🚀 Быстрый старт

### ⚙️ Требования

- 🐍 **Python 3.9+**
- 🔥 **PyTorch** (с поддержкой CUDA для GPU)
- 🎨 **LM Studio**
- 🎭 **VTube Studio** 
- 🔊 **Виртуальный аудиокабель** (VB-CABLE)

### 💻 Установка

```bash
# Клонирование репозитория
git clone https://github.com/BBQQYT/Vel4Twi.git
cd Vel4Twi

# Установка зависимостей
pip install -r requirements.txt

# Установка FFmpeg (необходим для аудио)
# Windows: скачать с официального сайта
# Linux: sudo apt install ffmpeg
```

### ⚙️ Настройка

<details>
<summary>🔧 <strong>Подробная настройка</strong></summary>

#### 1. **LM Studio**
- Скачайте и запустите LM Studio
- Загрузите совместимую модель (Mistral, Llama и т.д.)
- Запустите Local Server

#### 2. **VTube Studio** 
- Запустите VTube Studio и загрузите аватар
- Включите API (Start API)
- Создайте хоткеи для анимаций

#### 3. **config.json**
При первом запуске файл создастся автоматически. Отредактируйте:
- `discord_token`: Токен вашего Discord-бота
- `twitch_token`, `twitch_nickname`, `twitch_channel`: Данные Twitch
- `llm_model_name_lmstudio`: Имя модели из LM Studio
- `speaker_wav_path_tts`: Путь к .wav файлу для клонирования голоса

</details>

### 🚀 Запуск

```bash
python main.py
```

При первом запуске VTube Studio запросит разрешение на подключение. Разрешите его!

---

## 📈 Дорожная карта

- [ ] 🕰️ **v1.1** - Компьютерное зрение (OCR)
- [ ] 🌍 **v1.2** - Веб-интерфейс для настроек
- [ ] 🎥 **v1.3** - Поддержка YouTube Live
- [ ] 🤖 **v2.0** - Мультимодальные модели (Vision + Audio)

---

## 🤝 Вклад в развитие

Мы приветствуем любой вклад! 🎉

1. **Fork** репозитория
2. **Создайте** feature ветку
3. **Отправьте** Pull Request

### Идеи для вклада:
- 🔧 Оптимизация производительности
- 🌍 Новые платформы интеграции
- 🎨 Улучшение анимаций
- 📝 Документация и переводы

---

## 📜 Лицензия

Этот проект распространяется под лицензией **GPL-3.0**. Подробности см. в файле [LICENSE](LICENSE).

---

## 🙏 Благодарности

Спасибо командам за создание невероятных технологий:
- **OpenAI** за Whisper
- **Coqui** за открытое TTS-решение
- **LM Studio** за удобный интерфейс для локальных LLM
- **VTube Studio** за поддержку API

---

<div align="center">
  
  <h2>💖 Поддержать проект</h2>
  
  <p>Если Vel4Twi помог вам создать что-то удивительное:</p>
  
  <a href="#">
    <img src="https://img.shields.io/badge/%E2%AD%90_%D0%9F%D0%BE%D1%81%D1%82%D0%B0%D0%B2%D1%8C%D1%82%D0%B5_%D0%B7%D0%B2%D0%B5%D0%B7%D0%B4%D1%83-FFD700?style=for-the-badge&logo=github&logoColor=black" alt="Star" />
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/%F0%9F%94%84_Fork_%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82-6366F1?style=for-the-badge&logo=github&logoColor=white" alt="Fork" />
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/%F0%9F%92%AC_%D0%A1%D0%BE%D0%BE%D0%B1%D1%89%D0%B8%D1%82%D1%8C_%D0%BE_%D0%B1%D0%B0%D0%B3%D0%B5-FF6B6B?style=for-the-badge&logo=github&logoColor=white" alt="Report Bug" />
  </a>
  
  <!-- Волна внизу -->
  <br/><br/>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12&height=120&section=footer" />
  
</div>