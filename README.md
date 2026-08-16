<div align="center">
  
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=9B72C7,60BA70&height=200&section=header&text=Vel4Twi%20%E2%9C%A7&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=35" />
  
  <h1> ✧ Локальный AI-стример Velpur ✧ </h1>
  
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=22&duration=3000&pause=1000&color=9B72C7&center=true&vCenter=true&multiline=true&width=600&height=80&lines=%E2%9C%A7+%D0%9F%D0%BE%D0%BB%D0%BD%D0%BE%D1%84%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D0%BE%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F+AI-%D1%81%D1%82%D1%80%D0%B8%D0%BC%D0%B5%D1%80%D1%88%D0%B0;%E2%9C%A7+VTube+Studio+%7C+Discord+%7C+Twitch;%E2%9C%A7+%D0%9B%D0%BE%D0%BA%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5+LLM+%7C+Whisper+%7C+Coqui+TTS" alt="Typing SVG" />
  
  <p>
    <img src="https://img.shields.io/github/stars/BBQQYT/Vel4Twi?style=for-the-badge&logo=github&logoColor=white&color=9B72C7" alt="Stars" />
    <img src="https://img.shields.io/github/forks/BBQQYT/Vel4Twi?style=for-the-badge&logo=github&logoColor=white&color=60BA70" alt="Forks" />
    <img src="https://img.shields.io/github/license/BBQQYT/Vel4Twi?style=for-the-badge&color=9B72C7" alt="License" />
    <img src="https://img.shields.io/github/last-commit/BBQQYT/Vel4Twi?style=for-the-badge&color=60BA70" alt="Last Commit" />
  </p>
  
</div>

<p align="center">
  <img src="https://github.com/BBQQYT/Vel4Twi/blob/main/banner.jpg?raw=true" alt="Vel4Twi Banner" width="830" height="500" />
</p>

---

## ⋆ ˚｡⋆ О проекте ⋆ ˚｡⋆

**Vel4Twi** — это нежный и ламповый проект с открытым исходным кодом, который поможет вам запустить полнофункциональную AI-стримершу по имени **Velpur** прямо на вашем компьютере.

✧ **Velpur** — это не просто чат-бот. Это виртуальная личность со своим характером, уютной памятью и возможностью тепло общаться со зрителями.

---

## ❀ Ключевые возможности ❀

<table>
  <tr>
    <td width="33%" align="center">
      <h3>✦ Продвинутый AI</h3>
      <p>Локальная LLM через LM Studio</p>
      <p>Мягкий характер Velpur через промпт</p>
    </td>
    <td width="33%" align="center">
      <h3>✦ Обработка речи</h3>
      <p>STT: Whisper для распознавания</p>
      <p>TTS: Coqui TTS + XTTS v2</p>
    </td>
    <td width="33%" align="center">
      <h3>✦ VTube Studio</h3>
      <p>Автоматические анимации</p>
      <p>Lip Sync и Idle-поведения</p>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td width="50%" align="center">
      <h3>♡ Мультиплатформенность</h3>
      <p>⋆ Discord (текст + голос)</p>
      <p>⋆ Twitch Chat</p>
      <p>⋆ YouTube Live Chat</p>
      <p>⋆ Командная строка</p>
    </td>
    <td width="50%" align="center">
      <h3>♡ Долгосрочная память</h3>
      <p>⋆ SQLite база данных</p>
      <p>⋆ Автоизвлечение фактов</p>
      <p>⋆ История диалогов</p>
    </td>
  </tr>
</table>

---

## ✿ Технологический стек ✿

<div align="center">
  
  <table>
    <tr>
      <td align="center" width="20%">
        <h4>✧ Основа</h4>
        <strong>Python + PyTorch</strong>
      </td>
      <td align="center" width="20%">
        <h4>✧ AI/ML</h4>
        <strong>Whisper, LM Studio, Coqui TTS</strong>
      </td>
      <td align="center" width="20%">
        <h4>✧ Интеграции</h4>
        <strong>Discord, Twitch, YouTube, VTube Studio</strong>
      </td>
      <td align="center" width="20%">
        <h4>✧ Данные</h4>
        <strong>SQLite + JSON</strong>
      </td>
      <td align="center" width="20%">
        <h4>✧ Аудио</h4>
        <strong>FFmpeg, VB-CABLE</strong>
      </td>
    </tr>
  </table>
  
</div>

---

## ୨୧ Принцип работы ୨୧

```mermaid
graph TB
    A[ Ввод: Discord / Twitch / YouTube ] --> B[ Оркестратор ]
    C[ Голосовой ввод ] --> D[ STT Модуль ]
    D --> B
    B --> E[ Модуль памяти ]
    E --> F[ LLM Модуль ]
    F --> G[ TTS Модуль ]
    G --> H[ VTube Studio ]
    G --> I[ Аудио Выход ]
    B --> H
```

⋆ **Получение:** Входящие сообщения из чатов или голосовые сообщения бережно собираются в очередь.
⋆ **Обработка:** Whisper внимательно слушает и преобразует голос в текст.
⋆ **Мышление:** Локальная LLM придумывает ответ с учетом прошлых бесед и очищается от тегов размышлений.
⋆ **Ответ:** Coqui TTS синтезирует мягкий голос Velpur.
⋆ **Анимация:** VTube Studio добавляет жизнь и эмоции аватару.

---

## ⋆ Быстрый старт ⋆

### ❀ Требования

- ⋆ **Python 3.9+**
- ⋆ **PyTorch** (с поддержкой CUDA для GPU)
- ⋆ **LM Studio**
- ⋆ **VTube Studio**
- ⋆ **Виртуальный аудиокабель** (VB-CABLE)

### ❀ Установка

```bash
# Клонирование репозитория
git clone https://github.com/BBQQYT/Vel4Twi.git
cd Vel4Twi

# Установка FFmpeg (необходим для работы со звуком)
# Windows: скачать с официального сайта и добавить в PATH
# Linux: sudo apt install ffmpeg
```

### ❀ Настройка

<details>
<summary>✧ <strong>Подробная настройка</strong></summary>

⋆ **1. LM Studio**
- Скачайте и запустите LM Studio.
- Загрузите подходящую модель.
- Запустите локальный сервер.

⋆ **2. VTube Studio**
- Запустите VTube Studio и загрузите свой аватар.
- Включите API (Start API) и разрешите подключение.
- Настройте горячие клавиши для эмоций.

⋆ **3. Конфигурация**
- При запуске через веб-интерфейс или скрипт создастся нужная конфигурация.
- Вы можете настроить токены Discord, Twitch и другие параметры в удобном Web UI.

</details>

### ❀ Запуск

Теперь для удобства добавлены скрипты автозапуска! Они сами проверят окружение, установят зависимости и запустят проект.

**Для Windows:**
```cmd
start.bat
```

**Для Linux / macOS:**
```bash
bash start.sh
```

При первом запуске VTube Studio мягко попросит разрешение на подключение — обязательно разрешите его.

---

## ❁ Дорожная карта и недавние обновления ❁

- [x] ✦ **v1.1** - Компьютерное зрение (OCR) и поддержка Tool Calling для LLM
- [x] ✦ **v1.2** - Веб-интерфейс для настроек (Web UI на порту 8080)
- [x] ✦ **v1.3** - Интеграция с YouTube Live чатом
- [x] ✦ **v1.4** - Скрипты автозапуска (`start.bat` / `start.sh`) и автоустановка зависимостей
- [x] ✦ **v1.5** - Обработка и фильтрация тегов `<think>` из ответов
- [x] ✦ **v1.6** - Централизованная очередь запросов (Orchestrator)
- [ ] ✧ **v2.0** - Мультимодальные модели (Vision + Audio напрямую)

---

## ♡ Вклад в развитие ♡

Будем очень рады любой вашей помощи и пулл-реквестам!

1. **Форкните** репозиторий ⋆
2. **Создайте** новую ветку ⋆
3. **Отправьте** Pull Request ⋆

---

## ✧ Лицензия ✧

Проект распространяется под лицензией **GPL-3.0**. Все подробности можно найти в файле [LICENSE](LICENSE).

---

<div align="center">
  
  <h2>♡ Поддержать проект ♡</h2>
  
  <p>Если Velpur принесла немного уюта и радости в ваш день:</p>
  
  <a href="#">
    <img src="https://img.shields.io/badge/✧_Поставить_звезду-9B72C7?style=for-the-badge&logo=github&logoColor=white" alt="Star" />
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/✧_Форк_проекта-60BA70?style=for-the-badge&logo=github&logoColor=white" alt="Fork" />
  </a>
  
  <br/><br/>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=9B72C7,60BA70&height=120&section=footer" />
  
</div>