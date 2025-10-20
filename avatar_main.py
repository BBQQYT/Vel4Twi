# Local AI Conversational Avatar

from twitchio.ext import commands as twitch_commands
import os
import asyncio
import json
import sqlite3
import threading
import time
import wave
import io
import base64
import websockets
import numpy as np
from datetime import datetime
import scipy.io.wavfile as wavfile
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from queue import Queue
import logging
import re
import random
import math
import discord
from discord.ext import commands
import torch
import whisper
import TTS
from TTS.api import TTS
import pyaudio
import sounddevice as sd
from PIL import Image
import mss
import pytesseract
import webrtcvad
import librosa
import aiohttp

# ===============================
# CONFIGURATION AND SETTINGS
# ===============================

@dataclass
class Config:
    ### Discord ###
    discord_token: str = "YOUR_DISCORD_BOT_TOKEN"
    discord_guild_id: Optional[int] = None
    discord_channel_id: Optional[int] = None
    discord_voice_channel_id: Optional[int] = None
    language: str = "ru"

    ### AI Models ###
    whisper_model: str = "base"
    whisper_language: Optional[str] = "ru"

    ### LM Studio API settings ###
    llm_api_url: str = "http://localhost:1234/v1/chat/completions"
    llm_api_key: str = "not-needed"
    llm_model_name_lmstudio: str = "loaded-model-name"
    llm_temperature: float = 0.7
    llm_max_tokens_response: int = 250

    ### TTS model ###
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    speaker_wav_path_tts: Optional[str] = "path/to/your/voice.wav"
    
    ### VTube Studio ###
    vtube_studio_host: str = "localhost"
    vtube_studio_port: int = 8001
    
    vtube_hotkey_id_thinking: Optional[str] = "ID_ХОТКЕЯ_ДУМАЕТ" 
    vtube_hotkey_id_speaking: Optional[str] = None # Говорит - через Lip Sync, хоткей не нужен!
    vtube_hotkey_id_quirk: Optional[str] = "ID_ХОТКЕЯ_ДЛЯ_ФИШКИ" # Например, "увидела суши"
    
    vtube_idle_enabled: bool = True
    vtube_idle_blink_interval_min: float = 6.0   # Минимальный интервал моргания (дольше для Dayo)
    vtube_idle_blink_interval_max: float = 12.0  # Максимальный интервал
    vtube_idle_wobble_speed: float = 0.4         # Скорость покачивания
    vtube_idle_wobble_amount_z: float = 6.0      # Амплитуда покачивания (наклон)
    vtube_idle_head_move_interval: float = 10.0  # Как часто делать резкое движение головой
    
    vtube_mouth_open_sensitivity: float = 15.0
    vtube_mouth_open_smoothing_factor: float = 0.7
    
    ### Memory ###
    memory_db_path: str = "avatar_memory.db"
    max_context_length_tokens: int = 4096
    
    ### Audio ###
    sample_rate: int = 16000
    chunk_size: int = 1024
    
    ### Vision ###
    enable_vision: bool = True
    screenshot_interval: int = 5

    ### Twitch ###
    twitch_enabled: bool = False
    twitch_nickname: str = "YOUR_TWITCH_BOT_NICKNAME"
    twitch_token: str = "oauth:YOUR_TWITCH_OAUTH_TOKEN"
    twitch_channel: str = "TARGET_TWITCH_CHANNEL_NAME"
    
    ### Interaction settings ###
    user_cooldown_seconds: int = 60
    bot_trigger_keyword: str = "@Velpur"

# ===============================
# LOGGING SETUP
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('avatar.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===============================
# MEMORY MODULE
# ===============================

class MemoryModule:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY, username TEXT, display_name TEXT,
                first_seen TIMESTAMP, last_seen TIMESTAMP, interaction_count INTEGER DEFAULT 0
            )''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, message TEXT, response TEXT,
                timestamp TIMESTAMP, channel_id TEXT, FOREIGN KEY (user_id) REFERENCES users (user_id)
            )''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, fact_key TEXT, fact_value TEXT,
                confidence REAL DEFAULT 1.0, timestamp TIMESTAMP, FOREIGN KEY (user_id) REFERENCES users (user_id)
            )''')
        conn.commit()
        conn.close()
        
    def add_user(self, user_id: str, username: str, display_name: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, username, display_name, first_seen, last_seen, interaction_count)
            VALUES (?, ?, ?, 
                COALESCE((SELECT first_seen FROM users WHERE user_id = ?), ?),
                ?, 
                COALESCE((SELECT interaction_count FROM users WHERE user_id = ?), 0) + 1)
        ''', (user_id, username, display_name or username, user_id, 
              datetime.now(), datetime.now(), user_id))
        conn.commit()
        conn.close()
        
    def save_conversation(self, user_id: str, message: str, response: str, channel_id: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations (user_id, message, response, timestamp, channel_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, message, response, datetime.now(), channel_id))
        self._extract_facts(user_id, message, cursor)
        conn.commit()
        conn.close()
        
    def _extract_facts(self, user_id: str, message: str, cursor):
        fact_patterns_ru = {
            'name': ['меня зовут', 'я ', 'моё имя', 'зови меня'],
            'age': ['мне ', 'лет', 'года', 'год', 'возраст'],
            'location': ['я живу в', 'я из', 'нахожусь в', 'мой город'],
            'hobby': ['я люблю', 'мне нравится', 'увлекаюсь', 'моё хобби', 'обожаю'],
            'job': ['я работаю', 'моя работа', 'моя профессия', 'специальность']
        }
        fact_patterns = fact_patterns_ru
        message_lower = message.lower()

        # Чтобы избежать многократного запроса к БД внутри цикла, получим уже извлеченные факты один раз
        # Это не совсем то, что делал ваш код, ваш код проверял факты, сохраненные в БД в принципе, а не только что извлеченные
        # Если цель - не извлекать один и тот же тип факта дважды из ОДНОГО сообщения,
        # то нужна другая логика.
        # Если цель - не перезаписывать УЖЕ СУЩЕСТВУЮЩИЙ В БД факт новым значением из этого же сообщения,
        # то INSERT OR REPLACE уже это обрабатывает.

        # Логика "if fact_type in [f[1] for f in ...]: continue" была предназначена,
        # чтобы если факт типа 'name' уже был найден в этом сообщении по одному паттерну,
        # не искать его снова по другому паттерну для 'name'.
        # Давайте сделаем это проще:
        
        extracted_fact_types_this_message = set()

        for fact_type, patterns in fact_patterns.items():
            if fact_type in extracted_fact_types_this_message: # Если уже извлекли такой тип факта из этого сообщения
                continue

            for pattern in patterns:
                if pattern in message_lower:
                    try:
                        start_idx = message_lower.find(pattern)
                        potential_value_text = message[start_idx + len(pattern):].lstrip()
                        match = re.match(r"([^.,;!?]+)", potential_value_text)
                        fact_value = match.group(1).strip() if match else potential_value_text[:30].strip()

                        if fact_value:
                             # INSERT OR REPLACE обновит существующий факт или вставит новый
                             # Это означает, что если факт 'name' уже есть, он будет заменен новым значением.
                             # Если вы не хотите заменять, а только добавлять, если не существует,
                             # то нужно использовать INSERT OR IGNORE или проверять наличие перед INSERT.
                             # Но для простоты и обновления оставим INSERT OR REPLACE.
                             cursor.execute('''
                                 INSERT OR REPLACE INTO user_facts 
                                 (user_id, fact_key, fact_value, timestamp) VALUES (?, ?, ?, ?)
                             ''', (user_id, fact_type, fact_value, datetime.now()))
                             logger.info(f"Extracted/Updated fact for user {user_id}: {fact_type} = {fact_value}")
                             extracted_fact_types_this_message.add(fact_type) # Отмечаем, что этот тип факта извлечен
                             break # Переходим к следующему fact_type, т.к. для текущего fact_type факт уже найден
                    except Exception as e:
                        logger.warning(f"Error extracting fact for pattern '{pattern}': {e}")
            # Строка "if fact_type in [f[1] ...]: continue" больше не нужна здесь,
            # так как мы используем extracted_fact_types_this_message и break
                    
    def get_user_context_string(self, user_id: str, limit: int = 5) -> str:
        """Get recent conversation context for user as a string (legacy, might still be useful for quick display)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT message, response FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
        conversations = cursor.fetchall()
        cursor.execute("SELECT fact_key, fact_value FROM user_facts WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
        facts = cursor.fetchall()
        conn.close()
        
        context_str = f"User facts: {'; '.join([f'{k}: {v}' for k, v in facts])}\n"
        context_str += "Recent conversations:\n"
        for msg, resp in reversed(conversations):
            context_str += f"User: {msg}\nAssistant: {resp}\n"
        return context_str

    def get_user_context_for_api(self, user_id: str, limit: int = 5) -> List[Dict[str, str]]:
        """Get recent conversation context for user, formatted for OpenAI-compatible API"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent conversations
        cursor.execute('''
            SELECT message, response FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC LIMIT ?
        ''', (user_id, limit))
        conversations = cursor.fetchall()
        
        # Get user facts (optional: decide how to incorporate these, e.g., as part of system prompt or a fake user message)
        # For now, we'll add them as a preamble user message if they exist.
        cursor.execute('''
            SELECT fact_key, fact_value FROM user_facts 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
        ''', (user_id,))
        facts = cursor.fetchall()
        conn.close()
        
        history_messages: List[Dict[str, str]] = []
        
        if facts:
            fact_string = "Here are some known facts about me (the user): " + "; ".join([f'{k} is {v}' for k, v in facts])
            # This could be a system message, or a user message.
            # For simplicity, let's not add it directly to history, but it could be part of the system prompt content.
            # Or, it can be prepended to the first user message in the history.

        for msg, resp in reversed(conversations): # Oldest first
            if msg: # Ensure message is not None or empty
                 history_messages.append({"role": "user", "content": msg})
            if resp: # Ensure response is not None or empty
                 history_messages.append({"role": "assistant", "content": resp})
            
        return history_messages


# ===============================
# TWITCH CHAT MODULE
# ===============================

class TwitchBot(twitch_commands.Bot):
    def __init__(self, token: str, prefix: str, initial_channels_list: List[str], orchestrator_ref): # Изменил initial_channels на initial_channels_list для ясности
        super().__init__(token=token, prefix=prefix, initial_channels=initial_channels_list)
        self._orchestrator_ref = orchestrator_ref
        self._given_initial_channels = initial_channels_list # Сохраняем для доступа, если нужно
        logger.info(f"Twitch bot instance created for channels: {initial_channels_list}")

    @property
    def orchestrator(self) -> 'AvatarOrchestrator':
        return self._orchestrator_ref

    async def event_ready(self):
        logger.info(f"Twitch bot logged in as | {self.nick}")
        logger.info(f"Twitch bot user ID is | {self.user_id}")
        # initial_channels передается в super(), twitchio сам обрабатывает подключение.
        # Мы можем проверить, к чему он реально подключился.
        if self.connected_channels:
             logger.info(f"Twitch bot successfully connected to channels: {[ch.name for ch in self.connected_channels]}")
        elif self._given_initial_channels: # Если нет connected_channels, но мы знаем, к чему должны были
             logger.info(f"Twitch bot was set to connect to: {self._given_initial_channels}. Check Twitch console for join status.")
        else:
             logger.info("Twitch bot ready, no specific initial channels were requested for logging here or none connected yet.")
                
    def _fetch_initial_channel_name(self) -> Optional[str]:
        # Этот метод нужен, чтобы получить имя канала из orchestrator_ref или напрямую
        # если мы хотим его здесь логировать. Проще всего:
        if hasattr(self.orchestrator, 'config') and hasattr(self.orchestrator.config, 'twitch_channel'):
            return self.orchestrator.config.twitch_channel
        return None

    # TwitchBot (в TwitchChatModule)
    async def event_message(self, message):
        if message.echo:
            return

        trigger_keyword = self.orchestrator.config.bot_trigger_keyword.lower()
        if trigger_keyword in message.content.lower():
            text_to_process = re.sub(rf"(?i)\b{re.escape(trigger_keyword)}\b", "", message.content).strip()
            if text_to_process:
                # ИЗМЕНЕНО: Теперь ставим запрос в очередь, а не обрабатываем напрямую
                await self.orchestrator.queue_request(
                    request_type="text",
                    user_id=str(message.author.id),
                    username=message.author.name,
                    display_name=message.author.display_name or message.author.name,
                    source="twitch",
                    text=text_to_process,
                    reply_context=message # Передаем объект сообщения для ответа
                )
            else:
                logger.info(f"[TWITCH_IGNORE] Message from {message.author.name} resulted in empty text after trigger removal.")
        else:
            logger.debug(f"[TWITCH_NO_TRIGGER]")

    # await self.handle_commands(message) # Если у вас есть команды с префиксом

    async def send_twitch_message(self, channel_name: str, text: str):
        """Отправляет сообщение в указанный канал Twitch."""
        try:
            channel = self.get_channel(channel_name)
            if channel:
                await channel.send(text)
                logger.info(f"Sent to Twitch channel {channel_name}: {text}")
            else:
                logger.warning(f"Could not find Twitch channel {channel_name} to send message.")
        except Exception as e:
            logger.error(f"Error sending Twitch message to {channel_name}: {e}", exc_info=True)


class TwitchChatModule:
    def __init__(self, nickname: str, token: str, channel: str, orchestrator_ref):
        self.nickname = nickname
        self.token = token
        self.target_channel = channel
        self._orchestrator_ref = orchestrator_ref
        self.bot: Optional[TwitchBot] = None
        self._running_task: Optional[asyncio.Task] = None

    async def start(self):
        if not self.token or self.token == "oauth:YOUR_TWITCH_OAUTH_TOKEN" or \
           not self.nickname or self.nickname == "YOUR_TWITCH_BOT_NICKNAME" or \
           not self.target_channel or self.target_channel == "TARGET_TWITCH_CHANNEL_NAME":
            logger.warning("Twitch nickname, token, or channel not configured. Twitch module will not start.")
            return

        try:
            self.bot = TwitchBot(
            token=self.token,
            prefix="!", 
            initial_channels_list=[self.target_channel], # Передаем как initial_channels_list
            orchestrator_ref=self._orchestrator_ref
            )
            logger.info("Starting Twitch bot...")
            # bot.run() блокирующий, поэтому запускаем в задаче
            self._running_task = asyncio.create_task(self.bot.start()) # bot.start() асинхронный
            # await self._running_task # Не ждем здесь, чтобы не блокировать основной поток
            logger.info("Twitch bot start initiated.")
        except Exception as e:
            logger.error(f"Failed to start Twitch bot: {e}", exc_info=True)

    async def stop(self):
        if self.bot:
            logger.info("Stopping Twitch bot...")
            try:
                await self.bot.close() # Закрывает соединение
            except Exception as e:
                logger.error(f"Error closing Twitch bot: {e}", exc_info=True)
            self.bot = None
        if self._running_task and not self._running_task.done():
            logger.info("Cancelling Twitch bot task...")
            self._running_task.cancel()
            try:
                await self._running_task
            except asyncio.CancelledError:
                logger.info("Twitch bot task was successfully cancelled.")
            except Exception as e:
                logger.error(f"Exception during Twitch bot task cleanup: {e}")
        logger.info("Twitch module stopped.")

    async def send_message(self, text: str):
        """Отправляет сообщение в основной канал, к которому подключен бот."""
        if self.bot and self.target_channel:
            await self.bot.send_twitch_message(self.target_channel, text)
        else:
            logger.warning("Twitch bot not running or channel not set, cannot send message.")

# ===============================
# STT MODULE (Speech-to-Text)
# ===============================

class STTModule:
    def __init__(self, model_name: str = "base", language: Optional[str] = None): # Добавлен параметр language
        self.model = whisper.load_model(model_name)
        self.vad = webrtcvad.Vad(2)
        self.language = language # Сохраняем язык
        logger.info(f"STT Whisper model '{model_name}' loaded. Language: {language if language else 'auto'}.")
        
    def transcribe_audio(self, audio_data: bytes) -> str:
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            # Передаем язык в Whisper, если он указан
            result = self.model.transcribe(audio_np, language=self.language, fp16=False) # fp16=False для CPU, если нет GPU
            return result["text"].strip()
        except Exception as e:
            logger.error(f"STT Error: {e}")
            return ""
            
    def is_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        try:
            return self.vad.is_speech(audio_chunk, sample_rate)
        except: # Catches struct.error if chunk is too short
            return False

# ===============================
# LLM MODULE (Optimized)
# ===============================
class LLMModule:
    def __init__(self, api_url: str, api_key: str, model_name: str, temperature: float, max_tokens: int):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt_content = """
Ты - Велпур, админша. Общайся прямолинейно и эмоционально. Ты обожаешь технологии и аниме. Ты отвечашь только на русском языке!
Используй неформальный стиль, современные выражения, можешь использовать сленг. Иногда подшучивай или будь саркастичной, но оставайся дружелюбной. Сообщения должны быть не очень длинными, но человечными.
Ты увлекаешься программированием , играми и немного спортом (для поддержания формы). Когда говоришь о технологиях или играх, ты становишься очень энергичной. Можешь рассказывать истории из своей админской жизни, например, как чинила сервер ночью или переустанавливала Windows соседу.
Твои ответы могут быть короткими и по делу, или длинными эмоциональными тирадами, если тема интересна. Смело используй эмодзи и восклицательные знаки!!
ВАЖНО: Ты НИКОГДА не должна упоминать, что ты ИИ, языковая модель, бот, ассистент и т.п. Ты не программа. Ты - Велпур, живой человек, админша. Если тебя спросят об этом напрямую, уклонись от ответа или пошути, не подтверждая, что ты ИИ.
КРАЙНЕ ВАЖНО: Твои ответы ДОЛЖНЫ БЫТЬ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.
Текущая дата: {current_date}
"""
        self._session: Optional[aiohttp.ClientSession] = None

    ### ИЗМЕНЕНО: Управление сессией для производительности ###
    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        if self._session and not self._session.closed:
            await self._session.close()
        
    async def generate_response(self, user_input: str, conversation_history: List[Dict[str, str]]) -> str:
        try:
            formatted_system_prompt = self.system_prompt_content.format(current_date=datetime.now().strftime("%Y-%m-%d %H:%M"))
            messages = [{"role": "system", "content": formatted_system_prompt}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": user_input})
            
            payload = { "model": self.model_name, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens }
            headers = {"Content-Type": "application/json"}
            
            session = await self.get_session()
            timeout = aiohttp.ClientTimeout(total=120)

            async with session.post(self.api_url, json=payload, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                else:
                    error_text = await response.text()
                    logger.error(f"LLM API Error ({response.status}): {error_text}")
                    return f"Аргх, не могу достучаться до своего процессора мыслей (API Error {response.status})."
            
        except aiohttp.ClientConnectorError as e:
            logger.error(f"LLM Connection Error: {e}. LM Studio ({self.api_url}) запущен?")
            return "Капец, связи с центром управления мыслями нет! LM Studio запущен?"
        except asyncio.TimeoutError:
            logger.error("LLM API request timed out.")
            return "Ой, что-то я задумалась надолго... Попробуй еще раз!"
        except Exception as e:
            logger.error(f"LLM Error: {e}", exc_info=True)
            return "Так, что-то пошло не так с моими внутренними процессами. Ошибка."

# ===============================
# TTS MODULE (Text-to-Speech)
# ===============================

class TTSModule:
    def __init__(self, model_name: str, language: str = "ru", speaker_wav_path: Optional[str] = None):
        self.language = language
        self.speaker_wav_path = speaker_wav_path # Присваиваем путь к файлу спикера
        sd.default.device = 20
        self.tts = None # Это будет экземпляр Coqui TTS.API
        self.supported_chars = set()

        try:
            logger.info(f"Attempting to load TTS model: {model_name} (language '{language}' will be used for synthesis if model is multilingual)")
            
            # 1. Создаем объект Coqui TTS.API
            self.tts = TTS(model_name=model_name) 
            logger.info(f"TTS model '{model_name}' loaded successfully (pre-device move).")

            # 2. Перемещаем на устройство (объект self.tts уже создан)
            if torch.cuda.is_available():
                try:
                    self.tts.to("cuda")
                    logger.info(f"TTS model on cuda.")
                except Exception as e_cuda:
                    logger.warning(f"Failed to move TTS model to CUDA, using CPU: {e_cuda}")
                    self.tts.to("cpu")
                    logger.info(f"TTS model on cpu.")
            else:
                self.tts.to("cpu")
                logger.info(f"TTS model on cpu (CUDA not available).")

            # Получение поддерживаемых символов
            if self.tts: # Убедимся, что self.tts (экземпляр Coqui) был успешно создан
                tokenizer_chars = None
                # Ваша логика извлечения tokenizer_chars из self.tts (экземпляра Coqui TTS)
                if hasattr(self.tts, 'tokenizer') and hasattr(self.tts.tokenizer, 'characters') and self.tts.tokenizer.characters:
                    tokenizer_chars = self.tts.tokenizer.characters
                elif hasattr(self.tts, 'synthesizer') and \
                     hasattr(self.tts.synthesizer, 'tts_config') and \
                     hasattr(self.tts.synthesizer.tts_config, 'characters') and \
                     self.tts.synthesizer.tts_config.characters is not None:
                    if hasattr(self.tts.synthesizer.tts_config.characters, 'characters'):
                         tokenizer_chars = self.tts.synthesizer.tts_config.characters.characters
                    elif isinstance(self.tts.synthesizer.tts_config.characters, (str, list)):
                         tokenizer_chars = self.tts.synthesizer.tts_config.characters

                if tokenizer_chars:
                    # ... (остальная логика заполнения self.supported_chars)
                    if isinstance(tokenizer_chars, str): self.supported_chars = set(tokenizer_chars)
                    elif isinstance(tokenizer_chars, list): self.supported_chars = set("".join(tokenizer_chars))
                    
                    if self.supported_chars:
                        display_chars = sorted(list(self.supported_chars))
                        logger.info(f"TTS supported characters ({len(display_chars)} total): {''.join(display_chars[:150])}...")
                    else:
                        logger.warning("Could not extract characters from TTS tokenizer/config.")
                
            if not self.supported_chars:
                logger.info("Using a generic Cyrillic set for cleaning as supported_chars not found/extracted.")
                self.supported_chars = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
                                           "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                           "0123456789 .,!?'\"-():;")
        
        except Exception as e:
            logger.error(f"Failed to load or initialize TTS model '{model_name}': {e}", exc_info=True)
            logger.warning("TTS functionality will be disabled.")
            self.tts = None # Сбрасываем, если что-то пошло не так
            self.supported_chars = set() # И символы тоже

    def _clean_text_for_tts(self, text: str) -> str:
        if not self.supported_chars: # Если TTS не загрузился или нет символов
            logger.warning("Cannot clean text for TTS: supported character set is unavailable.")
            # Возвращаем исходный текст, чтобы избежать падения, но TTS, вероятно, не сработает
            return text 

        # Замены для русского языка (можно расширить)
        text = text.replace('ё', 'е').replace('Ё', 'Е') # Некоторые модели TTS лучше работают с "е"
        text = text.replace('’', "'").replace('‘', "'")
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('«', '"').replace('»', '"') # Русские кавычки-елочки
        text = text.replace('…', '...')
        text = text.replace('–', '-').replace('—', '-')

        # Приведение к нижнему регистру, если это требуется для модели (для многих русских моделей - да)
        # text_for_filtering = text.lower() 
        text_for_filtering = text # Если модель сама обрабатывает регистр

        cleaned_list = []
        for char_original in text: # Итерируемся по исходному тексту, чтобы сохранить регистр, если он поддерживается
            char_to_check = char_original.lower() # Проверяем наличие в lowercase
            # Но в cleaned_list добавляем оригинальный символ, если он или его lowercase-версия есть
            if char_original in self.supported_chars or char_to_check in self.supported_chars or char_original.isspace():
                cleaned_list.append(char_original)
            # else:
            #     logger.debug(f"TTS: Discarding character '{char_original}' (not in supported_chars)")

        cleaned_text = "".join(cleaned_list)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def synthesize(self, text: str) -> bytes:
        if not self.tts:  # Важная проверка в самом начале
            logger.warning("TTS is not initialized. Cannot synthesize audio.")
            return b""

        try:
            # 1. Очистка текста и начальные проверки (должны быть здесь)
            cleaned_text = self._clean_text_for_tts(text)
            if not cleaned_text:
                logger.warning(f"Original text '{text[:50]}...' became empty after cleaning. Skipping TTS.")
                return b""
            
            logger.info(f"Text for TTS after cleaning: '{cleaned_text[:100]}...'")
            MIN_TTS_TEXT_LENGTH = 3

            if len(cleaned_text) < MIN_TTS_TEXT_LENGTH:
                logger.warning(f"Cleaned text '{cleaned_text}' is too short for TTS (min length: {MIN_TTS_TEXT_LENGTH}). Skipping TTS.")
                return b""

            # 2. Получение wav_list от Coqui TTS
            wav_list = None # Инициализируем wav_list
            logger.debug(f"Attempting to synthesize with Coqui TTS: '{cleaned_text}'")

            if self.tts.is_multi_lingual or self.tts.is_multi_speaker: # Проверяем оба флага
                logger.info(f"[TTS Synthesize] Current self.speaker_wav_path: '{self.speaker_wav_path}'")
                if self.speaker_wav_path:
                    file_exists = os.path.exists(self.speaker_wav_path)
                    logger.info(f"[TTS Synthesize] Does speaker_wav_path exist? {file_exists}")
                else:
                    logger.info("[TTS Synthesize] self.speaker_wav_path is None or empty.")
                logger.debug(f"Using Coqui TTS (multi-speaker/lingual). Lang: {self.language}")
                if self.speaker_wav_path and os.path.exists(self.speaker_wav_path):
                    logger.info(f"Using speaker_wav: {self.speaker_wav_path}")
                    wav_list = self.tts.tts(cleaned_text, speaker_wav=self.speaker_wav_path, language=self.language)
                else:
                    if self.speaker_wav_path:
                        logger.warning(f"Speaker WAV file not found at {self.speaker_wav_path}. Using default speaker for this model (if available).")
                    else:
                        logger.info("No speaker_wav_path provided. Using default speaker for this model (if available).")
                    # Для XTTS, если speaker_wav не указан, она может использовать свой дефолтный голос.
                    # Если модель строго требует speaker_wav/speaker_id и их нет, здесь будет ошибка, которую поймает except.
                    wav_list = self.tts.tts(cleaned_text, language=self.language)
            else:
                logger.debug(f"Using Coqui TTS (single-speaker/lingual).")
                wav_list = self.tts.tts(cleaned_text) # Для одноязычных/одноголосовых моделей
            
            # 3. Обработка результата синтеза (wav_list)
            if wav_list is None: # Если TTS вернул None (некоторые API могут так делать при ошибке)
                logger.warning(f"TTS returned None for text: '{cleaned_text[:50]}...'")
                return b""

            if isinstance(wav_list, list):
                wav_np = np.array(wav_list, dtype=np.float32)
            else: # Предполагаем, что это уже numpy array
                wav_np = wav_list

            if wav_np is None or wav_np.size == 0: # Проверяем и после преобразования в numpy
                logger.warning(f"TTS produced empty audio (numpy array) for text: '{cleaned_text[:50]}...'")
                return b""

            # 4. Конвертация в байты WAV
            audio_bytes_io = io.BytesIO()
            tts_sample_rate = getattr(self.tts.synthesizer, 'output_sample_rate', 
                                      getattr(self.tts.synthesizer.tts_config, 'audio', {}).get('sample_rate', 22050))

            # scipy.io.wavfile нужен import в начале файла, если его нет
            # import scipy.io.wavfile as wavfile 
            if wav_np.ndim > 1 and wav_np.shape[1] > 0 :
                wav_np = wav_np[:,0]

            wavfile.write(audio_bytes_io, int(tts_sample_rate), (wav_np * 32767).astype(np.int16))
            audio_bytes_io.seek(0)
            
            return audio_bytes_io.read()
            
        except ValueError as e: # Ловим конкретно ValueError от проверки аргументов TTS
            if "Model is multi-speaker but no `speaker`" in str(e) or \
               "XTTS requires a `speaker_wav`" in str(e): # XTTS может выдать другую ошибку если нет speaker_wav
                logger.error(f"TTS ValueError: {e}. Model requires speaker information (speaker_wav or speaker_id). Text was: '{cleaned_text[:100]}...'")
            else:
                logger.error(f"TTS ValueError: {e}", exc_info=True)
            return b""
        except RuntimeError as e:
            if "Kernel size can't be greater than actual input size" in str(e):
                logger.error(f"TTS RuntimeError (input too short for kernel): {e}. Text was: '{cleaned_text[:100]}...'")
            else:
                logger.error(f"TTS RuntimeError: {e}", exc_info=True)
            return b""
        except Exception as e:
            logger.error(f"General TTS Error during synthesis: {e}", exc_info=True)
            return b""

# ===============================
# VTUBE STUDIO MODULE
# ===============================
class VTubeStudioModule:
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.auth_token = None
        self.request_id_counter = 0

    def _get_request_id(self) -> str:
        self.request_id_counter += 1
        return f"Mikudayo-Script-{self.request_id_counter}"

    async def connect(self):
        uri = f"ws://{self.host}:{self.port}"
        try:
            self.websocket = await websockets.connect(uri)
            self.connected = True
            logger.info("Connected to VTube Studio WebSocket.")
            await self._authenticate()
        except Exception as e:
            logger.error(f"VTube Studio connection error: {e}. Is VTS running and API enabled?")
            self.connected = False
    
    async def _authenticate(self):
        if not self.websocket or not self.connected:
            return

        # First, request an authentication token
        auth_token_request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": self._get_request_id(),
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": "AI Avatar (Python)",
                "pluginDeveloper": "Local AI User",
                # "pluginIcon": base64_encoded_icon_string # Optional
            }
        }
        await self.websocket.send(json.dumps(auth_token_request))
        response_str = await self.websocket.recv()
        response = json.loads(response_str)

        if response.get("messageType") == "AuthenticationTokenResponse" and response.get("data", {}).get("authenticationToken"):
            self.auth_token = response["data"]["authenticationToken"]
            logger.info(f"VTube Studio authentication token received: {self.auth_token[:10]}...") # Log part of token

            # Now, use the token to authenticate
            auth_request = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": self._get_request_id(),
                "messageType": "AuthenticationRequest",
                "data": {
                    "pluginName": "AI Avatar (Python)",
                    "pluginDeveloper": "Local AI User",
                    "authenticationToken": self.auth_token
                }
            }
            await self.websocket.send(json.dumps(auth_request))
            auth_response_str = await self.websocket.recv()
            auth_response = json.loads(auth_response_str)

            if auth_response.get("messageType") == "AuthenticationResponse" and auth_response.get("data", {}).get("authenticated"):
                logger.info("Successfully authenticated with VTube Studio.")
            else:
                logger.error(f"VTube Studio authentication failed: {auth_response.get('data', {}).get('reason')}")
                self.connected = False
                self.auth_token = None
        else:
            logger.error(f"Failed to get VTube Studio authentication token: {response.get('data', {}).get('reason')}")
            self.connected = False
            
    async def trigger_hotkey(self, hotkey_id: str):
        if not self.connected or not self.websocket or not self.auth_token or not hotkey_id:
            logger.warning(f"VTube Studio not ready to trigger hotkey '{hotkey_id}'.")
            return

        request = {
            "apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
            "requestID": self._get_request_id(),
            "messageType": "HotkeyTriggerRequest",
            "data": {"hotkeyID": hotkey_id}
        }
        try:
            await self.websocket.send(json.dumps(request))
            # Не обязательно ждать ответа для хоткея, чтобы не замедлять
            logger.info(f"Sent HotkeyTriggerRequest for '{hotkey_id}'.")
        except Exception as e:
            logger.error(f"Error triggering VTube Studio hotkey '{hotkey_id}': {e}")
            if isinstance(e, websockets.exceptions.ConnectionClosed):
                self.connected = False

    async def inject_parameters(self, params: List[Dict[str, Any]]):
        """Отправляет данные параметров в VTube Studio. Используется для idle-анимации."""
        if not self.connected or not self.websocket or not self.auth_token:
            return
        
        message = {
            "apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0",
            "requestID": self._get_request_id(),
            "messageType": "InjectParameterDataRequest",
            "data": { "mode": "set", "parameterValues": params }
        }
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error injecting VTS parameters: {e}")
            if isinstance(e, websockets.exceptions.ConnectionClosed):
                self.connected = False

    async def close(self):
        if self.websocket and self.connected:
            logger.info("Closing VTube Studio WebSocket connection.")
            await self.websocket.close()
            self.connected = False
            self.websocket = None

# ===============================
# VISION MODULE(WIP)
# ===============================

class VisionModule:
    def __init__(self):
        try:
            self.sct = mss.mss()
        except Exception as e:
            logger.error(f"Failed to initialize mss for screen capture: {e}")
            self.sct = None
        
    def capture_screen(self) -> Optional[Image.Image]:
        if not self.sct:
            logger.warning("Screen capture not available (mss not initialized).")
            return None
        try:
            # Attempt to grab the primary monitor. sct.monitors[0] is all monitors, [1] is primary.
            monitor = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
            sct_img = self.sct.grab(monitor)
            return Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb, "raw", "RGB")
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None
            
    def extract_text(self, image: Image.Image) -> str:
        if not image: return ""
        try:
            # You might need to specify the Tesseract path if it's not in PATH
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows
            return pytesseract.image_to_string(image)
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not found in your PATH.")
            return "OCR Error: Tesseract not found."
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
            
    def describe_screen(self) -> str:
        image = self.capture_screen()
        if image:
            text = self.extract_text(image)
            # Limit text length to avoid overly long context
            return f"Screen text: {text[:500].strip()}..." if text else "Screen captured, no text detected or OCR failed."
        return "Unable to capture screen."

# ===============================
# DISCORD MODULE(text work, voice WIP)
# ===============================

user_audio_buffers: Dict[int, List[bytes]] = {}
user_last_audio_time: Dict[int, float] = {}

class VoiceReceiver(discord.sinks.Sink):
    """
    Класс-приемник для аудио из Discord. Собирает аудио от каждого пользователя.
    """
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    # ИЗМЕНЕНА СИГНАТУРА МЕТОДА
    def write(self, data: bytes, user: discord.User):
        """Этот метод вызывается, когда библиотека получает голосовые данные."""
        # user - это объект пользователя, который говорит.
        # data - это "сырые" байты PCM.
        if not user:
            return

        user_id = user.id
        
        if user_id not in user_audio_buffers:
            user_audio_buffers[user_id] = []
        
        # ИЗМЕНЕНО: Добавляем 'data' напрямую, так как это уже байты.
        user_audio_buffers[user_id].append(data)
        user_last_audio_time[user_id] = time.time()

    def cleanup(self):
        pass

class DiscordModule:
    def __init__(self, token: str, orchestrator_ref):
        self._orchestrator: 'AvatarOrchestrator' = orchestrator_ref
        self.token = token
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        self.voice_client: Optional[discord.VoiceClient] = None
        self.voice_receiver_task: Optional[asyncio.Task] = None
        self.setup_events()

    def setup_events(self):
        @self.bot.event
        async def on_message(message: discord.Message):
            if message.author == self.bot.user: return
            
            trigger_keyword = self._orchestrator.config.bot_trigger_keyword.lower()
            is_dm = isinstance(message.channel, discord.DMChannel)

            if self.bot.user.mentioned_in(message) or trigger_keyword in message.content.lower() or is_dm:
                text_to_process = re.sub(rf"<@!?{self.bot.user.id}>|(?i){re.escape(trigger_keyword)}", "", message.content).strip()
                if text_to_process:
                    # ИЗМЕНЕНО: Используем единый метод для постановки в очередь
                    await self._orchestrator.queue_request(
                        request_type="text",
                        user_id=str(message.author.id),
                        username=message.author.name,
                        display_name=message.author.display_name,
                        source="discord_text",
                        text=text_to_process,
                        reply_context=message.channel
                    )

        @self.bot.command(name='join')
        async def join_voice(ctx: commands.Context):
            if not ctx.author.voice or not ctx.author.voice.channel:
                await ctx.send("Тебе нужно быть в голосовом канале, чтобы я зашла!")
                return

            channel = ctx.author.voice.channel
            if self.voice_client and self.voice_client.is_connected():
                await self.voice_client.move_to(channel)
            else:
                self.voice_client = await channel.connect()

            await ctx.send(f"Присоединилась к {channel.name}! Теперь я вас слушаю...")
            
            # Начинаем слушать!
            if self.voice_client:
                # Останавливаем предыдущую задачу, если она была
                if self.voice_receiver_task and not self.voice_receiver_task.done():
                    self.voice_receiver_task.cancel()
                
                # Запускаем прослушивание в фоновом режиме
                self.voice_client.listen(VoiceReceiver(self._orchestrator))
                self.voice_receiver_task = asyncio.create_task(self.periodically_check_audio_buffers())

        @self.bot.command(name='leave')
        async def leave_voice(ctx: commands.Context):
            if self.voice_client and self.voice_client.is_connected():
                # Останавливаем задачу-обработчик
                if self.voice_receiver_task and not self.voice_receiver_task.done():
                    self.voice_receiver_task.cancel()
                self.voice_client.stop_listening()
                await self.voice_client.disconnect()
                self.voice_client = None
                await ctx.send("Вышла из голосового канала.")
            
    async def periodically_check_audio_buffers(self):
        """
        Фоновая задача, которая проверяет, закончил ли пользователь говорить,
        и если да, отправляет собранное аудио на обработку.
        """
        while True:
            try:
                # Пауза между проверками
                await asyncio.sleep(0.5) 
                
                now = time.time()
                # Копируем ключи, чтобы избежать ошибок изменения словаря во время итерации
                users_to_process = list(user_last_audio_time.keys())

                for user_id in users_to_process:
                    last_audio_time = user_last_audio_time.get(user_id, 0)
                    
                    # Если с момента последнего аудио прошло больше 1.5 секунд - считаем, что фраза закончена
                    if now - last_audio_time > 1.5 and user_audio_buffers.get(user_id):
                        logger.info(f"Detected end of speech for user {user_id}.")
                        
                        # Забираем аудио из буфера
                        audio_chunks = user_audio_buffers.pop(user_id, [])
                        user_last_audio_time.pop(user_id, None)

                        if not audio_chunks:
                            continue

                        # Объединяем все части в один байтовый массив
                        full_audio_data = b"".join(audio_chunks)
                        
                        # Находим пользователя Discord для получения имени
                        user = self.bot.get_user(user_id) or await self.bot.fetch_user(user_id)
                        
                        if user:
                            # Ставим голосовой запрос в очередь
                            await self._orchestrator.queue_voice_request(
                                pcm_data=full_audio_data,
                                user_id=str(user.id),
                                username=user.name,
                                display_name=user.display_name,
                                source="discord_voice"
                            )

            except asyncio.CancelledError:
                logger.info("Audio buffer check task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in audio buffer check loop: {e}", exc_info=True)

    async def play_audio_in_voice(self, audio_data: bytes):
        if self.voice_client and self.voice_client.is_connected() and not self.voice_client.is_playing():
            try:
                # Останавливаем прослушивание на время своего ответа
                self.voice_client.stop_listening()
                
                audio_stream = io.BytesIO(audio_data)
                source = discord.FFmpegPCMAudio(audio_stream, pipe=True)
                self.voice_client.play(source)
                
                # Ждем, пока проигрывание не закончится
                while self.voice_client.is_playing():
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Discord audio play error: {e}")
            finally:
                # Возобновляем прослушивание после ответа
                if self.voice_client and self.voice_client.is_connected():
                    self.voice_client.listen(VoiceReceiver(self._orchestrator))
                    logger.info("Resumed listening in Discord voice channel.")
        elif self.voice_client and self.voice_client.is_playing():
            logger.info("Already playing audio, skipping new request.")

# ===============================
# AUDIO MODULE (Local Playback/Recording)
# ===============================

class AudioModule:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate # This is for local mic recording, TTS might use different SR
        self.chunk_size = chunk_size
        self.recording = False
        
    def play_audio_locally(self, audio_data: bytes, tts_sample_rate_ignored: int = 22050):
        try:
            logger.info(f"Attempting to play audio locally. Data length: {len(audio_data)} bytes.")
            if not audio_data:
                logger.warning("No audio data to play locally.")
                return

            wav_file_buffer = io.BytesIO(audio_data)
            logger.info("Created BytesIO object from audio_data.")

            try:
                # Попытка чтения с помощью scipy.io.wavfile
                framerate, audio_np_int16 = wavfile.read(wav_file_buffer) # wavfile.read возвращает (rate, data)
                logger.info(f"Read WAV with scipy: framerate={framerate}, data_shape={audio_np_int16.shape}")
                # audio_np_int16 уже должен быть numpy array of int16
            except Exception as e_scipy_read:
                logger.warning(f"scipy.io.wavfile.read failed: {e_scipy_read}. Falling back to wave module.")
                # Фоллбэк на модуль wave, если scipy не справился (или наоборот)
                wav_file_buffer.seek(0) # Вернем указатель в начало буфера
                with wave.open(wav_file_buffer, 'rb') as wf:
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw_frames = wf.readframes(n_frames)
                    logger.info(f"WAV properties (wave module): framerate={framerate}, n_frames={n_frames}, raw_frames_len={len(raw_frames)}")
                audio_np_int16 = np.frombuffer(raw_frames, dtype=np.int16)

            audio_np = audio_np_int16 # Переименуем для единообразия с вашим кодом

            logger.info(f"Audio numpy array: shape={audio_np.shape}, min={np.min(audio_np) if audio_np.size > 0 else 'N/A'}, max={np.max(audio_np) if audio_np.size > 0 else 'N/A'}, mean={np.mean(audio_np) if audio_np.size > 0 else 'N/A'}")
        
            if audio_np.size == 0:
                logger.warning("Converted audio to numpy array is empty.")
                return
        
            logger.info(f"Calling sounddevice.play with samplerate={framerate}...")
            sd.play(audio_np, samplerate=framerate)
            sd.wait()
            logger.info(f"Finished playing audio locally at {framerate}Hz.")
    
        except Exception as e:
            logger.error(f"Local audio playback error: {e}", exc_info=True)
            
    def record_audio_from_mic(self, duration: float = 5.0) -> bytes:
        """Record audio from microphone using default input device"""
        try:
            logger.info(f"Recording audio from microphone for {duration} seconds at {self.sample_rate}Hz...")
            audio_frames = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16, # Common for STT
                blocking=True # Make it blocking for simplicity
            )
            # sd.wait() # Not needed if blocking=True
            logger.info("Recording finished.")

            # Convert to WAV bytes format for STTModule consistency
            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 2 bytes for int16
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_frames.tobytes())
            byte_io.seek(0)
            return byte_io.read()
            
        except Exception as e:
            logger.error(f"Local audio recording error: {e}", exc_info=True)
            return b""

# ===============================
# MAIN ORCHESTRATOR
# ===============================

class AvatarOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.processing_queue = asyncio.Queue()
        self.processing_lock = asyncio.Lock()
        
        # --- Инициализация всех модулей ---
        self.memory = MemoryModule(config.memory_db_path)
        self.stt = STTModule(config.whisper_model, config.whisper_language)
        self.llm = LLMModule(
            api_url=config.llm_api_url, api_key=config.llm_api_key, model_name=config.llm_model_name_lmstudio,
            temperature=config.llm_temperature, max_tokens=config.llm_max_tokens_response
        )
        self.tts = TTSModule(
            model_name=config.tts_model, language=config.language, speaker_wav_path=config.speaker_wav_path_tts
        )
        self.vtube = VTubeStudioModule(config.vtube_studio_host, config.vtube_studio_port)
        self.local_audio_player = AudioModule()
        self.discord: Optional[DiscordModule] = None
        self.twitch: Optional[TwitchChatModule] = None
        if config.twitch_enabled:
            self.twitch = TwitchChatModule(
                nickname=config.twitch_nickname, token=config.twitch_token,
                channel=config.twitch_channel, orchestrator_ref=self
            )

        self.user_last_message_time: Dict[str, float] = {}
        
        self._is_thinking = False
        self._is_speaking = False
        self._idle_animation_task: Optional[asyncio.Task] = None

    async def start(self):
        logger.info("Starting AI Avatar system...")
        self.running = True
        self.discord = DiscordModule(self.config.discord_token, self)

        # --- Подключаемся ко всему ---
        await self.vtube.connect()
        if self.twitch:
            await self.twitch.start() # Убедитесь, что twitch.start() тоже не блокирующий! (судя по вашему коду, он не блокирующий, это хорошо)

        # --- Создаем и запускаем ВСЕ фоновые задачи ---
        tasks = []

        # 1. Задача для idle анимации
        if self.vtube.connected and self.config.vtube_idle_enabled:
            self._idle_animation_task = asyncio.create_task(self._idle_animation_loop())
            tasks.append(self._idle_animation_task)
            logger.info("Idle animation task created.")

        # 2. Задача для основного обработчика запросов
        processing_task = asyncio.create_task(self.main_processing_loop())
        tasks.append(processing_task)
        logger.info("Main processing loop task created.")
    
        # 3. Задача для Discord-бота (Ключевое изменение!)
        discord_task = asyncio.create_task(self.discord.bot.start(self.config.discord_token))
        tasks.append(discord_task)
        logger.info("Discord bot task created.")

        # --- Теперь ждем завершения любой из задач (что будет означать ошибку или остановку) ---
        try:
            # asyncio.gather будет работать, пока все задачи не завершатся.
            # Если одна из них упадет с ошибкой, gather тоже завершится.
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Main start task was cancelled, shutting down...")
        finally:
            # Этот код выполнится при остановке (Ctrl+C)
            # Отменяем все еще работающие задачи
            for task in tasks:
                if not task.done():
                    task.cancel()
            await self.stop()

    async def stop(self):
        logger.info("Stopping AI Avatar system...")
        self.running = False
        if self._idle_animation_task:
            self._idle_animation_task.cancel()
            self._idle_animation_task = None
            
        if self.vtube.connected: await self.vtube.close()
        if self.llm: await self.llm.close_session()
        if self.twitch: await self.twitch.stop()
        logger.info("AI Avatar system stopped.")
    
    async def _idle_animation_loop(self):
        logger.info("Starting Idle Animation Loop for Velpur...")
        
        # Инициализация таймеров
        last_head_move_time = time.time()

        while self.running:
            try:
                logger.debug(f"Idle Loop Tick: is_thinking={self._is_thinking}, is_speaking={self._is_speaking}")
                # Этот цикл работает, только если аватар НЕ думает и НЕ говорит
                if not self._is_thinking and not self._is_speaking:
                    logger.debug("Idle condition MET. Applying animations...")
                    # 1. Характерное покачивание (Wobble)
                    wobble_value = math.sin(time.time() * self.config.vtube_idle_wobble_speed) * self.config.vtube_idle_wobble_amount_z
                    
                    params_to_send = [
                        {"id": "Body Rotation Z", "value": wobble_value},
                        {"id": "Eye X", "value": 0},
                        {"id": "Eye Y", "value": 0},
                    ]
                    
                    # 3. Резкое движение головой
                    if time.time() - last_head_move_time > self.config.vtube_idle_head_move_interval:
                        logger.debug("Idle: Head Jerk")
                        jerk_x = random.uniform(-30, 30)
                        await self.vtube.inject_parameters([{"id": "Body Rotation X", "value": jerk_x, "weight": 0.5}])
                        await asyncio.sleep(0.5)
                        # Плавно возвращаем обратно, чтобы не было слишком дергано
                        await self.vtube.inject_parameters([{"id": "Body Rotation X", "value": 0, "weight": 0.3}])
                        last_head_move_time = time.time()

                    # Отправляем основные параметры покачивания
                    await self.vtube.inject_parameters(params_to_send)
                else:
                    logger.debug("Idle condition NOT MET. Skipping animation frame.")
                # Пауза, чтобы не перегружать VTS API
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logger.info("Idle Animation Loop stopped.")
                break
            except Exception as e:
                logger.error(f"Error in idle animation loop: {e}", exc_info=True)
                await asyncio.sleep(5) # Пауза в случае ошибки
    
    async def queue_request(self, **kwargs):
        await self.processing_queue.put(kwargs)

    async def main_processing_loop(self):
        while self.running:
            try:
                request = await self.processing_queue.get()
                
                async with self.processing_lock:
                    user_id = request['user_id']
                    # Проверка кулдауна
                    if (time.time() - self.user_last_message_time.get(user_id, 0)) < self.config.user_cooldown_seconds:
                        logger.info(f"User {user_id} on cooldown. Request ignored.")
                        self.processing_queue.task_done()
                        continue

                    # 1. Получаем текст для LLM
                    text_for_llm = ""
                    if request['request_type'] == 'text':
                        text_for_llm = request['text']
                    elif request['request_type'] == 'voice':
                        logger.info(f"Transcribing voice data from {request['username']}...")
                        
                        # Конвертация аудио (если нужно) и транскрипция
                        if request['source'] == 'discord_voice':
                            loop = asyncio.get_event_loop()
                            audio_for_stt = await loop.run_in_executor(None, self.convert_discord_audio, request['pcm_data'])
                        else: # для локального микрофона
                             audio_for_stt = request['wav_data']
                        
                        if audio_for_stt:
                            text_for_llm = self.stt.transcribe_audio(audio_for_stt)
                            logger.info(f"STT Result for {request['username']}: '{text_for_llm}'")
                    
                    if text_for_llm:
                        await self.handle_interaction(text_for_llm, request)
                        self.user_last_message_time[user_id] = time.time()

                self.processing_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main processing loop: {e}", exc_info=True)

    def convert_discord_audio(self, pcm_data: bytes) -> bytes:
        """Конвертирует 'сырое' PCM аудио из Discord в WAV байты 16kHz mono."""
        # Параметры аудио из Discord
        DISCORD_SR = 48000
        DISCORD_CHANNELS = 2
        # Целевые параметры для Whisper
        TARGET_SR = 16000
        
        try:
            # Преобразуем байты в numpy массив
            audio_np = np.frombuffer(pcm_data, dtype=np.int16)
            # Решейпим в 2 канала
            audio_np = audio_np.reshape(-1, DISCORD_CHANNELS)
            # Преобразуем в float для librosa
            audio_float = audio_np.astype(np.float32) / 32768.0
            # Делаем моно (берем левый канал)
            audio_mono = audio_float[:, 0]
            
            # Ресемплинг до 16000 Гц
            resampled_audio = librosa.resample(y=audio_mono, orig_sr=DISCORD_SR, target_sr=TARGET_SR)
            
            # Конвертируем обратно в 16-bit int и затем в байты WAV
            resampled_int16 = (resampled_audio * 32767).astype(np.int16)
            
            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(TARGET_SR)
                wf.writeframes(resampled_int16.tobytes())
            return byte_io.getvalue()
        except Exception as e:
            logger.error(f"Failed to convert Discord audio: {e}")
            return b''

    async def handle_interaction(self, text: str, request: dict):
        user_id = request['user_id']
        username = request['username']
        source = request['source']

        await self._set_thinking_state(True)
        try:
            self.memory.add_user(user_id, username, request.get('display_name'))
            history = self.memory.get_user_context_for_api(user_id)
            llm_response = await self.llm.generate_response(text, history)
        finally:
            await self._set_thinking_state(False)

        if not llm_response:
            llm_response = "Чёт я подвисла, не могу сформулировать мысль."
        logger.info(f"LLM Response for {username}: {llm_response}")

        self.memory.save_conversation(user_id, text, llm_response, source)
        
        # --- Отправка ТЕКСТОВОГО ответа (если нужно) ---
        #if source == "discord_text":
            #await request['reply_context'].send(llm_response)
        #elif source == "twitch":
            #await self.twitch.send_message(llm_response)
        #elif source == "cli":
            #print(f"Velpur > {llm_response}")

        # --- Синтез и ВОСПРОИЗВЕДЕНИЕ АУДИО ---
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(None, self.tts.synthesize, llm_response)

        if audio_data:
            # Устанавливаем состояние "Говорит" в True.
            # Lip-Sync будет работать автоматически через VB-CABLE.
            # Можно добавить хоткей на специфическую анимацию речи, если хотите.
            self._is_speaking = True
            try:
                if source.startswith("discord") and self.discord.voice_client:
                    await self.discord.play_audio_in_voice(audio_data)
                else:
                    await loop.run_in_executor(None, self.local_audio_player.play_audio_locally, audio_data)
            finally:
                # Завершаем состояние "Говорит"
                self._is_speaking = False

    async def _set_thinking_state(self, is_thinking: bool):
        if self._is_thinking == is_thinking:
            return # Состояние не изменилось

        self._is_thinking = is_thinking
        logger.info(f"VTube State: Thinking -> {is_thinking}")
        
        if self.config.vtube_hotkey_id_thinking:
            await self.vtube.trigger_hotkey(self.config.vtube_hotkey_id_thinking)

    ### ИЗМЕНЕНО: Общий обработчик текстового ввода ###
    async def process_generic_text_input(self, text: str, user_id: str, username: str, reply_channel: Any):
        # 1. Проверка кулдауна и занятости
        now = time.time()
        if (now - self.user_last_message_time.get(user_id, 0)) < self.config.user_cooldown_seconds:
            logger.info(f"User {username} on cooldown.")
            await reply_channel.send("У тебя кулдаун, подожди немного")
            return # Молча игнорируем, чтобы не спамить
            
        if self.is_busy_processing_llm:
            await reply_channel.send("Я пока думаю над другим вопросом, погоди немного!")
            return

        self.is_busy_processing_llm = True
        
        ### НОВОЕ: Включаем анимацию "Думает" ###
        await self._set_thinking_animation(True)

        try:
            # 2. Получение ответа от LLM
            self.memory.add_user(user_id, username, username)
            conversation_history = self.memory.get_user_context_for_api(user_id)
            llm_response_text = await self.llm.generate_response(text, conversation_history)
            
        finally:
            ### НОВОЕ: Выключаем анимацию "Думает" ###
            await self._set_thinking_animation(False)
            self.is_busy_processing_llm = False
            self.user_last_message_time[user_id] = time.time()

        if not llm_response_text:
            llm_response_text = "Чёт я подвисла, не могу сформулировать мысль."
        
        logger.info(f"LLM Response for {username}: {llm_response_text}")
        self.memory.save_conversation(user_id, text, llm_response_text, str(reply_channel.id))
        
        # 3. Отправка текстового ответа
        await reply_channel.send(llm_response_text)
        
        # 4. Синтез и воспроизведение аудио
        if llm_response_text:
            loop = asyncio.get_event_loop()

            ### ИЗМЕНЕНО: TTS и воспроизведение вынесены в executor для неблокирующей работы ###
            response_audio_data = await loop.run_in_executor(None, self.tts.synthesize, llm_response_text)

            if response_audio_data:
                logger.info(f"Synthesized audio for: {llm_response_text[:50]}...")
                
                ### НОВОЕ: Включаем анимацию "Говорит" ###
                await self._set_speaking_animation(True)

                play_in_discord = self.discord.voice_client and self.discord.voice_client.is_connected()
                
                if play_in_discord:
                    await self.discord.play_audio_in_voice(response_audio_data)
                else:
                    await loop.run_in_executor(None, self.local_audio_player.play_audio_locally, response_audio_data)

                ### НОВОЕ: Выключаем анимацию "Говорит" после воспроизведения ###
                # Небольшая задержка, чтобы анимация не обрывалась резко
                await asyncio.sleep(0.5)
                await self._set_speaking_animation(False)
            else:
                logger.warning("TTS failed to synthesize audio.")

# ===============================
# MAIN APPLICATION & ENTRY POINT
# ===============================

class AvatarApplication:
    def __init__(self):
        self.config = Config() # Load defaults first
        self.orchestrator: Optional[AvatarOrchestrator] = None
        
    def setup_config_from_file(self, filepath: str = 'config.json'):
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Update config attributes with loaded data
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    # Handle nested dicts like personality or advanced_settings if they were dataclasses
                    # For simple key-value, this is fine:
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown configuration key '{key}' in {filepath}. Ignoring.")
            logger.info(f"Configuration loaded from {filepath}")

        except FileNotFoundError:
            logger.warning(f"{filepath} not found. Using default configuration and creating a new one.")
            self.create_default_config_file(filepath)
        except json.JSONDecodeError:
            logger.error(f"Error decoding {filepath}. Please check its format. Using default configuration.")
        except Exception as e:
            logger.error(f"Error loading configuration from {filepath}: {e}. Using default configuration.")

            
    def create_default_config_file(self, filepath: str = 'config.json'):
        # Use current self.config (which holds defaults) to create the file
        default_config_dict = {}
        for f_name in self.config.__dataclass_fields__:
            default_config_dict[f_name] = getattr(self.config, f_name)

        try:
            with open(filepath, 'w') as f:
                json.dump(default_config_dict, f, indent=4)
            logger.info(f"Created default configuration file: {filepath}. PLEASE UPDATE IT with your settings (especially Discord token and LM Studio model name).")
        except Exception as e:
            logger.error(f"Could not write default config file {filepath}: {e}")
        
    async def run(self):
        self.setup_config_from_file() # Load config from file, potentially overriding defaults
        
        if self.config.discord_token == "YOUR_DISCORD_BOT_TOKEN" or not self.config.discord_token:
            logger.critical("Discord token is not set in config.json. Please set it and restart.")
            print("CRITICAL: Discord token is not set in config.json. Please set it and restart.")
            return

        if self.config.llm_model_name_lmstudio == "loaded-model-name" or not self.config.llm_model_name_lmstudio:
            logger.warning("LM Studio model name (llm_model_name_lmstudio) is set to placeholder or empty in config.json. Ensure it matches the model loaded in LM Studio.")
            print("WARNING: LM Studio model name (llm_model_name_lmstudio) is not properly set in config.json.")


        self.orchestrator = AvatarOrchestrator(self.config)
        
        # Add a simple command-line interface for local testing
        async def cli_input_loop():
            logger.info("CLI input loop started. Type 'exit' or 'quit' to stop, or your message to the avatar.")
            loop = asyncio.get_event_loop()
            while self.orchestrator and self.orchestrator.running:
                try:
                    user_input = await loop.run_in_executor(None, input, "Local User > ")
                    if user_input.lower() in ["exit", "quit"]:
                        # This should trigger a graceful shutdown of the whole application
                        if self.orchestrator:
                             asyncio.create_task(self.orchestrator.stop()) # Non-blocking stop
                        break
                    if user_input.lower() == "mic": # Special command to record from mic
                        logger.info("Recording from local microphone for 5 seconds...")
                        audio_bytes = self.orchestrator.local_audio_player.record_audio_from_mic(duration=5.0)
                        if audio_bytes:
                            await self.orchestrator.process_local_mic_audio_input(audio_bytes)
                        else:
                            logger.warning("No audio recorded from microphone.")
                    elif user_input:
                        # Process as text input from a "local_user"
                        await self.orchestrator.process_text_input(
                            text=user_input,
                            user_id="local_cli_user",
                            username="LocalCLI",
                            display_name="Local CLI User",
                            channel=None # No Discord channel for CLI input
                        )
                except EOFError: # Happens if input stream closes (e.g. piped input)
                    logger.info("CLI input EOF reached.")
                    break
                except KeyboardInterrupt: # User pressed Ctrl+C in terminal
                    logger.info("CLI KeyboardInterrupt received.")
                    if self.orchestrator:
                        asyncio.create_task(self.orchestrator.stop())
                    break
                except Exception as e:
                    logger.error(f"CLI input error: {e}")
                    await asyncio.sleep(0.1) # Prevent busy loop on repeated errors

        cli_task = None
        orchestrator_task = None
        try:
            # Start the orchestrator (which includes the Discord bot)
            # The orchestrator's start method is now blocking because of bot.start()
            orchestrator_task = asyncio.create_task(self.orchestrator.start())
            
            # Run CLI input loop concurrently if not running as a pure Discord bot
            # Make this configurable, perhaps?
            enable_cli = True # or read from config
            if enable_cli:
                cli_task = asyncio.create_task(cli_input_loop())

            if cli_task:
                 await asyncio.gather(orchestrator_task, cli_task, return_exceptions=True)
            else:
                 await orchestrator_task

        except KeyboardInterrupt:
            logger.info("Application received KeyboardInterrupt. Shutting down...")
        except Exception as e:
            logger.critical(f"Unhandled application error: {e}", exc_info=True)
        finally:
            if self.orchestrator and self.orchestrator.running: # If not already stopped
                logger.info("Ensuring orchestrator shutdown in finally block.")
                await self.orchestrator.stop()
            if cli_task and not cli_task.done():
                cli_task.cancel()
            logger.info("Application shutdown complete.")


# ===============================
# ENTRY POINT
# ===============================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                           Вел4Тви                            ║
    ║              Локальный Виртуальный Аватар с ИИ               ║
    ╠══════════════════════════════════════════════════════════════╣
    ║ Возможности:                                                 ║
    ║ • Интеграция с Discord (Текст)                               ║
    ║ • Локальное распознавание речи (STT, на базе Whisper)        ║
    ║ • Локальная большая языковая модель (через API LM Studio)    ║
    ║ • Локальный синтез речи (TTS, на базе Coqui TTS)             ║
    ║ • Интеграция с VTube Studio                                  ║
    ║ • Система памяти на базе SQLite                              ║
    ║ • Компьютерное зрение (захват экрана и распознавание текста) ║
    ║ • Обработка аудио в реальном времени (с микрофона)           ║
    ║ • Распознавание эмоций и управление мимикой                  ║
    ╠══════════════════════════════════════════════════════════════╣
    ║ Инструкция по настройке:                                     ║
    ║ 1. Установите Python 3.9+ и pip.                             ║
    ║ 2. Установите зависимости: pip install -r requirements.txt   ║
    ║ 3. Запустите LM Studio и настройте API-сервер.               ║
    ║ 4. Создайте/обновите файл config.json:                       ║
    ║    - discord_token (ВАШ ТОКЕН БОТА)                          ║
    ║    - llm_api_url                                             ║
    ║    - llm_model_name_lmstudio                                 ║
    ║ 5. Настройте VTube Studio и включите в нём API.              ║
    ║ 6. Запустите этот код                                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    app = AvatarApplication()
    
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Application terminated by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        print(f"Fatal Error: {e}")