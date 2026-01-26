from dataclasses import dataclass
from typing import Optional

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
    tts_device_id: Optional[int] = None  # Added to replace hardcoded device 20

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
    vision_trigger_keywords: tuple = ("что на экране", "what is on screen", "посмотри на экран", "look at screen")

    ### Twitch ###
    twitch_enabled: bool = False
    twitch_nickname: str = "YOUR_TWITCH_BOT_NICKNAME"
    twitch_token: str = "oauth:YOUR_TWITCH_OAUTH_TOKEN"
    twitch_channel: str = "TARGET_TWITCH_CHANNEL_NAME"

    ### Interaction settings ###
    user_cooldown_seconds: int = 60
    bot_trigger_keyword: str = "@Velpur"
