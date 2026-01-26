import asyncio
import time
import random
import math
import librosa
import numpy as np
import io
import wave
from typing import Dict, Optional, Any

from src.config import Config
from src.logger import logger
from src.modules.memory import MemoryModule
from src.modules.stt import STTModule
from src.modules.llm import LLMModule
from src.modules.tts import TTSModule
from src.modules.vtube import VTubeStudioModule
from src.modules.audio import AudioModule
from src.modules.vision import VisionModule
from src.modules.discord_bot import DiscordModule
from src.modules.twitch_bot import TwitchChatModule

class AvatarOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.processing_queue = asyncio.Queue()
        self.processing_lock = asyncio.Lock()

        # --- Init Modules ---
        self.memory = MemoryModule(config.memory_db_path)
        self.stt = STTModule(config.whisper_model, config.whisper_language)
        self.llm = LLMModule(
            api_url=config.llm_api_url, api_key=config.llm_api_key, model_name=config.llm_model_name_lmstudio,
            temperature=config.llm_temperature, max_tokens=config.llm_max_tokens_response
        )
        self.tts = TTSModule(
            model_name=config.tts_model, language=config.language,
            speaker_wav_path=config.speaker_wav_path_tts, device_id=config.tts_device_id
        )
        self.vtube = VTubeStudioModule(config.vtube_studio_host, config.vtube_studio_port)
        self.local_audio_player = AudioModule(sample_rate=config.sample_rate)

        self.vision = None
        if config.enable_vision:
            self.vision = VisionModule()

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

        await self.vtube.connect()
        if self.twitch:
            await self.twitch.start()

        tasks = []

        if self.vtube.connected and self.config.vtube_idle_enabled:
            self._idle_animation_task = asyncio.create_task(self._idle_animation_loop())
            tasks.append(self._idle_animation_task)
            logger.info("Idle animation task created.")

        processing_task = asyncio.create_task(self.main_processing_loop())
        tasks.append(processing_task)
        logger.info("Main processing loop task created.")

        discord_task = asyncio.create_task(self.discord.bot.start(self.config.discord_token))
        tasks.append(discord_task)
        logger.info("Discord bot task created.")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Main start task was cancelled, shutting down...")
        finally:
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
        last_head_move_time = time.time()

        while self.running:
            try:
                if not self._is_thinking and not self._is_speaking:
                    wobble_value = math.sin(time.time() * self.config.vtube_idle_wobble_speed) * self.config.vtube_idle_wobble_amount_z

                    params_to_send = [
                        {"id": "Body Rotation Z", "value": wobble_value},
                        {"id": "Eye X", "value": 0},
                        {"id": "Eye Y", "value": 0},
                    ]

                    if time.time() - last_head_move_time > self.config.vtube_idle_head_move_interval:
                        jerk_x = random.uniform(-30, 30)
                        await self.vtube.inject_parameters([{"id": "Body Rotation X", "value": jerk_x, "weight": 0.5}])
                        await asyncio.sleep(0.5)
                        await self.vtube.inject_parameters([{"id": "Body Rotation X", "value": 0, "weight": 0.3}])
                        last_head_move_time = time.time()

                    await self.vtube.inject_parameters(params_to_send)
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in idle animation loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def queue_request(self, **kwargs):
        await self.processing_queue.put(kwargs)

    # Alias for VoiceReceiver to use
    async def queue_voice_request(self, **kwargs):
        kwargs['request_type'] = 'voice'
        await self.queue_request(**kwargs)

    # --- Methods for CLI Integration (Fixed) ---
    async def process_text_input(self, text: str, user_id: str, username: str, display_name: str, channel: Any):
        """Processes text input from CLI or other direct sources."""
        await self.queue_request(
            request_type="text",
            user_id=user_id,
            username=username,
            display_name=display_name,
            source="cli",
            text=text,
            reply_context=channel # Can be None for CLI
        )

    async def process_local_mic_audio_input(self, audio_bytes: bytes):
        """Processes audio input from local microphone via CLI."""
        await self.queue_request(
            request_type="voice",
            user_id="local_user",
            username="LocalUser",
            display_name="Local User",
            source="local_mic",
            wav_data=audio_bytes
        )
    # -------------------------------------------

    async def main_processing_loop(self):
        while self.running:
            try:
                request = await self.processing_queue.get()

                async with self.processing_lock:
                    user_id = request['user_id']
                    if (time.time() - self.user_last_message_time.get(user_id, 0)) < self.config.user_cooldown_seconds:
                        logger.info(f"User {user_id} on cooldown. Request ignored.")
                        self.processing_queue.task_done()
                        continue

                    text_for_llm = ""
                    system_prompt_extras = ""

                    if request['request_type'] == 'text':
                        text_for_llm = request['text']
                    elif request['request_type'] == 'voice':
                        logger.info(f"Transcribing voice data from {request['username']}...")

                        audio_for_stt = None
                        if request['source'] == 'discord_voice':
                            loop = asyncio.get_event_loop()
                            audio_for_stt = await loop.run_in_executor(None, self.convert_discord_audio, request['pcm_data'])
                        elif request['source'] == 'local_mic':
                             audio_for_stt = request['wav_data']

                        if audio_for_stt:
                            text_for_llm = self.stt.transcribe_audio(audio_for_stt)
                            logger.info(f"STT Result for {request['username']}: '{text_for_llm}'")

                    if text_for_llm:
                        # --- Vision Logic Integration ---
                        if self.vision and any(keyword in text_for_llm.lower() for keyword in self.config.vision_trigger_keywords):
                            logger.info("Vision trigger detected in text. Capturing screen...")
                            vision_desc = self.vision.describe_screen()
                            logger.info(f"Vision result: {vision_desc}")
                            system_prompt_extras += f"\n[VISION DATA]: User asked to look at screen. {vision_desc}"
                        # --------------------------------

                        await self.handle_interaction(text_for_llm, request, system_prompt_extras)
                        self.user_last_message_time[user_id] = time.time()

                self.processing_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main processing loop: {e}", exc_info=True)

    def convert_discord_audio(self, pcm_data: bytes) -> bytes:
        DISCORD_SR = 48000
        DISCORD_CHANNELS = 2
        TARGET_SR = 16000

        try:
            audio_np = np.frombuffer(pcm_data, dtype=np.int16)
            audio_np = audio_np.reshape(-1, DISCORD_CHANNELS)
            audio_float = audio_np.astype(np.float32) / 32768.0
            audio_mono = audio_float[:, 0]

            resampled_audio = librosa.resample(y=audio_mono, orig_sr=DISCORD_SR, target_sr=TARGET_SR)

            resampled_int16 = (resampled_audio * 32767).astype(np.int16)

            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TARGET_SR)
                wf.writeframes(resampled_int16.tobytes())
            return byte_io.getvalue()
        except Exception as e:
            logger.error(f"Failed to convert Discord audio: {e}")
            return b''

    async def handle_interaction(self, text: str, request: dict, system_prompt_extras: str = ""):
        user_id = request['user_id']
        username = request['username']
        source = request['source']

        await self._set_thinking_state(True)
        try:
            self.memory.add_user(user_id, username, request.get('display_name'))
            history = self.memory.get_user_context_for_api(user_id)
            llm_response = await self.llm.generate_response(text, history, system_prompt_extras)
        finally:
            await self._set_thinking_state(False)

        if not llm_response:
            llm_response = "Чёт я подвисла, не могу сформулировать мысль."
        logger.info(f"LLM Response for {username}: {llm_response}")

        self.memory.save_conversation(user_id, text, llm_response, source)

        # --- Send Text Response ---
        if source == "discord_text":
            try:
                await request['reply_context'].send(llm_response)
            except Exception as e:
                logger.error(f"Failed to send Discord text response: {e}")
        elif source == "twitch":
            await self.twitch.send_message(llm_response)
        elif source == "cli":
            print(f"Velpur > {llm_response}")
        # For voice sources, we might want to text back too if possible?
        # For now, voice usually expects voice back.

        # --- TTS & Playback ---
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(None, self.tts.synthesize, llm_response)

        if audio_data:
            self._is_speaking = True
            try:
                if source.startswith("discord") and self.discord.voice_client:
                    await self.discord.play_audio_in_voice(audio_data)
                else:
                    await loop.run_in_executor(None, self.local_audio_player.play_audio_locally, audio_data)
            finally:
                self._is_speaking = False

    async def _set_thinking_state(self, is_thinking: bool):
        if self._is_thinking == is_thinking:
            return

        self._is_thinking = is_thinking
        logger.info(f"VTube State: Thinking -> {is_thinking}")

        if self.config.vtube_hotkey_id_thinking:
            await self.vtube.trigger_hotkey(self.config.vtube_hotkey_id_thinking)
