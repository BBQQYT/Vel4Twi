import discord
from discord.ext import commands
import asyncio
import time
import re
import io
import logging
from typing import Optional, Dict, List, Any
from src.logger import logger

# Global or class-level storage for audio buffers (kept simple as in original)
user_audio_buffers: Dict[int, List[bytes]] = {}
user_last_audio_time: Dict[int, float] = {}

class VoiceReceiver(discord.sinks.Sink):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator

    def write(self, user: discord.User, data: bytes):
        if not user:
            return

        user_id = user.id
        if user_id not in user_audio_buffers:
            user_audio_buffers[user_id] = []

        user_audio_buffers[user_id].append(data)
        user_last_audio_time[user_id] = time.time()

    def cleanup(self):
        pass

class DiscordModule:
    def __init__(self, token: str, orchestrator_ref):
        self._orchestrator = orchestrator_ref
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
        async def on_ready():
            logger.info(f"Discord Bot logged in as {self.bot.user}")

        @self.bot.event
        async def on_message(message: discord.Message):
            if message.author == self.bot.user: return

            trigger_keyword = self._orchestrator.config.bot_trigger_keyword.lower()
            is_dm = isinstance(message.channel, discord.DMChannel)

            if self.bot.user.mentioned_in(message) or trigger_keyword in message.content.lower() or is_dm:
                text_to_process = re.sub(rf"<@!?{self.bot.user.id}>|(?i){re.escape(trigger_keyword)}", "", message.content).strip()
                if text_to_process:
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

            if self.voice_client:
                if self.voice_receiver_task and not self.voice_receiver_task.done():
                    self.voice_receiver_task.cancel()

                self.voice_client.listen(VoiceReceiver(self._orchestrator))
                self.voice_receiver_task = asyncio.create_task(self.periodically_check_audio_buffers())

        @self.bot.command(name='leave')
        async def leave_voice(ctx: commands.Context):
            if self.voice_client and self.voice_client.is_connected():
                if self.voice_receiver_task and not self.voice_receiver_task.done():
                    self.voice_receiver_task.cancel()
                self.voice_client.stop_listening()
                await self.voice_client.disconnect()
                self.voice_client = None
                await ctx.send("Вышла из голосового канала.")

    async def periodically_check_audio_buffers(self):
        while True:
            try:
                await asyncio.sleep(0.5)
                now = time.time()
                users_to_process = list(user_last_audio_time.keys())

                for user_id in users_to_process:
                    last_audio_time = user_last_audio_time.get(user_id, 0)

                    if now - last_audio_time > 1.5 and user_audio_buffers.get(user_id):
                        logger.info(f"Detected end of speech for user {user_id}.")

                        audio_chunks = user_audio_buffers.pop(user_id, [])
                        user_last_audio_time.pop(user_id, None)

                        if not audio_chunks:
                            continue

                        full_audio_data = b"".join(audio_chunks)

                        user = self.bot.get_user(user_id) or await self.bot.fetch_user(user_id)

                        if user:
                            await self._orchestrator.queue_voice_request(
                                pcm_data=full_audio_data,
                                user_id=str(user.id),
                                username=user.name,
                                display_name=user.display_name,
                                source="discord_voice",
                                reply_context=user.dm_channel or await user.create_dm() # Send response to DM or find a way to reply in voice
                            )

            except asyncio.CancelledError:
                logger.info("Audio buffer check task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in audio buffer check loop: {e}", exc_info=True)

    async def play_audio_in_voice(self, audio_data: bytes):
        if self.voice_client and self.voice_client.is_connected() and not self.voice_client.is_playing():
            try:
                self.voice_client.stop_listening()

                audio_stream = io.BytesIO(audio_data)
                source = discord.FFmpegPCMAudio(audio_stream, pipe=True)
                self.voice_client.play(source)

                while self.voice_client.is_playing():
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Discord audio play error: {e}")
            finally:
                if self.voice_client and self.voice_client.is_connected():
                    self.voice_client.listen(VoiceReceiver(self._orchestrator))
                    logger.info("Resumed listening in Discord voice channel.")
        elif self.voice_client and self.voice_client.is_playing():
            logger.info("Already playing audio, skipping new request.")
