from twitchio.ext import commands as twitch_commands
import logging
import asyncio
import re
from typing import List, Optional
from src.logger import logger

class TwitchBot(twitch_commands.Bot):
    def __init__(self, token: str, prefix: str, initial_channels_list: List[str], orchestrator_ref):
        super().__init__(token=token, prefix=prefix, initial_channels=initial_channels_list)
        self._orchestrator_ref = orchestrator_ref
        self._given_initial_channels = initial_channels_list
        logger.info(f"Twitch bot instance created for channels: {initial_channels_list}")

    @property
    def orchestrator(self):
        return self._orchestrator_ref

    async def event_ready(self):
        logger.info(f"Twitch bot logged in as | {self.nick}")
        logger.info(f"Twitch bot user ID is | {self.user_id}")
        if self.connected_channels:
             logger.info(f"Twitch bot successfully connected to channels: {[ch.name for ch in self.connected_channels]}")
        elif self._given_initial_channels:
             logger.info(f"Twitch bot was set to connect to: {self._given_initial_channels}. Check Twitch console for join status.")
        else:
             logger.info("Twitch bot ready, no specific initial channels were requested for logging here or none connected yet.")

    async def event_message(self, message):
        if message.echo:
            return

        trigger_keyword = self.orchestrator.config.bot_trigger_keyword.lower()
        if trigger_keyword in message.content.lower():
            text_to_process = re.sub(rf"(?i)\b{re.escape(trigger_keyword)}\b", "", message.content).strip()
            if text_to_process:
                await self.orchestrator.queue_request(
                    request_type="text",
                    user_id=str(message.author.id),
                    username=message.author.name,
                    display_name=message.author.display_name or message.author.name,
                    source="twitch",
                    text=text_to_process,
                    reply_context=message
                )
            else:
                logger.info(f"[TWITCH_IGNORE] Message from {message.author.name} resulted in empty text after trigger removal.")
        else:
            logger.debug(f"[TWITCH_NO_TRIGGER]")

    async def send_twitch_message(self, channel_name: str, text: str):
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
            initial_channels_list=[self.target_channel],
            orchestrator_ref=self._orchestrator_ref
            )
            logger.info("Starting Twitch bot...")
            self._running_task = asyncio.create_task(self.bot.start())
            logger.info("Twitch bot start initiated.")
        except Exception as e:
            logger.error(f"Failed to start Twitch bot: {e}", exc_info=True)

    async def stop(self):
        if self.bot:
            logger.info("Stopping Twitch bot...")
            try:
                await self.bot.close()
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
        if self.bot and self.target_channel:
            await self.bot.send_twitch_message(self.target_channel, text)
        else:
            logger.warning("Twitch bot not running or channel not set, cannot send message.")
