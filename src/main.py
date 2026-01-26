import asyncio
import json
import logging
import sys
from src.config import Config
from src.orchestrator import AvatarOrchestrator
from src.logger import logger

class AvatarApplication:
    def __init__(self):
        self.config = Config()
        self.orchestrator: AvatarOrchestrator = None

    def setup_config_from_file(self, filepath: str = 'config.json'):
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)

            for key, value in config_data.items():
                if hasattr(self.config, key):
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
        self.setup_config_from_file()

        if self.config.discord_token == "YOUR_DISCORD_BOT_TOKEN" or not self.config.discord_token:
            logger.critical("Discord token is not set in config.json. Please set it and restart.")
            print("CRITICAL: Discord token is not set in config.json. Please set it and restart.")
            # We don't return here to allow CLI testing even without Discord
            # return

        if self.config.llm_model_name_lmstudio == "loaded-model-name" or not self.config.llm_model_name_lmstudio:
            logger.warning("LM Studio model name (llm_model_name_lmstudio) is set to placeholder or empty in config.json. Ensure it matches the model loaded in LM Studio.")
            print("WARNING: LM Studio model name (llm_model_name_lmstudio) is not properly set in config.json.")


        self.orchestrator = AvatarOrchestrator(self.config)

        async def cli_input_loop():
            logger.info("CLI input loop started. Type 'exit' or 'quit' to stop, or your message to the avatar.")
            loop = asyncio.get_event_loop()
            while self.orchestrator and self.orchestrator.running:
                try:
                    user_input = await loop.run_in_executor(None, input, "Local User > ")
                    if user_input.lower() in ["exit", "quit"]:
                        if self.orchestrator:
                             asyncio.create_task(self.orchestrator.stop())
                        break
                    if user_input.lower() == "mic":
                        logger.info("Recording from local microphone for 5 seconds...")
                        audio_bytes = self.orchestrator.local_audio_player.record_audio_from_mic(duration=5.0)
                        if audio_bytes:
                            await self.orchestrator.process_local_mic_audio_input(audio_bytes)
                        else:
                            logger.warning("No audio recorded from microphone.")
                    elif user_input:
                        await self.orchestrator.process_text_input(
                            text=user_input,
                            user_id="local_cli_user",
                            username="LocalCLI",
                            display_name="Local CLI User",
                            channel=None
                        )
                except EOFError:
                    logger.info("CLI input EOF reached.")
                    break
                except KeyboardInterrupt:
                    logger.info("CLI KeyboardInterrupt received.")
                    if self.orchestrator:
                        asyncio.create_task(self.orchestrator.stop())
                    break
                except Exception as e:
                    logger.error(f"CLI input error: {e}")
                    await asyncio.sleep(0.1)

        cli_task = None
        orchestrator_task = None
        try:
            orchestrator_task = asyncio.create_task(self.orchestrator.start())

            enable_cli = True
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
            if self.orchestrator and self.orchestrator.running:
                logger.info("Ensuring orchestrator shutdown in finally block.")
                await self.orchestrator.stop()
            if cli_task and not cli_task.done():
                cli_task.cancel()
            logger.info("Application shutdown complete.")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                           Вел4Тви                            ║
    ║              Локальный Виртуальный Аватар с ИИ               ║
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
