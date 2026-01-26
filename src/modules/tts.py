import torch
import sounddevice as sd
import numpy as np
import io
import re
import os
import scipy.io.wavfile as wavfile
from typing import Optional, Set
from TTS.api import TTS
from src.logger import logger

class TTSModule:
    def __init__(self, model_name: str, language: str = "ru", speaker_wav_path: Optional[str] = None, device_id: Optional[int] = None):
        self.language = language
        self.speaker_wav_path = speaker_wav_path

        # FIX: Use configured device ID or let sounddevice choose default
        if device_id is not None:
            sd.default.device = device_id

        self.tts = None
        self.supported_chars = set()

        try:
            logger.info(f"Attempting to load TTS model: {model_name} (language '{language}' will be used for synthesis if model is multilingual)")

            self.tts = TTS(model_name=model_name)
            logger.info(f"TTS model '{model_name}' loaded successfully (pre-device move).")

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

            if self.tts:
                tokenizer_chars = None
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
            self.tts = None
            self.supported_chars = set()

    def _clean_text_for_tts(self, text: str) -> str:
        if not self.supported_chars:
            logger.warning("Cannot clean text for TTS: supported character set is unavailable.")
            return text

        text = text.replace('ё', 'е').replace('Ё', 'Е')
        text = text.replace('’', "'").replace('‘', "'")
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('«', '"').replace('»', '"')
        text = text.replace('…', '...')
        text = text.replace('–', '-').replace('—', '-')

        cleaned_list = []
        for char_original in text:
            char_to_check = char_original.lower()
            if char_original in self.supported_chars or char_to_check in self.supported_chars or char_original.isspace():
                cleaned_list.append(char_original)

        cleaned_text = "".join(cleaned_list)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def synthesize(self, text: str) -> bytes:
        if not self.tts:
            logger.warning("TTS is not initialized. Cannot synthesize audio.")
            return b""

        try:
            cleaned_text = self._clean_text_for_tts(text)
            if not cleaned_text:
                logger.warning(f"Original text '{text[:50]}...' became empty after cleaning. Skipping TTS.")
                return b""

            logger.info(f"Text for TTS after cleaning: '{cleaned_text[:100]}...'")
            MIN_TTS_TEXT_LENGTH = 3

            if len(cleaned_text) < MIN_TTS_TEXT_LENGTH:
                logger.warning(f"Cleaned text '{cleaned_text}' is too short for TTS (min length: {MIN_TTS_TEXT_LENGTH}). Skipping TTS.")
                return b""

            wav_list = None
            logger.debug(f"Attempting to synthesize with Coqui TTS: '{cleaned_text}'")

            if self.tts.is_multi_lingual or self.tts.is_multi_speaker:
                logger.debug(f"Using Coqui TTS (multi-speaker/lingual). Lang: {self.language}")
                if self.speaker_wav_path and os.path.exists(self.speaker_wav_path):
                    wav_list = self.tts.tts(cleaned_text, speaker_wav=self.speaker_wav_path, language=self.language)
                else:
                    if self.speaker_wav_path:
                        logger.warning(f"Speaker WAV file not found at {self.speaker_wav_path}. Using default speaker.")
                    else:
                        logger.info("No speaker_wav_path provided. Using default speaker.")
                    wav_list = self.tts.tts(cleaned_text, language=self.language)
            else:
                logger.debug(f"Using Coqui TTS (single-speaker/lingual).")
                wav_list = self.tts.tts(cleaned_text)

            if wav_list is None:
                logger.warning(f"TTS returned None for text: '{cleaned_text[:50]}...'")
                return b""

            if isinstance(wav_list, list):
                wav_np = np.array(wav_list, dtype=np.float32)
            else:
                wav_np = wav_list

            if wav_np is None or wav_np.size == 0:
                logger.warning(f"TTS produced empty audio (numpy array) for text: '{cleaned_text[:50]}...'")
                return b""

            audio_bytes_io = io.BytesIO()
            tts_sample_rate = getattr(self.tts.synthesizer, 'output_sample_rate',
                                      getattr(self.tts.synthesizer.tts_config, 'audio', {}).get('sample_rate', 22050))

            if wav_np.ndim > 1 and wav_np.shape[1] > 0 :
                wav_np = wav_np[:,0]

            wavfile.write(audio_bytes_io, int(tts_sample_rate), (wav_np * 32767).astype(np.int16))
            audio_bytes_io.seek(0)

            return audio_bytes_io.read()

        except ValueError as e:
            if "Model is multi-speaker but no `speaker`" in str(e) or \
               "XTTS requires a `speaker_wav`" in str(e):
                logger.error(f"TTS ValueError: {e}. Model requires speaker information.")
            else:
                logger.error(f"TTS ValueError: {e}", exc_info=True)
            return b""
        except RuntimeError as e:
            if "Kernel size can't be greater than actual input size" in str(e):
                logger.error(f"TTS RuntimeError (input too short): {e}.")
            else:
                logger.error(f"TTS RuntimeError: {e}", exc_info=True)
            return b""
        except Exception as e:
            logger.error(f"General TTS Error during synthesis: {e}", exc_info=True)
            return b""
