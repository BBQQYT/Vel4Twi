import whisper
import webrtcvad
import numpy as np
from typing import Optional
from src.logger import logger

class STTModule:
    def __init__(self, model_name: str = "base", language: Optional[str] = None):
        self.model = whisper.load_model(model_name)
        self.vad = webrtcvad.Vad(2)
        self.language = language
        logger.info(f"STT Whisper model '{model_name}' loaded. Language: {language if language else 'auto'}.")

    def transcribe_audio(self, audio_data: bytes) -> str:
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_np, language=self.language, fp16=False)
            return result["text"].strip()
        except Exception as e:
            logger.error(f"STT Error: {e}")
            return ""

    def is_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        try:
            return self.vad.is_speech(audio_chunk, sample_rate)
        except:
            return False
