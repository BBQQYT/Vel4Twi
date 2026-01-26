import sounddevice as sd
import numpy as np
import io
import wave
import scipy.io.wavfile as wavfile
from src.logger import logger

class AudioModule:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
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
                framerate, audio_np_int16 = wavfile.read(wav_file_buffer)
                logger.info(f"Read WAV with scipy: framerate={framerate}, data_shape={audio_np_int16.shape}")
            except Exception as e_scipy_read:
                logger.warning(f"scipy.io.wavfile.read failed: {e_scipy_read}. Falling back to wave module.")
                wav_file_buffer.seek(0)
                with wave.open(wav_file_buffer, 'rb') as wf:
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw_frames = wf.readframes(n_frames)
                    logger.info(f"WAV properties (wave module): framerate={framerate}, n_frames={n_frames}, raw_frames_len={len(raw_frames)}")
                audio_np_int16 = np.frombuffer(raw_frames, dtype=np.int16)

            audio_np = audio_np_int16

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
                dtype=np.int16,
                blocking=True
            )
            logger.info("Recording finished.")

            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_frames.tobytes())
            byte_io.seek(0)
            return byte_io.read()

        except Exception as e:
            logger.error(f"Local audio recording error: {e}", exc_info=True)
            return b""
