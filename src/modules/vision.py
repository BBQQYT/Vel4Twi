import mss
from PIL import Image
import pytesseract
from typing import Optional
from src.logger import logger

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
            monitor = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
            sct_img = self.sct.grab(monitor)
            return Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb, "raw", "RGB")
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None

    def extract_text(self, image: Image.Image) -> str:
        if not image: return ""
        try:
            # Note: Tesseract must be installed on the system and in PATH
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
            return f"Screen text content: {text[:800].strip()}..." if text else "Screen captured, no text detected or OCR failed."
        return "Unable to capture screen."
