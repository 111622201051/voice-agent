# audio/stt.py
import re
import whisper
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Common romanized Tanglish phonetic markers
TANGLISH_KEYWORDS = {
    "naan", "enaku", "enakku", "unga", "ungala", "ungakitta", "pathi", "paththi",
    "sollunga", "pannunga", "pannalam", "iruku", "irukku", "iruken", "irukeenga",
    "theriyum", "theriyala", "aama", "illa", "illai", "romba", "konjam", "nalla",
    "epdi", "enna", "edhu", "yaaru", "yen", "vaanga", "ponga", "mudiyum", "panna",
    "kooda", "apdi", "ipdi", "oru", "adhan", "idhan", "puriyala", "purinjidhu"
}


class SpeechToText:
    def __init__(self, model_size="small", device="cpu"):
        # If user passed English-only model (e.g. base.en), fallback to multilingual equivalent
        if model_size.endswith(".en"):
            clean_size = model_size.replace(".en", "")
            logger.info(f"Model '{model_size}' is English-only. Switching to multilingual '{clean_size}' for Tamil/Tanglish support.")
            model_size = clean_size

        logger.info(f"Loading multilingual Whisper model ({model_size}) on {device}...")
        try:
            self.model = whisper.load_model(model_size, device=device)
            self.last_detected_lang = "en"
            logger.info("✅ Multilingual Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model '{model_size}': {e}")
            logger.info("Retrying with 'base' model...")
            self.model = whisper.load_model("base", device=device)
            self.last_detected_lang = "en"

    def detect_language_style(self, text: str, whisper_lang: str) -> str:
        """
        Determines if speech is pure Tamil ('ta'), English ('en'), or mixed Tanglish ('tanglish').
        """
        has_tamil_script = bool(re.search(r'[\u0B80-\u0BFF]', text))
        has_latin_script = bool(re.search(r'[a-zA-Z]', text))

        words = set(re.findall(r'\b[a-z]+\b', text.lower()))
        has_tanglish_keywords = bool(words & TANGLISH_KEYWORDS)

        if has_tamil_script and has_latin_script:
            return "tanglish"
        elif has_tamil_script or whisper_lang == "ta":
            if has_tanglish_keywords or has_latin_script:
                return "tanglish"
            return "ta"
        elif has_tanglish_keywords:
            return "tanglish"
        else:
            return "en"

    def transcribe_with_language(self, audio_data: np.ndarray) -> tuple[str, str]:
        """
        Transcribes speech and returns (transcribed_text, detected_language).
        detected_language is one of: 'ta' (Tamil), 'en' (English), 'tanglish' (Mixed).
        """
        if len(audio_data) < 1600:
            return "", "en"
        try:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val

            # Conditioning prompt guides Whisper to properly recognize English, Tamil, and Tanglish
            prompt = "Hello, vanakkam. This is an interview conversation in English, தமிழ் (Tamil), and Tanglish. Shree is the AI assistant."

            result = self.model.transcribe(
                audio_data,
                language=None,  # Auto-detect language per turn
                initial_prompt=prompt,
                condition_on_previous_text=False,
                temperature=0.0,
                fp16=False
            )
            text = result.get("text", "").strip()
            detected_whisper_lang = result.get("language", "en")
            style = self.detect_language_style(text, detected_whisper_lang)
            self.last_detected_lang = style
            logger.info(f"STT: '{text}' [Detected Style: {style} | Whisper Lang: {detected_whisper_lang}]")
            return text, style
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return "", "en"

    def transcribe(self, audio_data: np.ndarray) -> str:
        text, _ = self.transcribe_with_language(audio_data)
        return text