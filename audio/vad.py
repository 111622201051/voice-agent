# audio/vad.py
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    def __init__(self, threshold=0.4):
        logger.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        self.model = model
        self.threshold = threshold
        logger.info("✅ Silero VAD loaded successfully.")

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000, threshold: float = None) -> bool:
        # Use custom threshold if provided (for echo suppression), else use default
        current_threshold = threshold if threshold is not None else self.threshold

        # Silero strictly requires exactly 512 samples at 16kHz
        if len(audio_chunk) < 512:
            audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
        elif len(audio_chunk) > 512:
            audio_chunk = audio_chunk[:512]

        tensor = torch.from_numpy(audio_chunk).float()
        with torch.no_grad():
            speech_prob = self.model(tensor, sample_rate).item()

        return speech_prob > current_threshold