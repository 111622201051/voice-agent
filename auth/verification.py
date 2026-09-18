# auth/verification.py
from resemblyzer import VoiceEncoder, preprocess_wav
from numpy.linalg import norm
import numpy as np
import pickle
import os
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SpeakerVerifier:
    def __init__(self, threshold=0.60, interrupt_threshold=0.50):
        logger.info("Loading Speaker Verification model...")
        self.encoder = VoiceEncoder()
        self.threshold = threshold
        # Relaxed cut-off for quick barge-in checks, where only ~0.5s of audio
        # is available (short interruptions naturally score lower).
        self.interrupt_threshold = interrupt_threshold
        self.profile_path = os.path.join(BASE_DIR, "data", "voice_profiles", "my_voice.pkl")
        self.embeddings = []  # List of voice fingerprints
        logger.info("✅ Speaker Verification model loaded.")

    def _get_embedding(self, audio):
        """Converts raw audio into a voice fingerprint (embedding)."""
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        wav = preprocess_wav(audio, source_sr=16000)
        if len(wav) < 16000:
            wav = np.pad(wav, (0, 16000 - len(wav)))
        emb = self.encoder.embed_utterance(wav)
        emb_norm = norm(emb)
        if emb_norm == 0:
            return emb
        return emb / emb_norm

    def enroll(self, audio_chunks):
        """Creates MULTIPLE voice profiles from the samples and saves to disk."""
        logger.info(f"Generating {len(audio_chunks)} voice profiles...")
        self.embeddings = []

        for chunk in audio_chunks:
            emb = self._get_embedding(chunk)
            self.embeddings.append(emb)

        # Save to both canonical project path and current working directory
        save_paths = [
            self.profile_path,
            os.path.join(os.getcwd(), "data", "voice_profiles", "my_voice.pkl")
        ]
        for p in set(save_paths):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                pickle.dump(self.embeddings, f)
        logger.info("✅ Multiple voice profiles enrolled and saved!")

    def enroll_embeddings(self, embeddings_list):
        """Saves pre-computed embeddings list directly to disk."""
        self.embeddings = embeddings_list
        save_paths = [
            self.profile_path,
            os.path.join(os.getcwd(), "data", "voice_profiles", "my_voice.pkl")
        ]
        for p in set(save_paths):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                pickle.dump(self.embeddings, f)
        logger.info(f"✅ Saved {len(self.embeddings)} biometric voice profiles!")

    def load_profile(self):
        """Loads existing voice profiles from disk with fallback checking."""
        paths_to_check = [
            self.profile_path,
            os.path.join(os.getcwd(), "data", "voice_profiles", "my_voice.pkl"),
            os.path.join(os.path.dirname(BASE_DIR), "data", "voice_profiles", "my_voice.pkl")
        ]
        for p in paths_to_check:
            if os.path.exists(p):
                try:
                    with open(p, "rb") as f:
                        self.embeddings = pickle.load(f)
                    logger.info(f"✅ Loaded {len(self.embeddings)} existing voice profiles from {p}.")
                    # Sync to canonical project location
                    if p != self.profile_path:
                        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
                        with open(self.profile_path, "wb") as f_out:
                            pickle.dump(self.embeddings, f_out)
                    return True
                except Exception as e:
                    logger.warning(f"Failed loading profile from {p}: {e}")
        return False

    def verify(self, audio, min_threshold: float = None):
        if not self.embeddings:
            return False, 0.0
        if len(audio) < 4800:
            return False, 0.0
        try:
            duration = len(audio) / 16000
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)))

            test_emb = self._get_embedding(audio)
            similarities = [float(np.dot(e, test_emb)) for e in self.embeddings]
            avg_sim = float(np.mean(similarities))
            max_sim = float(np.max(similarities))
            matches = sum(1 for s in similarities if s >= 0.55)

            threshold_target = min_threshold or self.threshold
            is_authorized = (max_sim >= threshold_target) or (avg_sim >= 0.55) or (matches >= 2)

            logger.info(
                f"Verify: duration={duration:.2f}s, max={max_sim:.2f}, avg={avg_sim:.2f}, auth={is_authorized}")
            return is_authorized, max(avg_sim, max_sim)
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False, 0.0