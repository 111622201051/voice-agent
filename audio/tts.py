# audio/tts.py
import os
import json
import time
import queue
import logging
import asyncio
import threading
import pygame
import edge_tts
import soundfile as sf
import numpy as np

try:
    from fishaudio import FishAudio
    from fishaudio.types import ReferenceAudio
    FISH_AUDIO_AVAILABLE = True
except ImportError:
    FISH_AUDIO_AVAILABLE = False

import sys
import torch
import torchaudio

# Safe soundfile-based audio loader to completely bypass Windows torchcodec/DLL issues
def _safe_torchaudio_load(filepath, *args, **kwargs):
    data, sr = sf.read(filepath, dtype='float32')
    if data.ndim == 1:
        tensor = torch.from_numpy(data).unsqueeze(0)
    else:
        tensor = torch.from_numpy(data.T)
    return tensor, sr

torchaudio.load = _safe_torchaudio_load
sys.modules['torchcodec'] = None

try:
    from f5_tts.api import F5TTS
    F5_TTS_AVAILABLE = True
except ImportError:
    F5_TTS_AVAILABLE = False

logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(PROJECT_DIR, "data", "recordings")
VOICE_PROFILES_DIR = os.path.join(PROJECT_DIR, "data", "voice_profiles")
CLONE_AUDIO_PATH = os.path.join(VOICE_PROFILES_DIR, "clone_reference.wav")
CLONE_META_PATH = os.path.join(VOICE_PROFILES_DIR, "clone_reference.json")
os.makedirs(RECORDINGS_DIR, exist_ok=True)


class TextToSpeech:
    def __init__(
        self,
        engine="local",
        api_key=None,
        base_url="https://api.fish.audio",
        default_voice="en-US-GuyNeural"
    ):
        self.engine = engine
        self.default_voice = default_voice
        self.temp_file = os.path.join(RECORDINGS_DIR, "temp_response.mp3")
        self.fish_client = None
        self.local_f5_model = None
        self.clone_ref_bytes = None
        self.clone_prompt_text = ""
        self.enable_voice_clone = True
        self.clone_from_verified_speech = True
        self.require_gpu_for_clone = True
        self._cpu_clone_notice_shown = False

        # Load configuration from config.yaml if available
        config_path = os.path.join(PROJECT_DIR, "config.yaml")
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                    tts_cfg = cfg.get("tts", {})
                    self.enable_voice_clone = bool(tts_cfg.get("enable_voice_clone", True))
                    self.clone_from_verified_speech = bool(tts_cfg.get("clone_from_verified_speech", True))
                    self.require_gpu_for_clone = bool(tts_cfg.get("require_gpu_for_clone", True))
                    if not api_key:
                        api_key = tts_cfg.get("fish_api_key", "")
                    if base_url == "https://api.fish.audio":
                        base_url = tts_cfg.get("fish_base_url", base_url)
            except Exception:
                try:
                    import re
                    with open(config_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        key_match = re.search(r"fish_api_key:\s*[\"']?([^\"'\s\r\n]+)", content)
                        if key_match and not api_key:
                            api_key = key_match.group(1).strip()
                except Exception:
                    pass

        self.api_key = api_key or os.environ.get("FISH_API_KEY", "")
        self.base_url = base_url or os.environ.get("FISH_BASE_URL", "https://api.fish.audio")

        # Multilingual voice settings
        self.tamil_voice = "ta-IN-PallaviNeural"
        self.english_voice = "en-IN-NeerjaExpressiveNeural"
        self.tanglish_voice = "en-IN-NeerjaExpressiveNeural"

        if FISH_AUDIO_AVAILABLE and (self.api_key or "127.0.0.1" in self.base_url or "localhost" in self.base_url):
            try:
                self.fish_client = FishAudio(api_key=self.api_key if self.api_key else "local", base_url=self.base_url)
                logger.info(f"✅ Fish Audio TTS initialized ({self.base_url}).")
            except Exception as e:
                logger.warning(f"Fish Audio init note: {e}")

        # Load reference voice for cloning if available
        self.reload_clone_reference()

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
        except Exception as e:
            logger.warning(f"Pygame mixer init warning: {e}")

    def select_voice(self, text: str, language: str = None) -> str:
        """
        Dynamically selects the appropriate neural voice based on language and script:
        - Tamil script or 'ta' -> ta-IN-PallaviNeural
        - Tanglish / Indian English -> en-IN-NeerjaExpressiveNeural
        - Standard English -> en-IN-NeerjaExpressiveNeural
        """
        has_tamil_script = bool(re.search(r'[\u0B80-\u0BFF]', text))
        if has_tamil_script or language == "ta":
            return self.tamil_voice
        elif language == "tanglish":
            return self.tanglish_voice
        else:
            return self.english_voice

    def stop_playback(self):
        """Immediately stops audio playback and unloads the mixer buffer."""
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
        except Exception as e:
            logger.debug(f"Mixer stop notice: {e}")


    def reload_clone_reference(self):
        """Loads or reloads the recorded voice clone reference sample from disk."""
        if os.path.exists(CLONE_AUDIO_PATH):
            try:
                with open(CLONE_AUDIO_PATH, "rb") as f:
                    self.clone_ref_bytes = f.read()

                if os.path.exists(CLONE_META_PATH):
                    with open(CLONE_META_PATH, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.clone_prompt_text = meta.get("prompt_text", "")
                
                logger.info(f"✅ Loaded voice clone reference ({len(self.clone_ref_bytes)} bytes) for TTS.")
                return True
            except Exception as e:
                logger.warning(f"Could not load clone reference: {e}")
        return False

    def _generate_local_f5(self, text: str) -> bool:
        """Generates 100% offline zero-shot cloned voice using local F5-TTS model.

        Refuses to run on CPU: F5-TTS diffusion synthesis takes ~10 minutes per
        reply without a GPU, which stalls the whole agent. Falls back to the
        fast Edge-TTS / Fish Audio path instead.
        """
        if not F5_TTS_AVAILABLE or not os.path.exists(CLONE_AUDIO_PATH):
            return False
        if self.require_gpu_for_clone and not torch.cuda.is_available():
            if not self._cpu_clone_notice_shown:
                self._cpu_clone_notice_shown = True
                print("ℹ️  Running on CPU - replies use the fast neural voice.")
                print("   (Want replies in YOUR cloned voice? Add a Fish Audio API")
                print("    key in config.yaml, or use a CUDA GPU. Not an error!)")
            return False

        try:
            if not self.local_f5_model:
                print("⏳ [TTS] Loading Local High-Fidelity Voice Cloning Engine (F5-TTS)...")
                self.local_f5_model = F5TTS()

            out_wav = os.path.join(RECORDINGS_DIR, "temp_response.wav")
            self.local_f5_model.infer(
                ref_file=CLONE_AUDIO_PATH,
                ref_text=self.clone_prompt_text or "Hello! I am your AI assistant.",
                gen_text=text,
                file_wave=out_wav
            )
            self.temp_file = out_wav
            return True
        except Exception as e:
            logger.error(f"Local F5-TTS voice cloning error: {e}")
            return False

    def _generate_fish_audio(self, text: str) -> bool:
        """Generates cloned speech using Fish Audio SDK."""
        if not self.fish_client:
            api_key = self.api_key or os.environ.get("FISH_API_KEY", "")
            base_url = self.base_url or os.environ.get("FISH_BASE_URL", "https://api.fish.audio")
            if api_key or "127.0.0.1" in base_url:
                try:
                    self.fish_client = FishAudio(api_key=api_key or "local", base_url=base_url)
                except Exception as e:
                    logger.warning(f"Fish client init failed: {e}")
                    return False

        if not self.fish_client or not self.clone_ref_bytes:
            return False

        try:
            ref = ReferenceAudio(
                audio=self.clone_ref_bytes,
                text=self.clone_prompt_text or "Hello! I am your AI assistant."
            )
            audio_bytes = self.fish_client.tts.convert(
                text=text,
                references=[ref],
                format="mp3",
                latency="balanced"
            )
            self.temp_file = os.path.join(RECORDINGS_DIR, "temp_response.mp3")
            with open(self.temp_file, "wb") as f:
                f.write(audio_bytes)
            return True
        except Exception as e:
            logger.warning(f"Fish Audio API not available or no credits: {e}")
            return False

    async def _generate_edge_fallback(self, text: str, voice: str = None, language: str = None):
        """Fallback TTS using Edge-TTS with language-aware voice selection."""
        selected_voice = voice or self.select_voice(text, language)
        self.temp_file = os.path.join(RECORDINGS_DIR, "temp_response.mp3")
        communicate = edge_tts.Communicate(text, selected_voice)
        await communicate.save(self.temp_file)

    def generate_audio_bytes(self, text: str, voice: str = None, language: str = None) -> bytes:
        """Generates audio bytes in memory for browser/UI bridge playback."""
        if not text:
            return b""
        if self.enable_voice_clone and (self._generate_fish_audio(text) or self._generate_local_f5(text)):
            try:
                with open(self.temp_file, "rb") as f:
                    return f.read()
            except Exception:
                pass
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.generate_audio_bytes_async(text, voice=voice, language=language)).result()
        else:
            return asyncio.run(self.generate_audio_bytes_async(text, voice=voice, language=language))

    async def generate_audio_bytes_async(self, text: str, voice: str = None, language: str = None) -> bytes:
        """Asynchronous audio generation for callers with language-aware voice selection."""
        if not text:
            return b""
        if self.enable_voice_clone and (self._generate_fish_audio(text) or self._generate_local_f5(text)):
            with open(self.temp_file, "rb") as f:
                return f.read()
        selected_voice = voice or self.select_voice(text, language)
        communicate = edge_tts.Communicate(text, selected_voice)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    def generate_audio_file(self, text: str, language: str = None) -> bool:
        """Generates speech audio into self.temp_file (NO playback).

        Returns True if an audio file was produced. Never blocks on the local
        clone engine unless it is actually viable (GPU path).
        """
        if not text:
            return False

        self.stop_playback()

        print(f"🔊 [TTS] Generating voice reply ({language or 'auto'})...")
        success = False

        # 1. Try Local Zero-Shot Voice Cloning (only viable on GPU - CPU is
        #    thousands of times too slow, see require_gpu_for_clone).
        if self.enable_voice_clone and self.clone_ref_bytes and language not in ["ta"]:
            print("🎭 [TTS] Synthesizing speech with Local Cloned Voice Profile...")
            success = self._generate_local_f5(text)
            if not success:
                # 2. Try Fish Audio if local model not active
                success = self._generate_fish_audio(text)

            if success:
                print("✨ [TTS] Speech synthesized in your Cloned Voice!")

        # 3. Fallback to Edge-TTS if voice cloning fails or language is Tamil/Tanglish
        if not success:
            selected_voice = self.select_voice(text, language)
            print(f"ℹ️  [TTS] (Using Edge-TTS: {selected_voice}).")
            try:
                asyncio.run(self._generate_edge_fallback(text, voice=selected_voice, language=language))
                success = True
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")
                return False

        return success

    def play(self, stop_event: threading.Event):
        """Plays the already-generated audio with live interrupt/barge-in support."""
        print("🔊 [TTS] Playing audio...")
        try:
            pygame.mixer.music.load(self.temp_file)
            pygame.mixer.music.play()

            # Check every 0.03 seconds for barge-in stop event for instant responsiveness
            while pygame.mixer.music.get_busy():
                if stop_event.is_set():
                    print("🛑 [TTS] STOP EVENT RECEIVED! Interrupting speech NOW.")
                    self.stop_playback()
                    return
                time.sleep(0.03)

            self.stop_playback()
            print("🔊 [TTS] Finished naturally.")
        except Exception as e:
            logger.error(f"TTS playback failed: {e}")
            self.stop_playback()

    def speak(self, text: str, stop_event: threading.Event, language: str = None):
        """Generates and plays speech with live interrupt/barge-in support."""
        if not text:
            return
        if self.generate_audio_file(text, language):
            self.play(stop_event)