# agent/harness.py
import threading
import queue
import numpy as np
import sounddevice as sd
import logging
import time
from enum import Enum

# Optional UI bridge import
try:
    from ui.bridge import UIBridge
except ImportError:
    UIBridge = None

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class AgentHarness:
    def __init__(self, stt, vad, tts, llm, verifier, memory, ui_bridge=None):
        self.stt = stt
        self.vad = vad
        self.tts = tts
        self.llm = llm
        self.verifier = verifier
        self.memory = memory
        self.ui_bridge = ui_bridge

        self.state = AgentState.IDLE
        self.tts_stop_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.audio_queue = queue.Queue()

        if self.ui_bridge:
            self.ui_bridge.is_running = True
            self.ui_bridge.set_status("Idle")

        # Warm up models once at boot so the user's first query has 0 cold-start lag
        self._warmup()

        self.mic_thread = threading.Thread(target=self._mic_listener, daemon=True)
        self.mic_thread.start()

        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

    def _warmup(self):
        print("🔥 [DEBUG] Warming up AI models (STT, LLM, Verifier) to eliminate initial lag...")
        try:
            # Low-amplitude noise to warm up models without zero-division warning
            dummy_audio = (np.random.randn(16000) * 0.02).astype(np.float32)
            self.stt.transcribe(dummy_audio)
            self.verifier._get_embedding(dummy_audio)
            self.llm.chat([{"role": "user", "content": "hi"}])
            print("✅ [DEBUG] Models warmed up and ready.")
        except Exception as e:
            logger.warning(f"Model warmup note: {e}")

    def _mic_listener(self):
        chunk_size = 512
        audio_buffer = []
        is_speaking = False
        silence_counter = 0
        sample_counter = 0

        # Pre-speech ring buffer (~320ms: 10 chunks of 512 @ 16kHz) to guarantee opening words are never cut
        pre_speech_chunks = []

        # Interruption tracking buffers
        interrupt_buffer = []
        interrupt_candidate_speaking = False
        interrupt_speech_samples = 0
        interrupt_last_check_samples = 0
        interrupt_silence_counter = 0

        print("🎧 [DEBUG] Mic listener thread started (Always On, True Barge-In with Biometric Verification).")

        with sd.InputStream(samplerate=16000, channels=1, dtype='float32') as stream:
            while not self.shutdown_event.is_set():
                chunk, _ = stream.read(chunk_size)
                chunk = chunk.flatten()
                chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))

                # -------------------------------------------------------------
                # CASE 1: AI IS SPEAKING (AgentState.SPEAKING)
                # -------------------------------------------------------------
                if self.state == AgentState.SPEAKING:
                    # While AI speaks, potential user speech must be audible above speaker echo
                    is_loud = chunk_rms > 0.045
                    has_speech = self.vad.is_speech(chunk, threshold=0.55) and is_loud

                    if has_speech:
                        if not interrupt_candidate_speaking:
                            # Start accumulating potential interruption audio
                            # CRITICAL: Pre-populate with pre-speech chunks so the first syllable/words are preserved!
                            interrupt_buffer = list(pre_speech_chunks)
                            interrupt_candidate_speaking = True
                            interrupt_speech_samples = len(interrupt_buffer) * chunk_size
                            interrupt_last_check_samples = 0
                            interrupt_silence_counter = 0

                        interrupt_buffer.append(chunk)
                        interrupt_speech_samples += chunk_size
                        interrupt_silence_counter = 0

                        # Quick speaker biometric check once we have ~0.5s of audio (>= 8000 samples).
                        # Re-check every ~125ms afterwards, so a genuine barge-in that was only
                        # borderline at 0.5s still gets confirmed before the 0.75s non-user cutoff.
                        if (interrupt_speech_samples >= 8000
                                and (interrupt_speech_samples - interrupt_last_check_samples) >= 2000):
                            interrupt_last_check_samples = interrupt_speech_samples
                            candidate_audio = np.concatenate(interrupt_buffer)
                            is_auth, score = self.verifier.verify(candidate_audio, min_threshold=0.58)

                            if is_auth:
                                # CANDIDATE INTERRUPT CONFIRMED!
                                print(f"\n⚡ [USER INTERRUPT CONFIRMED] Candidate voice verified! (Score: {score:.2f}). Stopping AI immediately!")
                                # 1. Immediately halt and unload currently playing AI speech
                                self.tts_stop_event.set()
                                self.tts.stop_playback()

                                # 2. Immediately transition to listening to the candidate's new speech
                                self.state = AgentState.IDLE
                                is_speaking = True
                                silence_counter = 0
                                sample_counter = interrupt_speech_samples
                                audio_buffer = list(interrupt_buffer)

                                # Reset interrupt state
                                interrupt_buffer = []
                                interrupt_candidate_speaking = False
                                interrupt_speech_samples = 0
                                interrupt_last_check_samples = 0
                                interrupt_silence_counter = 0
                            else:
                                # If accumulated >= 12000 samples (~0.75s) and still not verified:
                                # It is a background speaker or environmental noise!
                                if interrupt_speech_samples >= 12000:
                                    print(f"\r🛡️ [INTERRUPT IGNORED] Non-user/Background voice detected (Score: {score:.2f}). AI continues speaking...   ", end="", flush=True)
                                    interrupt_buffer = []
                                    interrupt_candidate_speaking = False
                                    interrupt_speech_samples = 0
                                    interrupt_last_check_samples = 0
                                    interrupt_silence_counter = 0

                    elif interrupt_candidate_speaking:
                        # Candidate paused or was brief background blip
                        interrupt_buffer.append(chunk)
                        interrupt_silence_counter += 1
                        if interrupt_silence_counter >= 8:
                            # Silence reached without verification -> discard
                            interrupt_buffer = []
                            interrupt_candidate_speaking = False
                            interrupt_speech_samples = 0
                            interrupt_last_check_samples = 0
                            interrupt_silence_counter = 0
                    else:
                        # Update rolling pre-speech ring buffer while speaking
                        if len(pre_speech_chunks) >= 10:
                            pre_speech_chunks.pop(0)
                        pre_speech_chunks.append(chunk)

                # -------------------------------------------------------------
                # CASE 2: AI IS THINKING (AgentState.PROCESSING)
                # -------------------------------------------------------------
                elif self.state == AgentState.PROCESSING:
                    audio_buffer = []
                    is_speaking = False
                    silence_counter = 0
                    sample_counter = 0
                    interrupt_buffer = []
                    interrupt_candidate_speaking = False
                    interrupt_speech_samples = 0
                    interrupt_last_check_samples = 0
                    interrupt_silence_counter = 0

                # -------------------------------------------------------------
                # CASE 3: IDLE / LISTENING (AgentState.IDLE)
                # -------------------------------------------------------------
                else:
                    has_speech = self.vad.is_speech(chunk, threshold=self.vad.threshold)

                    if has_speech:
                        if not is_speaking:
                            # First moment of speech: seed with pre-speech buffer so opening word is never cut!
                            audio_buffer = list(pre_speech_chunks)
                            sample_counter = len(audio_buffer) * chunk_size
                            is_speaking = True
                            print(f"\r👂 [DEBUG] Speech detected! Volume: {chunk_rms:.4f}   ", end="", flush=True)

                        audio_buffer.append(chunk)
                        sample_counter += chunk_size
                        silence_counter = 0

                    elif is_speaking:
                        silence_counter += 1
                        sample_counter += chunk_size
                        audio_buffer.append(chunk)

                        silence_limit = 14   # ~450ms silence marks completion of utterance
                        max_samples = 80000  # 5 seconds max per utterance turn

                        if silence_counter >= silence_limit or sample_counter >= max_samples:
                            full_audio = np.concatenate(audio_buffer)
                            if len(full_audio) >= 4800:
                                # Clear any stale backlog
                                while not self.audio_queue.empty():
                                    try:
                                        self.audio_queue.get_nowait()
                                    except queue.Empty:
                                        break

                                print(f"\n📦 [DEBUG] Captured {len(full_audio) / 16000:.1f}s of speech for processing.")
                                self.audio_queue.put(full_audio)

                            audio_buffer = []
                            is_speaking = False
                            silence_counter = 0
                            sample_counter = 0
                    else:
                        # Keep pre-speech ring buffer populated
                        if len(pre_speech_chunks) >= 10:
                            pre_speech_chunks.pop(0)
                        pre_speech_chunks.append(chunk)

    def _process_loop(self):
        print("🧠 [DEBUG] Process loop thread started.")
        if self.ui_bridge:
            self.ui_bridge.set_status("Idle")

        while not self.shutdown_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # 1. Biometric Speaker Verification
            is_authorized, similarity = self.verifier.verify(audio_data)
            if not is_authorized:
                print(f"🚫 [DEBUG] Non-user voice ignored! Score: {similarity:.2f}")
                if self.ui_bridge:
                    self.ui_bridge.set_status("🚫 Unknown Speaker")
                continue

            print(f"✅ [DEBUG] Authorized user verified! Score: {similarity:.2f}")
            if self.ui_bridge:
                self.ui_bridge.set_status("✅ Authorized")

            # 2. Multilingual Transcribe (Tamil / English / Tanglish)
            self.state = AgentState.PROCESSING
            if self.ui_bridge:
                self.ui_bridge.set_status("🔄 Processing...")

            user_text, detected_lang = self.stt.transcribe_with_language(audio_data)
            if not user_text or len(user_text.strip()) < 2:
                self.state = AgentState.IDLE
                if self.ui_bridge:
                    self.ui_bridge.set_status("Idle")
                continue

            print(f"\n👤 You said ({detected_lang}): '{user_text}'")
            if self.ui_bridge:
                self.ui_bridge.add_message("user", user_text)

            if user_text.lower() in ['quit', 'exit', 'stop', 'goodbye']:
                print("👋 Goodbye!")
                if self.ui_bridge:
                    self.ui_bridge.set_status("👋 Goodbye!")
                self.shutdown_event.set()
                break

            # 3. LLM Reasoning with Language Mirroring
            self.memory.add_message("user", user_text)
            history = self.memory.get_recent_history(limit=10)
            print(f"🧠 Thinking ({detected_lang})...")
            if self.ui_bridge:
                self.ui_bridge.set_status(f"🧠 Thinking ({detected_lang})...")

            assistant_text = self.llm.chat(history, language_style=detected_lang)
            print(f"🤖 Assistant ({detected_lang}): '{assistant_text}'")
            self.memory.add_message("assistant", assistant_text)

            # 4. Voice Synthesis & Playback
            audio_bytes = None
            if self.ui_bridge:
                try:
                    audio_bytes = self.tts.generate_audio_bytes(assistant_text, language=detected_lang)
                except Exception as e:
                    logger.warning(f"Audio bytes generation for web UI note: {e}")

                self.ui_bridge.add_message("assistant", assistant_text, audio_bytes=audio_bytes)
                self.ui_bridge.set_status("🔊 Speaking...")
                est_duration = max(1.0, len(assistant_text.split()) * 0.35)
                time.sleep(est_duration)
            else:
                # Standalone CLI mode: Play on physical speakers with live interrupt support
                self.state = AgentState.SPEAKING
                self.tts_stop_event.clear()

                print(f"🔊 [TTS] Starting playback ({detected_lang})...")
                self.tts.speak(assistant_text, self.tts_stop_event, language=detected_lang)

                # Brief echo clearing pause
                time.sleep(0.2)

            # Reset
            self.state = AgentState.IDLE
            self.tts_stop_event.clear()

            # NOTE: The queue is NOT drained here on purpose. A verified
            # barge-in may have just queued the user's fresh utterance while
            # this response was being interrupted. Draining would silently
            # drop those first (interrupted) words.

            if self.ui_bridge:
                self.ui_bridge.set_status("Idle")

    def run(self):
        print("\n✅ Agent Harness is running.")
        print("💡 Tip: Speak clearly. The AI understands English, Tamil (தமிழ்), and Tanglish.")
        print("💡 Tip: If you interrupt while the AI speaks, it will recognize your voice and stop immediately.")
        try:
            while not self.shutdown_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            self.tts.stop_playback()
            self.shutdown_event.set()