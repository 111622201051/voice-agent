# auth/enrollment.py
import re
import os
import json
import time
import logging
import threading
import soundfile as sf
import sounddevice as sd
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLONE_AUDIO_PATH = os.path.join(BASE_DIR, "data", "voice_profiles", "clone_reference.wav")
CLONE_META_PATH = os.path.join(BASE_DIR, "data", "voice_profiles", "clone_reference.json")


def clean_words(text: str):
    """Normalize text into lower-case alphanumeric word tokens."""
    return re.findall(r"\b[a-z0-9']+\b", text.lower())


def record_sentence_with_silence_detection(vad, sample_rate: int = 16000, silence_timeout: float = 1.8):
    """
    Records speech with zero rush:
    - Waits patiently for the user to start speaking at their own pace.
    - Records as long as the user speaks (no fixed timer).
    - Concludes only after the user stops speaking and stays silent for `silence_timeout` seconds.
    - User can also press ENTER anytime to manually finish.
    """
    audio_chunks = []
    has_spoken = False
    silence_chunks = 0
    silence_limit = int(silence_timeout / (512 / sample_rate))
    chunk_size = 512

    stop_event = threading.Event()

    def wait_for_enter():
        try:
            input()
            stop_event.set()
        except Exception:
            pass

    enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
    enter_thread.start()

    print("🎙️  Microphone is active. Speak whenever you are ready (Press ENTER if you finish early):\n")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        while not stop_event.is_set():
            chunk, _ = stream.read(chunk_size)
            chunk = chunk.flatten()
            chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))

            # Voice activity detection
            is_speech = vad.is_speech(chunk, threshold=0.30) or (chunk_rms > 0.009)

            if is_speech:
                has_spoken = True
                silence_chunks = 0
                audio_chunks.append(chunk)
            elif has_spoken:
                audio_chunks.append(chunk)
                silence_chunks += 1

                # Check if user has finished speaking
                recorded_seconds = (len(audio_chunks) * chunk_size) / sample_rate
                if recorded_seconds >= 2.5 and silence_chunks >= silence_limit:
                    break
            else:
                if len(audio_chunks) > 10:
                    audio_chunks.pop(0)
                audio_chunks.append(chunk)

            # Live UI status bar
            vol_blocks = int(min(chunk_rms * 150, 10))
            vol_bar = "█" * vol_blocks + "░" * (10 - vol_blocks)

            if not has_spoken:
                status = "👂 Waiting for you to start speaking..."
            else:
                dur = (len(audio_chunks) * chunk_size) / sample_rate
                silence_sec = (silence_chunks * chunk_size) / sample_rate
                if silence_chunks > 4:
                    status = f"🔴 Recording ({dur:.1f}s) | ⏳ Pause detected: {silence_sec:.1f}s / {silence_timeout:.1f}s"
                else:
                    status = f"🔴 Recording ({dur:.1f}s) | 🗣️  Speaking..."

            print(f"\r{status}  [Vol: {vol_bar}]   ", end="", flush=True)

    print("\n\n✅ Audio successfully captured!")
    if audio_chunks:
        return np.concatenate(audio_chunks)
    return np.zeros(sample_rate, dtype=np.float32)


def record_voice_clone_sample(vad, stt, sample_rate: int = 24000):
    """
    Step 2: Voice Cloning Reference Audio Recorder.
    Captures high-fidelity reference audio for Fish Audio / zero-shot cloning.
    """
    print("\n" + "=" * 65)
    print("🎭 STEP 2: VOICE CLONING (High-Fidelity Fish Audio Capture)")
    print("=" * 65)
    print("The agent will clone this exact voice, tone, timbre, emotion,")
    print("and prosody to reply to all your queries in your own voice!\n")

    clone_prompt = "Hello! I am your AI assistant, speaking directly in your cloned voice, emotion, and tone."

    while True:
        print("-" * 65)
        print("🎯 Please read this expressive sample aloud:")
        print(f"👉 \"{clone_prompt}\"")
        print("-" * 65)
        print("👉 Press [ENTER] when ready to record your cloned voice...")
        input()

        # Capture audio at 24kHz for high fidelity TTS
        audio = record_sentence_with_silence_detection(vad, sample_rate=sample_rate, silence_timeout=2.0)
        dur = len(audio) / sample_rate
        rms = float(np.sqrt(np.mean(audio ** 2)))

        if dur < 2.0 or rms < 0.005:
            print("\n⚠️  Sample too short or quiet. Let's record the voice clone phrase again.")
            continue

        print("🔍 Transcribing reference voice & extracting audio characteristics...")
        
        # Resample to 16kHz for Whisper transcription
        if sample_rate != 16000:
            import scipy.signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio_16k = scipy.signal.resample(audio, num_samples).astype(np.float32)
        else:
            audio_16k = audio

        transcribed_text = stt.transcribe(audio_16k)
        if not transcribed_text:
            transcribed_text = clone_prompt

        # Save reference WAV
        os.makedirs(os.path.dirname(CLONE_AUDIO_PATH), exist_ok=True)
        sf.write(CLONE_AUDIO_PATH, audio, sample_rate, subtype='PCM_16')

        # Save metadata
        metadata = {
            "prompt_text": transcribed_text,
            "original_prompt": clone_prompt,
            "sample_rate": sample_rate,
            "duration": dur,
            "rms": rms,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(CLONE_META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n🗣️  Cloned Reference Spoken : \"{transcribed_text}\"")
        print(f"📊 Reference Audio Duration : {dur:.1f}s ({sample_rate} Hz High Fidelity)")
        print(f"💾 Saved Reference Audio    : {CLONE_AUDIO_PATH}")
        print("✨ Voice Cloning Profile Configured Successfully!")
        print("=" * 65 + "\n")
        break


def enroll_user_interactive(verifier, vad, stt):
    """
    Combined Voice Authentication & Voice Cloning Wizard:
    - Step 1: Voice Verification Enrollment (3 sentences)
    - Step 2: High-Fidelity Voice Cloning Recording (Fish Audio)
    """
    print("\n" + "=" * 65)
    print("🎙️  VOICE ENROLLMENT - (Take your time & speak naturally)")
    print("=" * 65)
    print("Please read the following 3 sentences for speaker verification.\n")

    sentences = [
        "My voice is my password, verify me and grant access.",
        "The quick brown fox jumps over the lazy dog on a calm evening.",
        "I am the authorized user speaking naturally to train the voice agent."
    ]

    all_embeddings = []

    for idx, sentence in enumerate(sentences, 1):
        while True:
            print("\n" + "-" * 65)
            print(f"🎯 Sentence [{idx}/{len(sentences)}]:")
            print(f"👉 \"{sentence}\"")
            print("-" * 65)
            print("👉 Press [ENTER] to begin this sentence...")
            input()

            audio = record_sentence_with_silence_detection(vad, sample_rate=16000, silence_timeout=1.8)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            dur = len(audio) / 16000

            if dur < 1.5 or rms < 0.005:
                print("\n⚠️  Audio was too short or quiet. Let's record this sentence again.")
                continue

            print("🔍 Processing speech & extracting biometric signature...")
            transcribed = stt.transcribe(audio)
            print(f"🗣️  Transcribed : \"{transcribed}\"")
            print(f"📊 Sample Length: {dur:.1f} seconds")

            embedding = verifier._get_embedding(audio)
            all_embeddings.append(embedding)

            print(f"✅ Sentence [{idx}/{len(sentences)}] Enrolled Successfully!")
            time.sleep(0.5)
            break

    # Finalize Profile
    print("\n" + "=" * 65)
    print("🔐 Saving your biometric voice profile...")
    verifier.enroll_embeddings(all_embeddings)
    print("✅ Voice Verification Enrolled (3 biometric samples).")

    # Step 2: Record Voice Cloning Reference
    record_voice_clone_sample(vad, stt)
