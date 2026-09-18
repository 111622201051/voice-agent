# api/routes.py
import io
import os
import sys
import base64
import logging
import soundfile as sf
import librosa
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

# Ensure project root is importable regardless of launch directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from audio.stt import SpeechToText
from audio.tts import TextToSpeech
from agent.llm_client import LLMClient
from auth.verification import SpeakerVerifier
from agent.memory import MemoryManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Lazy-loaded engine singletons shared across the web app
stt_engine = None
tts_engine = None
llm_engine = None
verifier_engine = None
memory_engine = None


def init_engines():
    global stt_engine, tts_engine, llm_engine, verifier_engine, memory_engine
    if stt_engine is None:
        logger.info("Initializing API AI engines...")
        stt_engine = SpeechToText(model_size="small", device="cpu")
        tts_engine = TextToSpeech()
        llm_engine = LLMClient(model_name="qwen2.5:3b")
        verifier_engine = SpeakerVerifier(threshold=0.60)
        verifier_engine.load_profile()
        memory_engine = MemoryManager()
        system_instruction = (
            "You are Shree, a fast and natural AI interviewer. "
            "LANGUAGE RULES: "
            "- If user speaks in Tamil (தமிழ்), reply strictly in natural spoken Tamil (தமிழ்). "
            "- If user speaks in English, reply in clear, professional English. "
            "- If user speaks in Tanglish (Tamil-English mixed), reply naturally in Tanglish matching their exact style. "
            "Keep answers concise (1 to 2 short sentences). Do NOT use bullet points, asterisks (*), or markdown."
        )
        memory_engine.add_message("system", system_instruction)
        logger.info("API AI engines ready.")


def decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    audio_io = io.BytesIO(audio_bytes)
    try:
        data, sr = sf.read(audio_io, dtype='float32')
    except Exception:
        audio_io.seek(0)
        data, sr = librosa.load(audio_io, sr=16000)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    if len(data) < 16000:
        data = np.pad(data, (0, 16000 - len(data)))
    return data


@router.get("/health")
def health():
    init_engines()
    return {
        "status": "ok",
        "profile_loaded": len(verifier_engine.embeddings) > 0 if verifier_engine else False,
        "voice_clone_enabled": tts_engine.enable_voice_clone if tts_engine else False,
    }


@router.post("/process_voice")
async def process_voice(
    audio: UploadFile = File(...),
    turn_token: str = Form(""),
):
    init_engines()
    raw_bytes = await audio.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    print(f"[API] Received voice chunk ({len(raw_bytes)} bytes) turn={turn_token}")
    audio_array = decode_audio_bytes(raw_bytes)

    # 1. Biometric Authentication
    verifier_engine.load_profile()
    is_auth, score = verifier_engine.verify(audio_array)
    print(f"[API] Verification: authorized={is_auth}, score={score:.2f}")

    if not is_auth:
        return {
            "authorized": False,
            "similarity": round(float(score), 3),
            "turn_token": turn_token,
            "user_text": "",
            "assistant_text": "Access denied: Unknown speaker.",
            "audio_base64": "",
            "language": "en"
        }

    # 2. Multilingual STT (Whisper)
    user_text, detected_lang = stt_engine.transcribe_with_language(audio_array)
    print(f"[API] Transcribed STT ({detected_lang}): '{user_text}'")

    if not user_text or len(user_text.strip()) < 2:
        return {
            "authorized": True,
            "similarity": round(float(score), 3),
            "turn_token": turn_token,
            "user_text": "",
            "assistant_text": "",
            "audio_base64": "",
            "language": detected_lang
        }

    # 3. LLM Reasoning with Language Mirroring
    memory_engine.add_message("user", user_text)
    history = memory_engine.get_recent_history(limit=10)
    assistant_text = llm_engine.chat(history, language_style=detected_lang)
    print(f"[API] Assistant Response ({detected_lang}): '{assistant_text}'")
    memory_engine.add_message("assistant", assistant_text)

    # 4. Neural TTS Generation with Language-Aware Voice
    tts_bytes = await tts_engine.generate_audio_bytes_async(assistant_text, language=detected_lang)
    audio_base64 = base64.b64encode(tts_bytes).decode("utf-8") if tts_bytes else ""
    print(f"[API] Generated TTS audio ({len(tts_bytes)} bytes) for client playback.")

    return {
        "authorized": True,
        "similarity": round(float(score), 3),
        "turn_token": turn_token,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "audio_base64": audio_base64,
        "language": detected_lang
    }


@router.post("/enroll")
async def enroll_voice(s1: UploadFile = File(...), s2: UploadFile = File(...), s3: UploadFile = File(...)):
    init_engines()
    b1 = await s1.read()
    b2 = await s2.read()
    b3 = await s3.read()
    samples = [decode_audio_bytes(b1), decode_audio_bytes(b2), decode_audio_bytes(b3)]
    verifier_engine.enroll(samples)
    return {"status": "enrolled", "count": len(verifier_engine.embeddings)}