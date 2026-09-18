# main.py
import os
import sys
import logging
import sounddevice as sd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from auth.enrollment import enroll_user_interactive

def main():
    try:
        print("=" * 60)
        print("🤖 Shree Voice AI Interviewer")
        print("🌐 Multilingual: Tamil (தமிழ்) | English | Tanglish")
        print("🛡️ Biometric Speaker Recognition & Instant Interruption")
        print("=" * 60)

        print("1. Initializing Multilingual STT (Whisper)...")
        from audio.stt import SpeechToText
        stt = SpeechToText(model_size="small", device="cpu")
        print("   ✅ STT Ready (Multilingual)")

        print("2. Initializing VAD...")
        from audio.vad import VoiceActivityDetector
        vad = VoiceActivityDetector(threshold=0.4)
        print("   ✅ VAD Ready")

        print("3. Initializing Multilingual TTS...")
        from audio.tts import TextToSpeech
        tts = TextToSpeech()
        print("   ✅ TTS Ready (Tamil + English Neural Voices)")

        print("4. Initializing LLM...")
        from agent.llm_client import LLMClient
        llm = LLMClient(model_name="qwen2.5:3b")
        print("   ✅ LLM Ready")

        print("5. Initializing Voice Auth...")
        from auth.verification import SpeakerVerifier
        verifier = SpeakerVerifier(threshold=0.60)

        from auth.enrollment import record_voice_clone_sample
        CLONE_AUDIO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "voice_profiles", "clone_reference.wav")

        force_enroll = "--enroll" in sys.argv or "--re-enroll" in sys.argv
        if force_enroll or not verifier.load_profile():
            enroll_user_interactive(verifier, vad, stt)
            tts.reload_clone_reference()
        elif "--clone-voice" in sys.argv or not os.path.exists(CLONE_AUDIO):
            record_voice_clone_sample(vad, stt)
            tts.reload_clone_reference()
        else:
            print("   ✅ Voice Profile Loaded.")

        print("6. Initializing Database Memory...")
        from agent.memory import MemoryManager
        memory = MemoryManager()
        print("   ✅ Memory Ready")

        system_instruction = (
            "You are Shree, a helpful, fast, and natural AI interviewer. "
            "LANGUAGE RULES: "
            "1. Detect and mirror the candidate's language: "
            "- If user speaks in Tamil (தமிழ்), reply strictly in natural spoken Tamil (தமிழ்). "
            "- If user speaks in English, reply in clear, professional English. "
            "- If user speaks in Tanglish (Tamil-English mixed, e.g. 'Machine learning pathi sollunga'), reply naturally in Tanglish matching their exact style. "
            "2. Keep responses brief (1 to 2 short sentences), natural for voice speech. "
            "3. NEVER use bullet points, numbered lists, asterisks (*), hashtags, or markdown formatting."
        )
        memory.add_message("system", system_instruction)

        print("\n✅ All engines loaded. Starting Agent Harness...")

        # Initialize Harness
        from agent.harness import AgentHarness
        harness = AgentHarness(stt, vad, tts, llm, verifier, memory)
        harness.run()

    except Exception as e:
        print(f"\n❌ FATAL ERROR during startup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()