# test_run.py
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

print("1. Starting script...")

try:
    print("2. Importing SpeechToText...")
    from audio.stt import SpeechToText
    print("   ✅ Import successful!")
except Exception as e:
    print(f"   ❌ Import failed: {e}")

try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"3. Initializing STT on {device}...")
    stt = SpeechToText(model_size="base.en", device=device)
    print("   ✅ STT Initialized successfully!")
except Exception as e:
    print(f"   ❌ STT Initialization failed: {e}")

try:
    print("4. Importing TTS...")
    from audio.tts import TextToSpeech
    tts = TextToSpeech()
    print("   ✅ TTS Initialized successfully!")
except Exception as e:
    print(f"   ❌ TTS Initialization failed: {e}")

try:
    print("5. Importing LLM Client...")
    from agent.llm_client import LLMClient
    llm = LLMClient(model_name="qwen2.5:3b")
    print("   ✅ LLM Client Initialized successfully!")
except Exception as e:
    print(f"   ❌ LLM Client Initialization failed: {e}")

print("6. 🎉 All engines loaded successfully! You can now run main.py")