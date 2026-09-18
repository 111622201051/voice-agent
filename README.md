# 🤖 Shree — Local Voice Agent

A fully local, voice-controlled AI assistant. It listens, verifies your voice, transcribes your speech, generates a response with a local LLM, and speaks it back — with true barge-in support (you can interrupt mid-sentence).

No cloud API keys needed. Everything runs on your machine.

---

## ⚠️ Important before you start

This folder may contain a `data/` directory with the **original owner's private files** — their voice profile (`my_voice.pkl`) and conversation history (`agent.db`). If you received this as a zip:

1. **Delete the `data/` folder** first (or don't copy it into the project).
2. Then continue with the setup below.
3. On your first run, the app will automatically ask you to **enroll your own voice** (read 3 sentences). This is required and only takes a minute.

Don't try to use someone else's voice profile — the app won't recognize you.

---

## ✅ Prerequisites

Before installing the Python packages, install these first:

1. **Python 3.10 or newer** (64-bit)
   - Download from [python.org](https://www.python.org/downloads/)
   - During install, tick **"Add Python to PATH"**

2. **Ollama** (runs the local LLM)
   - Download from [ollama.com](https://ollama.com/download)
   - After installing, open a terminal and pull the AI model:
     ```
     ollama pull qwen2.5:3b
     ```
   - Keep Ollama running in the background while using this app.

3. **A working microphone**

---

## 📦 Installation

> **Required:** Python 3.10, 3.11 or **3.12** (64-bit). Do not use Python 3.13/3.14 —
> core packages (torch, pygame, webrtcvad-wheels) don't publish wheels for them yet.
> Download from [python.org](https://www.python.org/downloads/) and tick **"Add Python to PATH"**.

### Windows — one-click setup (recommended)

Double-click **`setup_windows.bat`**. It creates a `.venv`, installs every
dependency from the pinned lockfile, handles the resemblyzer/webrtcvad quirk,
asks about the optional local voice-cloning engine, and checks for Ollama.

### Manual install

To see exactly what gets installed, read `requirements.txt`. To reproduce the
exact pinned environment, use the full lockfile:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install webrtcvad-wheels==2.0.14        # binary wheel, avoids MSVC build
pip install -r requirements-lock.txt       # every package pinned
pip install resemblyzer==0.1.4 --no-deps --no-build-isolation
```

(`resemblyzer` is installed separately because its obsolete `webrtcvad>=2.0.10`
dependency only ships as source and would force a Microsoft Visual C++ build;
`webrtcvad-wheels` supplies the same module.)

It will download several large packages (PyTorch, Whisper, etc.), so it may
take a few minutes.

> **Note:** The first time you use Whisper (speech-to-text) it downloads a small model automatically (~75MB).

---

## 🔊 First Run — Voice Enrollment

The assistant only listens to **your voice**. The first time you run it, you'll be asked to read 3 short sentences so it can learn your voice.

```
python main.py
```

Follow the prompts. Read each sentence clearly for at least 5 seconds.

Once enrolled, your voice profile is saved in `data/voice_profiles/`. You only enroll once.

---

## 🚀 Running

### Option 1 — Terminal / CLI (recommended)

```
python main.py
```

### Option 2 — Web UI (Streamlit)

After enrolling your voice (step above), start the web chat interface:

```
python -m streamlit run ui/app.py
```

Your browser will auto-open. Click **Start Agent**, then use it from the web page.

---

## 🧪 Quick Test

To check all engines load correctly without speaking:

```
python test_run.py
```

To view what's stored in the conversation database:

```
python check_db.py
```

---

## 🛠️ Useful Info

- **AI model**: `qwen2.5:3b`, running locally via Ollama
- **Language model location**: wherever Ollama stores it
- **Conversation history**: stored in `data/agent.db` (SQLite)
- **Voice profile**: `data/voice_profiles/my_voice.pkl`
- **TTS voice**: Microsoft Edge neural voice (`en-US-GuyNeural`) — needs internet only for audio generation

---

## ❗ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` on startup | Make sure setup completed: run `setup_windows.bat` (or `pip install -r requirements-lock.txt`) |
| "Cannot connect to Ollama" error | Ollama isn't running. Start the Ollama app, then check `ollama list` shows `qwen2.5:3b` |
| No microphone audio / silence | Check your default microphone in Windows sound settings |
| Whisper model download fails | Re-run — it resumes. Or check internet connection |
| UI shows error about `AgentHarness` | Make sure `requirements.txt` installed fine and you enrolled your voice first |

---

## 📁 Project Layout

```
voice-agent/
├── main.py              # CLI entry point
├── setup_windows.bat    # one-click installer (creates .venv, installs deps)
├── requirements.txt     # pinned dependencies (Python 3.10-3.12)
├── requirements-lock.txt# full locked dependency tree (reproducible install)
├── agent/               # LLM client, memory, orchestration harness
├── audio/               # STT, TTS, VAD
├── auth/                # Voice verification/enrollment
├── db/                  # Database session + models
├── ui/                  # Streamlit web UI + bridge
├── api/                 # (planned) REST API
└── data/                # Runtime data (db, profiles, recordings) - not shared
```

---

Built with: Whisper (STT) · Silero VAD · Edge TTS · Ollama/Qwen (LLM) · Resemblyzer (voice ID) · Streamlit (UI) · SQLite (memory)
