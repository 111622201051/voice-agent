# ui/app.py
import sys
import os
import threading
import time
import io
import soundfile as sf
import librosa
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import uvicorn

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from api.server import app as fastapi_app, init_engines, verifier_engine

# Start background FastAPI server on port 8000
@st.cache_resource
def start_fastapi_server():
    init_engines()
    def run_api():
        try:
            uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="warning")
        except Exception as e:
            print(f"API server note: {e}")
    
    t = threading.Thread(target=run_api, daemon=True)
    t.start()
    time.sleep(1.0)
    return True

start_fastapi_server()

# Page config
st.set_page_config(page_title="Voice Agent - Live Call", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main { background: #0e1117; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Live Voice Call Mode")
st.caption("ChatGPT-Style Continuous Hands-Free Voice Agent — Audio plays directly on your connected device")

def decode_audio_file(uploaded_file):
    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()
    if not raw_bytes:
        raise ValueError("Audio recording was empty.")
    audio_io = io.BytesIO(raw_bytes)
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

tab_call, tab_enroll = st.tabs(["📞 Continuous Live Call", "🔐 Voice Registration"])

with tab_call:
    # Full-Duplex Continuous Voice Call Web Component with 16kHz PCM WAV Encoder
    voice_call_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            body { background: #12141c; color: #e2e8f0; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 15px; }
            
            .call-container { width: 100%; max-width: 760px; background: #1a1d28; border: 1px solid #2d3348; border-radius: 20px; padding: 30px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
            
            /* Glowing Voice Orb */
            .orb-wrapper { position: relative; width: 140px; height: 140px; margin: 20px auto; display: flex; align-items: center; justify-content: center; }
            .orb { width: 90px; height: 90px; border-radius: 50%; background: linear-gradient(135deg, #3b82f6, #8b5cf6); transition: all 0.3s ease; box-shadow: 0 0 20px rgba(139, 92, 246, 0.4); }
            
            .orb.idle { background: linear-gradient(135deg, #4b5563, #6b7280); box-shadow: 0 0 10px rgba(107, 114, 128, 0.2); }
            .orb.listening { background: linear-gradient(135deg, #10b981, #059669); animation: pulse-green 1.4s infinite; box-shadow: 0 0 35px rgba(16, 185, 129, 0.6); }
            .orb.speaking { background: linear-gradient(135deg, #ec4899, #8b5cf6); animation: pulse-purple 1.1s infinite; box-shadow: 0 0 45px rgba(236, 72, 153, 0.7); }
            .orb.thinking { background: linear-gradient(135deg, #f59e0b, #d97706); animation: spin-orb 1.8s infinite linear; box-shadow: 0 0 30px rgba(245, 158, 11, 0.6); }

            @keyframes pulse-green { 0% { transform: scale(1); } 50% { transform: scale(1.14); } 100% { transform: scale(1); } }
            @keyframes pulse-purple { 0% { transform: scale(1); } 50% { transform: scale(1.18); } 100% { transform: scale(1); } }
            @keyframes spin-orb { 0% { transform: rotate(0deg) scale(1.05); } 50% { transform: rotate(180deg) scale(0.95); } 100% { transform: rotate(360deg) scale(1.05); } }

            .status-text { font-size: 20px; font-weight: 600; margin-top: 10px; color: #94a3b8; }
            .status-text.listening { color: #34d399; }
            .status-text.speaking { color: #f472b6; }
            .status-text.thinking { color: #fbbf24; }

            /* Call Controls */
            .controls { margin-top: 25px; display: flex; justify-content: center; gap: 15px; }
            .btn { padding: 14px 28px; font-size: 16px; font-weight: 600; border-radius: 50px; border: none; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 8px; }
            .btn-start { background: linear-gradient(135deg, #10b981, #059669); color: white; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4); }
            .btn-start:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6); }
            .btn-stop { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4); }
            .btn-stop:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6); }

            /* Live Conversation Box */
            .transcript-box { margin-top: 25px; max-height: 240px; overflow-y: auto; text-align: left; background: #0f111a; border-radius: 12px; padding: 15px; border: 1px solid #232838; }
            .msg { margin-bottom: 12px; padding: 10px 14px; border-radius: 10px; font-size: 15px; line-height: 1.4; }
            .msg.user { background: #1e293b; color: #93c5fd; border-left: 4px solid #3b82f6; }
            .msg.assistant { background: #281e3b; color: #f472b6; border-left: 4px solid #d946ef; }
        </style>
    </head>
    <body>
        <div class="call-container">
            <h2 style="color: #f8fafc; font-size: 22px; margin-bottom: 6px;">🎙️ AI Voice Call</h2>
            <p style="color: #64748b; font-size: 14px;">Hands-Free Voice Flow • Speak naturally into your mic</p>

            <div class="orb-wrapper">
                <div id="orb" class="orb idle"></div>
            </div>

            <div id="status" class="status-text">Click "Start Call" to Connect</div>

            <div class="controls">
                <button id="btn-toggle" class="btn btn-start" onclick="toggleCall()">
                    <span>📞</span> Start Voice Call
                </button>
            </div>

            <div class="transcript-box" id="transcript">
                <div style="color: #64748b; font-style: italic; font-size: 13px;">Live conversation will stream here...</div>
            </div>
        </div>

        <script>
            let isCalling = false;
            let audioContext = null;
            let mediaStream = null;
            let scriptProcessor = null;
            let pcmData = [];
            let isUserSpeaking = false;
            let silenceCount = 0;
            let speechCount = 0;
            let currentAudioPlayer = null;
            
            // Resolve API host dynamically (handling remote IP, iframe origin, etc.)
            let apiHost = window.location.hostname;
            if (!apiHost || apiHost === "about:srcdoc" || apiHost === "null") {
                try {
                    apiHost = window.parent.location.hostname;
                } catch(e) {
                    apiHost = "127.0.0.1";
                }
            }
            if (!apiHost) apiHost = "127.0.0.1";
            const API_URL = `http://${apiHost}:8000/api/process_voice`;

            function encodeWAV(samples, sampleRate) {
                const buffer = new ArrayBuffer(44 + samples.length * 2);
                const view = new DataView(buffer);

                function writeString(view, offset, string) {
                    for (let i = 0; i < string.length; i++) {
                        view.setUint8(offset + i, string.charCodeAt(i));
                    }
                }

                writeString(view, 0, 'RIFF');
                view.setUint32(4, 36 + samples.length * 2, true);
                writeString(view, 8, 'WAVE');
                writeString(view, 12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true); // PCM
                view.setUint16(22, 1, true); // 1 channel
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * 2, true);
                view.setUint16(32, 2, true);
                view.setUint16(34, 16, true); // 16-bit
                writeString(view, 36, 'data');
                view.setUint32(40, samples.length * 2, true);

                let offset = 44;
                for (let i = 0; i < samples.length; i++, offset += 2) {
                    let s = Math.max(-1, Math.min(1, samples[i]));
                    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                }

                return new Blob([view], { type: 'audio/wav' });
            }

            function setOrbState(state, text) {
                const orb = document.getElementById("orb");
                const status = document.getElementById("status");
                orb.className = "orb " + state;
                status.className = "status-text " + state;
                status.innerText = text;
            }

            function addMessage(role, text) {
                const box = document.getElementById("transcript");
                const msg = document.createElement("div");
                msg.className = "msg " + role;
                msg.innerHTML = `<strong>${role === 'user' ? '👤 You' : '🤖 Shree'}:</strong> ${text}`;
                box.appendChild(msg);
                box.scrollTop = box.scrollHeight;
            }

            async function toggleCall() {
                if (!isCalling) {
                    await startCall();
                } else {
                    stopCall();
                }
            }

            async function startCall() {
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({ 
                        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 } 
                    });
                    
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    const source = audioContext.createMediaStreamSource(mediaStream);
                    
                    // Use ScriptProcessor for real-time PCM audio streaming
                    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                    pcmData = [];
                    isUserSpeaking = false;
                    silenceCount = 0;
                    speechCount = 0;

                    scriptProcessor.onaudioprocess = e => {
                        if (!isCalling) return;
                        const input = e.inputBuffer.getChannelData(0);
                        
                        // Calculate RMS Volume
                        let sum = 0;
                        for (let i = 0; i < input.length; i++) {
                            sum += input[i] * input[i];
                        }
                        const rms = Math.sqrt(sum / input.length);

                        if (rms > 0.015) {
                            speechCount++;
                            silenceCount = 0;
                            if (speechCount >= 2 && !isUserSpeaking) {
                                isUserSpeaking = true;
                                if (currentAudioPlayer && !currentAudioPlayer.paused) {
                                    currentAudioPlayer.pause(); // Barge-in interrupt
                                }
                                setOrbState("listening", "Listening to you...");
                            }
                        } else if (isUserSpeaking) {
                            silenceCount++;
                        }

                        if (isUserSpeaking) {
                            for (let i = 0; i < input.length; i++) {
                                pcmData.push(input[i]);
                            }

                            // If ~600ms of pause detected after speaking
                            if (silenceCount >= 3 && pcmData.length >= 12000) {
                                isUserSpeaking = false;
                                speechCount = 0;
                                silenceCount = 0;
                                
                                const finalPcm = new Float32Array(pcmData);
                                pcmData = [];
                                const wavBlob = encodeWAV(finalPcm, 16000);
                                sendVoiceQuery(wavBlob);
                            }
                        }
                    };

                    source.connect(scriptProcessor);
                    scriptProcessor.connect(audioContext.destination);

                    isCalling = true;
                    document.getElementById("btn-toggle").className = "btn btn-stop";
                    document.getElementById("btn-toggle").innerHTML = "<span>🛑</span> End Voice Call";
                    setOrbState("listening", "Listening... Speak naturally");
                } catch (err) {
                    alert("Microphone error: " + err.message);
                }
            }

            function stopCall() {
                isCalling = false;
                if (mediaStream) {
                    mediaStream.getTracks().forEach(track => track.stop());
                }
                if (audioContext) {
                    audioContext.close();
                }
                if (currentAudioPlayer) {
                    currentAudioPlayer.pause();
                }
                document.getElementById("btn-toggle").className = "btn btn-start";
                document.getElementById("btn-toggle").innerHTML = "<span>📞</span> Start Voice Call";
                setOrbState("idle", "Call Ended");
            }

            async function sendVoiceQuery(blob) {
                if (!isCalling) return;
                setOrbState("thinking", "Thinking & Generating...");

                const formData = new FormData();
                formData.append("audio", blob, "voice.wav");

                try {
                    const response = await fetch(API_URL, {
                        method: "POST",
                        body: formData
                    });
                    const data = await response.json();

                    if (!data.authorized) {
                        setOrbState("idle", "🚫 Unknown Speaker");
                        addMessage("assistant", "Access Denied: Voice does not match authorized profile.");
                        setTimeout(() => { if (isCalling) setOrbState("listening", "Listening... Speak naturally"); }, 2000);
                        return;
                    }

                    if (data.user_text) {
                        addMessage("user", data.user_text);
                    }

                    if (data.assistant_text && data.audio_base64) {
                        addMessage("assistant", data.assistant_text);
                        setOrbState("speaking", "Shree is speaking...");
                        
                        // Play MP3 audio directly in Computer A's speakers
                        const audioBytes = atob(data.audio_base64);
                        const arrayBuffer = new ArrayBuffer(audioBytes.length);
                        const bufferView = new Uint8Array(arrayBuffer);
                        for (let i = 0; i < audioBytes.length; i++) {
                            bufferView[i] = audioBytes.charCodeAt(i);
                        }

                        const audioBlob = new Blob([arrayBuffer], { type: 'audio/mp3' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        currentAudioPlayer = new Audio(audioUrl);

                        currentAudioPlayer.onended = () => {
                            if (isCalling) setOrbState("listening", "Listening... Speak naturally");
                        };

                        await currentAudioPlayer.play();
                    } else {
                        if (isCalling) setOrbState("listening", "Listening... Speak naturally");
                    }

                } catch (err) {
                    console.error("Voice process error:", err);
                    if (isCalling) setOrbState("listening", "Listening... Speak naturally");
                }
            }
        </script>
    </body>
    </html>
    """

    components.html(voice_call_html, height=580)

with tab_enroll:
    st.subheader("🔐 Register / Update Voice Profile")
    st.write("Record each sentence clearly using the microphone widgets below, then click Save.")

    st.markdown("1️⃣ **Sentence 1 of 3:**")
    st.info("👉 *'My voice is my password, verify me. I am speaking naturally so the system can learn my voice properly.'*")
    s1 = st.audio_input("Record Sentence 1", key="enroll_s1")

    st.markdown("2️⃣ **Sentence 2 of 3:**")
    st.info("👉 *'The quick brown fox jumps over the lazy dog while the sun sets slowly on a calm evening.'*")
    s2 = st.audio_input("Record Sentence 2", key="enroll_s2")

    st.markdown("3️⃣ **Sentence 3 of 3:**")
    st.info("👉 *'I am the authorized user of this system, providing my natural speaking voice for accurate enrollment.'*")
    s3 = st.audio_input("Record Sentence 3", key="enroll_s3")

    if st.button("💾 Save & Register Voice Profile", type="primary", use_container_width=True):
        if not s1 or not s2 or not s3:
            st.warning("⚠️ Please record all 3 sentences before saving your profile.")
        else:
            with st.spinner("Processing voice fingerprint..."):
                try:
                    samples = [decode_audio_file(s1), decode_audio_file(s2), decode_audio_file(s3)]
                    verifier_engine.enroll(samples)
                    st.success("✅ Voice profile registered successfully! You can now start the Live Voice Call.")
                except Exception as e:
                    st.error(f"Error registering voice: {e}")






