# ui/bridge.py
import threading


class UIBridge:
    def __init__(self):
        self.lock = threading.Lock()
        self.messages = []
        self.status = "Idle"
        self.is_running = False
        self.latest_audio_bytes = None
        self.audio_id = 0

    def add_message(self, role: str, content: str, audio_bytes: bytes = None):
        """Adds a chat message safely from any thread with optional audio stream for browser."""
        with self.lock:
            self.messages.append({"role": role, "content": content})
            if audio_bytes:
                self.latest_audio_bytes = audio_bytes
                self.audio_id += 1

    def set_status(self, status: str):
        """Updates the current AI status (e.g., 'Listening', 'Thinking')."""
        with self.lock:
            self.status = status

    def clear_chat(self):
        """Clears the chat history and audio buffer."""
        with self.lock:
            self.messages = []
            self.latest_audio_bytes = None
            self.audio_id = 0

    def get_state(self):
        """Returns a thread-safe copy of the current UI state."""
        with self.lock:
            return {
                "messages": list(self.messages),
                "status": self.status,
                "is_running": self.is_running,
                "latest_audio_bytes": self.latest_audio_bytes,
                "audio_id": self.audio_id
            }