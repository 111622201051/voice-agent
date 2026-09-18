# audio/capture.py
# Reusable microphone capture utilities (16 kHz mono float32 PCM).
import sounddevice as sd
import numpy as np


class MicStream:
    """Background-safe microphone input wrapper.

    Yields flattened float32 mono chunks (default 512 samples @ 16 kHz)
    matching the Silero VAD window. Safe to use as a context manager:

        with MicStream() as mic:
            for chunk in mic:
                ...
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_size: int = 512, dtype: str = 'float32',
                 device: int = None, blocksize: int = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.device = device
        self.blocksize = blocksize or chunk_size
        self.stream = None

    def __enter__(self) -> "MicStream":
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device,
                blocksize=self.blocksize,
            )
            self.stream.start()
        except Exception as e:
            raise RuntimeError(f"Could not open microphone stream: {e}") from e
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def read_chunk(self) -> np.ndarray:
        """Reads a single chunk as a 1-D float32 array."""
        if self.stream is None:
            raise RuntimeError("MicStream not started (use it as a context manager).")
        chunk, _overflowed = self.stream.read(self.chunk_size)
        return np.asarray(chunk, dtype=np.float32).flatten()

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.stream is None:
            raise StopIteration
        return self.read_chunk()

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    @property
    def running(self) -> bool:
        return self.stream is not None


def capture_samples(duration_sec: float, sample_rate: int = 16000, block: int = 512) -> np.ndarray:
    """Records `duration_sec` seconds of audio and returns a 1-D float32 array."""
    frames = int(duration_sec * sample_rate)
    recording = sd.rec(frames, samplerate=sample_rate, channels=1,
                       dtype='float32', blocking=True)
    return np.asarray(recording, dtype=np.float32).flatten()