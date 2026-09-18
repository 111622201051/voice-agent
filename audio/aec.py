# audio/aec.py
# Acoustic Echo Cancellation (AEC) for full-duplex voice.
#
# Removes the AI's own speaker output ("reference") from the microphone input
# so that the VAD / speaker-verification stage hears only the human caller.
# Uses a classic normalized least-mean-squares (NLMS) adaptive filter.
import numpy as np


class NLMSAcousticEchoCanceller:
    """Real-time AEC via an NLMS adaptive FIR filter.

    `process(mic, ref)` returns the echo-suppressed mic signal.
    - mic:  float32 1-D chunk captured from the microphone (K samples)
    - ref:  the audio currently played on the speakers, same length
    Both arrays must have the same sample rate (use 16 kHz).

    Feed the SAME rolling reference window the TTS engine plays; the filter
    then subtracts the echo component before downstream processing.
    """

    def __init__(self, filter_len: int = 2048, mu: float = 0.05):
        self.filter_len = filter_len
        self.mu = mu
        self.w = np.zeros(filter_len, dtype=np.float32)
        self._state = np.zeros(filter_len, dtype=np.float32)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float32).flatten()
        ref = np.asarray(ref, dtype=np.float32).flatten()
        n = max(len(mic), len(ref))
        mic_p = np.pad(mic, (0, n - len(mic)))
        ref_p = np.pad(ref, (0, n - len(ref)))

        out = np.empty(n, dtype=np.float32)
        w = self.w
        state = self._state.copy()
        mu = self.mu

        for i in range(n):
            state[1:] = state[:-1]
            state[0] = ref_p[i]
            est = float(np.dot(state, w))
            err = mic_p[i] - est
            power = float(np.dot(state, state)) + 1e-10
            w += (mu * err / power) * state
            out[i] = err

        self.w = w
        self._state = state
        return out

    def reset(self):
        self.w.fill(0.0)
        self._state.fill(0.0)


class IdealAEC:
    """Full-suppression variant: mutes the mic signal while audio plays.

    Useful as a lightweight fallback when the reference signal is unknown
    (e.g. the AI is definitely speaking => treat mic as echo only).
    """

    def __init__(self, noise_floor: float = 1e-4):
        self.noise_floor = noise_floor

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        mic = np.asarray(mic, dtype=np.float32).flatten()
        ref = np.asarray(ref, dtype=np.float32).flatten()
        active = float(np.max(np.abs(ref))) > self.noise_floor
        if active:
            return np.zeros_like(mic)
        return mic


def create_aec(mode: str = "nllms", filter_len: int = 2048):
    """Factory: returns an AEC instance ('nlms' | 'ideal')."""
    mode = (mode or "nlms").lower()
    if mode == "ideal":
        return IdealAEC()
    return NLMSAcousticEchoCanceller(filter_len=filter_len)