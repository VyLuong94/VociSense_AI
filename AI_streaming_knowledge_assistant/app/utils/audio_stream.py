import numpy as np

def bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    try:
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        if np.max(np.abs(audio)) <= 1.5:
            return audio
    except:
        pass

    audio = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0

def normalize_audio(audio: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    peak = np.max(np.abs(audio)) + eps
    return audio / peak