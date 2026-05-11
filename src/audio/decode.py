from pathlib import Path

import numpy as np
from src.audio.audio_io import load_audio as load_audio_librosa
from src.audio.transforms import resample_audio


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    audio, sr = load_audio_librosa(str(path))
    if sr != sample_rate:
        audio = resample_audio(audio, orig_sr=sr, target_sr=sample_rate)
    return audio.astype(np.float32)
