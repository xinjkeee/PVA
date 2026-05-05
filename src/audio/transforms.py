import numpy as np
import librosa
from .augmentations import apply_augmentations


TARGET_SR = 16000


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    if len(audio) == 0:
        return audio
    return audio - np.mean(audio)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if len(audio) == 0:
        return audio

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def preprocess_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    audio = remove_dc_offset(audio)
    audio = resample_audio(audio, orig_sr=orig_sr, target_sr=target_sr)
    audio = normalize_audio(audio)
    return audio.astype(np.float32), target_sr


def process_audio_with_augmentations(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = TARGET_SR,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    audio, target_sr = preprocess_audio(audio, orig_sr, target_sr)
    if augment:
        audio = apply_augmentations(audio, target_sr, rng=rng)
        audio = normalize_audio(audio)
    return audio.astype(np.float32), target_sr
