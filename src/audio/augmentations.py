import librosa
import numpy as np
from scipy.signal import butter, lfilter, sosfilt


def apply_augmentations(audio: np.ndarray, sr: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    original_len = len(audio)
    audio = np.asarray(audio, dtype=np.float32)
    if original_len == 0:
        return audio

    if rng.random() < 0.55:
        audio = add_noise(audio, snr_db=float(rng.uniform(10.0, 20.0)), rng=rng)

    if rng.random() < 0.45:
        audio = time_stretch(audio, speed_factor=float(rng.uniform(0.96, 1.04)), target_len=original_len)

    if rng.random() < 0.35:
        n_steps = int(rng.choice([-2, -1, 1, 2]))
        audio = change_pitch(audio, sr, n_steps=n_steps)

    if rng.random() < 0.35:
        audio = time_shift(audio, max_shift=int(0.08 * sr), rng=rng)

    # if rng.random() < 0.30:
    #     audio = low_pass(audio, sr, cutoff=float(rng.uniform(2800.0, 5200.0)))

    # if rng.random() < 0.20:
    #     audio = high_pass(audio, sr, cutoff=float(rng.uniform(80.0, 220.0)))

    if rng.random() < 0.30:
        audio = change_gain(audio, gain_db=float(rng.uniform(-6.0, 4.0)))

    if rng.random() < 0.25:
        audio = dynamic_range_compression(audio)

    if rng.random() < 0.30:
        audio = parametric_eq(
            audio,
            sr,
            peak_freq=float(rng.uniform(200.0, 6000.0)),
            peak_gain_db=float(rng.uniform(-8.0, 8.0)),
            peak_q=float(rng.uniform(0.5, 3.0)),
            low_shelf_gain_db=float(rng.uniform(-4.0, 4.0)),
            high_shelf_gain_db=float(rng.uniform(-4.0, 4.0)),
        )

    return prevent_clipping(fix_length(audio, original_len)).astype(np.float32)


def fix_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio[:target_len]


def add_noise(audio: np.ndarray, snr_db: float = 24.0, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, 1.0, size=len(audio)).astype(np.float32)
    audio_rms = np.sqrt(np.mean(np.square(audio))) + 1e-8
    noise_rms = np.sqrt(np.mean(np.square(noise))) + 1e-8
    target_noise_rms = audio_rms / (10 ** (snr_db / 20.0))
    return audio + noise * (target_noise_rms / noise_rms)


def time_stretch(audio: np.ndarray, speed_factor: float = 1.0, target_len: int | None = None) -> np.ndarray:
    stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
    if target_len is None:
        return stretched

    return fix_length(stretched, target_len)


def change_pitch(
    audio: np.ndarray,
    sr: int,
    n_steps: int | None = None,
    max_steps: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if n_steps is None:
        rng = rng or np.random.default_rng()
        n_steps = int(rng.integers(-max_steps, max_steps + 1))

    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    return prevent_clipping(shifted)


def time_shift(audio: np.ndarray, max_shift: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if max_shift <= 0:
        return audio

    rng = rng or np.random.default_rng()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return audio

    shifted = np.zeros_like(audio)
    if shift > 0:
        shifted[shift:] = audio[:-shift]
    else:
        shifted[:shift] = audio[-shift:]
    return shifted


def low_pass(audio: np.ndarray, sr: int, cutoff: float = 3000.0) -> np.ndarray:
    nqst = sr // 2
    norm_cutoff = cutoff / nqst

    b, a = butter(4, norm_cutoff, btype="low")

    return lfilter(b, a, audio)


def high_pass(audio: np.ndarray, sr: int, cutoff: float = 300.0) -> np.ndarray:
    nqst = sr // 2
    norm_cutoff = cutoff / nqst

    b, a = butter(4, norm_cutoff, btype="high")

    return lfilter(b, a, audio)


def _peak_biquad(sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    w0 = 2.0 * np.pi * freq / sr 
    A = 10.0 ** (gain_db / 40.0) 
    alpha = np.sin(w0) / (2.0 * max(q, 0.1))
    cos_w0 = np.cos(w0)
    b0 =  1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 =  1.0 - alpha * A
    a0 =  1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 =  1.0 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _low_shelf_biquad(sr: int, freq: float, gain_db: float) -> np.ndarray:
    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    w0 = 2.0 * np.pi * freq / sr
    A = 10.0 ** (gain_db / 40.0)
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) * np.sqrt(0.5)
    t = 2.0 * np.sqrt(A) * alpha
    b0 =  A * ((A + 1) - (A - 1) * cos_w0 + t)
    b1 =  2.0 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 =  A * ((A + 1) - (A - 1) * cos_w0 - t)
    a0 =      (A + 1) + (A - 1) * cos_w0 + t
    a1 = -2.0 * ((A - 1) + (A + 1) * cos_w0)
    a2 =      (A + 1) + (A - 1) * cos_w0 - t
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _high_shelf_biquad(sr: int, freq: float, gain_db: float) -> np.ndarray:
    freq = float(np.clip(freq, 20.0, sr / 2.0 - 1.0))
    w0 = 2.0 * np.pi * freq / sr
    A = 10.0 ** (gain_db / 40.0)
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) * np.sqrt(0.5)
    t = 2.0 * np.sqrt(A) * alpha
    b0 =  A * ((A + 1) + (A - 1) * cos_w0 + t)
    b1 = -2.0 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 =  A * ((A + 1) + (A - 1) * cos_w0 - t)
    a0 =      (A + 1) - (A - 1) * cos_w0 + t
    a1 =  2.0 * ((A - 1) - (A + 1) * cos_w0)
    a2 =      (A + 1) - (A - 1) * cos_w0 - t
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def parametric_eq(
    audio: np.ndarray,
    sr: int,
    peak_freq: float = 1000.0,
    peak_gain_db: float = 0.0,
    peak_q: float = 1.0,
    low_shelf_freq: float = 200.0,
    low_shelf_gain_db: float = 0.0,
    high_shelf_freq: float = 4000.0,
    high_shelf_gain_db: float = 0.0,
) -> np.ndarray:
    sections = []
    if peak_gain_db != 0.0:
        sections.append(_peak_biquad(sr, peak_freq, peak_gain_db, peak_q))
    if low_shelf_gain_db != 0.0:
        sections.append(_low_shelf_biquad(sr, low_shelf_freq, low_shelf_gain_db))
    if high_shelf_gain_db != 0.0:
        sections.append(_high_shelf_biquad(sr, high_shelf_freq, high_shelf_gain_db))
    if not sections:
        return audio
    sos = np.concatenate(sections, axis=0)
    return sosfilt(sos, audio).astype(np.float32)


def change_gain(audio: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
    return audio * (10 ** (gain_db / 20.0))


def dynamic_range_compression(audio: np.ndarray, threshold: float = 0.5, ratio: float = 4.0) -> np.ndarray:
    compressed = audio.copy()

    mask = np.abs(audio) > threshold
    compressed[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)

    return compressed


def prevent_clipping(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    if len(audio) == 0:
        return audio

    max_val = np.max(np.abs(audio))
    if max_val > peak:
        audio = audio * (peak / max_val)
    return audio
