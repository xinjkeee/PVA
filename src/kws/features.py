import math

import numpy as np
import torch


DEFAULT_FEATURE_CONFIG = {
    "sample_rate": 16000,
    "duration_sec": 1.5,
    "n_fft": 400,
    "hop_length": 160,
    "n_mels": 40,
    "f_min": 80.0,
    "f_max": 7600.0,
}


def fit_audio_length(audio: np.ndarray, sample_rate: int, duration_sec: float) -> np.ndarray:
    target_len = int(sample_rate * duration_sec)
    if len(audio) == target_len:
        return audio.astype(np.float32)
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio))).astype(np.float32)

    start = max(0, (len(audio) - target_len) // 2)
    return audio[start : start + target_len].astype(np.float32)


def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    device: torch.device,
) -> torch.Tensor:
    min_mel = hz_to_mel(torch.tensor(f_min, device=device))
    max_mel = hz_to_mel(torch.tensor(f_max, device=device))
    mel_points = torch.linspace(min_mel, max_mel, n_mels + 2, device=device)
    hz_points = mel_to_hz(mel_points)
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    bins = torch.clamp(bins, min=0, max=n_fft // 2)

    filters = torch.zeros(n_mels, n_fft // 2 + 1, device=device)
    for mel_idx in range(n_mels):
        left = int(bins[mel_idx].item())
        center = int(bins[mel_idx + 1].item())
        right = int(bins[mel_idx + 2].item())

        if center > left:
            filters[mel_idx, left:center] = torch.linspace(0.0, 1.0, center - left, device=device)
        if right > center:
            filters[mel_idx, center:right] = torch.linspace(1.0, 0.0, right - center, device=device)

    return filters


def audio_to_log_mel(audio: np.ndarray, config: dict) -> torch.Tensor:
    sample_rate = int(config["sample_rate"])
    duration_sec = float(config["duration_sec"])
    n_fft = int(config["n_fft"])
    hop_length = int(config["hop_length"])
    n_mels = int(config["n_mels"])
    f_min = float(config["f_min"])
    f_max = float(config["f_max"])

    fitted = fit_audio_length(audio, sample_rate=sample_rate, duration_sec=duration_sec)
    waveform = torch.from_numpy(fitted)
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    power = stft.abs().pow(2)
    filters = mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max, waveform.device)
    mel = filters @ power
    log_mel = torch.log(torch.clamp(mel, min=1e-6))
    return (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)


def expected_frames(config: dict) -> int:
    sample_rate = int(config["sample_rate"])
    duration_sec = float(config["duration_sec"])
    hop_length = int(config["hop_length"])
    samples = int(sample_rate * duration_sec)
    return 1 + math.floor(samples / hop_length)
