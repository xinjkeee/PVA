from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.audio.audio_io import load_audio
from src.audio.transforms import TARGET_SR, resample_audio


def plot_waveform(ax, audio, sr, title: str):
    times = librosa.times_like(audio, sr=sr)
    ax.plot(times, audio)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)


def plot_stft(ax, audio, sr, title: str, n_fft: int = 2048, hop_length: int = 512):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    img = librosa.display.specshow(
        stft_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
    )
    ax.set_title(title)
    return img


def plot_mel(ax, audio, sr, title: str, n_mels: int = 80, n_fft: int = 2048, hop_length: int = 512):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    ax.set_title(title)
    return img


def visualize_before_after(before_path: str, after_path: str, sample_num: int, save_path: str | None = None):
    before_audio, before_sr = load_audio(before_path)
    after_audio, after_sr = load_audio(after_path)

    if before_sr != TARGET_SR:
        before_audio = resample_audio(before_audio, orig_sr=before_sr, target_sr=TARGET_SR)
        before_sr = TARGET_SR
    if after_sr != TARGET_SR:
        after_audio = resample_audio(after_audio, orig_sr=after_sr, target_sr=TARGET_SR)
        after_sr = TARGET_SR

    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle(f"Sample {sample_num}", fontsize=18)

    plot_waveform(axes[0, 0], before_audio, before_sr, f"Sample {sample_num} — Before — Waveform")
    plot_waveform(axes[0, 1], after_audio, after_sr, f"Sample {sample_num} — After — Waveform")

    img1 = plot_stft(axes[1, 0], before_audio, before_sr, f"Sample {sample_num} — Before — STFT")
    img2 = plot_stft(axes[1, 1], after_audio, after_sr, f"Sample {sample_num} — After — STFT")

    img3 = plot_mel(axes[2, 0], before_audio, before_sr, f"Sample {sample_num} — Before — Mel")
    img4 = plot_mel(axes[2, 1], after_audio, after_sr, f"Sample {sample_num} — After — Mel")

    fig.colorbar(img1, ax=axes[1, 0], format="%+2.0f dB")
    fig.colorbar(img2, ax=axes[1, 1], format="%+2.0f dB")
    fig.colorbar(img3, ax=axes[2, 0], format="%+2.0f dB")
    fig.colorbar(img4, ax=axes[2, 1], format="%+2.0f dB")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)
