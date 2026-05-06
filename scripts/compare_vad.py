from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.audio.audio_io import load_audio
from src.audio.transforms import TARGET_SR, resample_audio
from src.audio.vad import energy_vad
from src.audio.vad_webrtc import webrtc_vad, webrtc_vad_segments


def _segment_total_duration(segments: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in segments)


def _plot_waveform(ax, audio: np.ndarray, sr: int, title: str) -> None:
    times = np.arange(len(audio), dtype=np.float32) / sr
    ax.plot(times, audio, color="steelblue", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)


def _plot_segments(ax, segments: list[tuple[float, float]], color: str, label: str) -> None:
    for idx, (start, end) in enumerate(segments):
        ax.axvspan(start, end, color=color, alpha=0.25, label=label if idx == 0 else None)


def _plot_mask(ax, mask: np.ndarray, frame_ms: float, title: str, color: str) -> None:
    if len(mask) == 0:
        ax.set_title(title)
        ax.set_ylabel("Speech")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.25)
        return

    frame_times = np.arange(len(mask), dtype=np.float32) * (frame_ms / 1000.0)
    ax.step(frame_times, mask.astype(int), where="post", color=color, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("Speech")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.25)


def compare_vad(
    audio_path: str,
    save_path: str | None = None,
    energy_frame_ms: float = 30.0,
    energy_hop_ms: float = 10.0,
    webrtc_frame_ms: int = 30,
    webrtc_mode: int = 2,
) -> dict[str, object]:
    audio, sr = load_audio(audio_path)
    if sr != TARGET_SR:
        audio = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    energy_mask, energy_segments, threshold_db = energy_vad(
        audio,
        sr=sr,
        frame_ms=energy_frame_ms,
        hop_ms=energy_hop_ms,
    )
    webrtc_mask = webrtc_vad(
        audio,
        sr=sr,
        mode=webrtc_mode,
        frame_ms=webrtc_frame_ms,
    )
    webrtc_segments = webrtc_vad_segments(
        audio,
        sr=sr,
        mode=webrtc_mode,
        frame_ms=webrtc_frame_ms,
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    file_name = Path(audio_path).name
    fig.suptitle(f"VAD comparison: {file_name}", fontsize=16)

    _plot_waveform(axes[0], audio, sr, "Waveform with detected speech regions")
    _plot_segments(axes[0], energy_segments, color="tab:green", label="Energy VAD")
    _plot_segments(axes[0], webrtc_segments, color="tab:orange", label="WebRTC VAD")
    axes[0].legend(loc="upper right")

    _plot_mask(
        axes[1],
        energy_mask,
        frame_ms=energy_hop_ms,
        title=f"Energy VAD mask (threshold {threshold_db:.1f} dB)",
        color="tab:green",
    )
    _plot_mask(
        axes[2],
        webrtc_mask,
        frame_ms=webrtc_frame_ms,
        title=f"WebRTC VAD mask (mode {webrtc_mode}, frame {webrtc_frame_ms} ms)",
        color="tab:orange",
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_xlim(0.0, len(audio) / sr)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "audio_seconds": len(audio) / sr,
        "energy_segments": energy_segments,
        "energy_speech_seconds": _segment_total_duration(energy_segments),
        "energy_threshold_db": threshold_db,
        "webrtc_segments": webrtc_segments,
        "webrtc_speech_seconds": _segment_total_duration(webrtc_segments),
    }
