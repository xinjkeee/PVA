from pathlib import Path

import av
import numpy as np


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    try:
        from src.audio.audio_io import load_audio as load_audio_librosa
        from src.audio.transforms import resample_audio

        audio, sr = load_audio_librosa(str(path))
        if sr != sample_rate:
            audio = resample_audio(audio, orig_sr=sr, target_sr=sample_rate)
        return audio.astype(np.float32)
    except ModuleNotFoundError as exc:
        if exc.name not in {"librosa", "soundfile"}:
            raise
        return load_audio_av(path, sample_rate=sample_rate)


def load_audio_av(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    container = av.open(str(path))
    try:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise RuntimeError(f"No audio stream found: {path}")

        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=sample_rate,
        )

        chunks = []
        for frame in container.decode(audio_stream):
            resampled = resampler.resample(frame)
            if resampled is None:
                continue
            if not isinstance(resampled, list):
                resampled = [resampled]

            for audio_frame in resampled:
                samples = audio_frame.to_ndarray()
                chunks.append(samples.reshape(-1).astype(np.float32) / 32768.0)

        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(chunks).astype(np.float32)
    finally:
        container.close()
