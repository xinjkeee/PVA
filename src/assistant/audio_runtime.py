import wave
from pathlib import Path

import numpy as np


def save_wav_pcm16(path: Path | str, audio: np.ndarray, sample_rate: int) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    return output_path


def record_microphone(
    output_path: Path | str,
    seconds: float = 5.0,
    sample_rate: int = 16000,
) -> Path:
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Microphone recording requires the optional 'sounddevice' package. "
            "Install project requirements and try again."
        ) from exc

    frames = int(round(seconds * sample_rate))
    recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return save_wav_pcm16(output_path, recording.reshape(-1), sample_rate)
