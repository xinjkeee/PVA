from pathlib import Path

import librosa
import soundfile as sf


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def load_audio(path: str):
    audio, sr = librosa.load(path, sr=None, mono=True)
    return audio, sr


def save_audio(path: str, audio, sr: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def find_audio_files(input_dir: str | Path) -> list[Path]:
    input_path = Path(input_dir)
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
