from pathlib import Path
from typing import Any

import numpy as np


WHISPER_SAMPLE_RATE = 16000


class WhisperASR:
    """Lazy wrapper around OpenAI Whisper for one-off and GUI inference."""

    def __init__(
        self,
        model_name: str = "tiny",
        download_root: Path | str = "./data/models/whisper",
        language: str = "ru",
        fp16: bool = False,
    ) -> None:
        self.model_name = model_name
        self.download_root = Path(download_root)
        self.language = language
        self.fp16 = fp16
        self.sample_rate = WHISPER_SAMPLE_RATE
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            import whisper

            self.download_root.mkdir(parents=True, exist_ok=True)
            self._model = whisper.load_model(
                self.model_name,
                download_root=str(self.download_root),
            )
        return self._model

    def transcribe_audio(self, audio: np.ndarray) -> str:
        model = self._load_model()
        result = model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.language,
            task="transcribe",
            fp16=self.fp16,
        )
        return result.get("text", "").strip()

    def transcribe_file(self, audio_path: Path | str) -> str:
        from src.audio.decode import load_audio

        audio = load_audio(audio_path, sample_rate=self.sample_rate)
        return self.transcribe_audio(audio)
