from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.asr.whisper_runner import WhisperASR
from src.assistant.audio_runtime import record_microphone, save_wav_pcm16
from src.assistant.nlu import interpret_text
from src.audio.vad import energy_vad
from src.tts.system import play_audio_file, synthesize_with_macos_say


@dataclass
class AssistantConfig:
    sample_rate: int = 16000
    artifacts_dir: Path = Path("./data/assistant")
    speech_output_name: str = "last_speech.wav"
    microphone_output_name: str = "last_microphone.wav"
    vad_frame_ms: float = 30.0
    vad_hop_ms: float = 10.0
    vad_margin_db: float = 10.0
    vad_min_speech_ms: float = 200.0
    vad_pad_ms: float = 100.0
    kws_checkpoint: Path | None = Path("./data/kws/checkpoints/привет_cnn.pt")
    kws_threshold: float = 0.5
    require_wake_word: bool = False
    device: str = "auto"
    whisper_model: str = "tiny"
    whisper_download_root: Path = Path("./data/models/whisper")
    language: str = "ru"
    fp16: bool = False
    tts_output: Path = Path("./data/tts/assistant_response.aiff")
    tts_voice: str | None = None
    tts_rate: int | None = None
    synthesize_response: bool = True
    play_response: bool = True


@dataclass
class AssistantResult:
    status: str
    source_audio_path: str
    speech_audio_path: str | None = None
    vad_segments: list[tuple[float, float]] = field(default_factory=list)
    vad_threshold_db: float | None = None
    vad_speech_seconds: float = 0.0
    kws_available: bool = False
    kws_probability: float | None = None
    wake: bool | None = None
    transcript: str = ""
    intent: str = ""
    nlu_confidence: float = 0.0
    response_text: str = ""
    tts_path: str | None = None
    played: bool = False
    errors: list[str] = field(default_factory=list)

    def stage_rows(self) -> list[tuple[str, str, str]]:
        kws_detail = "checkpoint недоступен"
        if self.kws_probability is not None:
            kws_detail = f"p={self.kws_probability:.3f}, wake={self.wake}"

        return [
            (
                "VAD",
                "ok" if self.vad_segments else "empty",
                f"segments={len(self.vad_segments)}, speech={self.vad_speech_seconds:.2f}s",
            ),
            ("KWS", "ok" if self.kws_available else "skip", kws_detail),
            ("ASR", "ok" if self.transcript else "skip", self.transcript),
            (
                "NLU",
                "ok" if self.intent else "skip",
                f"{self.intent} ({self.nlu_confidence:.2f})" if self.intent else "",
            ),
            ("TTS", "ok" if self.tts_path else "skip", self.tts_path or ""),
            ("Speaker", "ok" if self.played else "skip", "played" if self.played else ""),
        ]


class AssistantPipeline:
    def __init__(self, config: AssistantConfig | None = None) -> None:
        self.config = config or AssistantConfig()
        self._asr: WhisperASR | None = None

    @property
    def asr(self) -> WhisperASR:
        if self._asr is None:
            self._asr = WhisperASR(
                model_name=self.config.whisper_model,
                download_root=self.config.whisper_download_root,
                language=self.config.language,
                fp16=self.config.fp16,
            )
        return self._asr

    def run_microphone(self, seconds: float = 5.0) -> AssistantResult:
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.artifacts_dir / self.config.microphone_output_name
        recorded_path = record_microphone(
            output_path=output_path,
            seconds=seconds,
            sample_rate=self.config.sample_rate,
        )
        return self.run_file(recorded_path)

    def run_file(self, audio_path: Path | str) -> AssistantResult:
        path = Path(audio_path)
        result = AssistantResult(status="started", source_audio_path=str(path))

        try:
            from src.audio.decode import load_audio

            audio = load_audio(path, sample_rate=self.config.sample_rate)
        except Exception as exc:
            result.status = "audio_error"
            result.errors.append(f"Audio decode failed: {exc}")
            return result

        speech_audio = self._apply_vad(audio, result)
        if len(speech_audio) == 0:
            result.status = "no_speech"
            result.response_text = "Я не услышала речь."
            self._try_tts(result)
            return result

        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        speech_path = self.config.artifacts_dir / self.config.speech_output_name
        save_wav_pcm16(speech_path, speech_audio, self.config.sample_rate)
        result.speech_audio_path = str(speech_path)

        self._try_kws(result, speech_path)
        if (
            self.config.require_wake_word
            and result.kws_available
            and result.wake is False
        ):
            result.status = "wake_not_detected"
            result.response_text = "Wake word не распознан, команда пропущена."
            self._try_tts(result)
            return result

        if self.config.require_wake_word and not result.kws_available:
            result.status = "kws_unavailable"
            result.response_text = "KWS-модель недоступна, поэтому wake word не проверен."
            self._try_tts(result)
            return result

        try:
            result.transcript = self.asr.transcribe_audio(speech_audio)
        except Exception as exc:
            result.status = "asr_error"
            result.errors.append(f"ASR failed: {exc}")
            result.response_text = "ASR сейчас недоступен, но VAD и KWS уже отработали."
            self._try_tts(result)
            return result

        nlu = interpret_text(result.transcript)
        result.intent = nlu.intent
        result.nlu_confidence = nlu.confidence
        result.response_text = nlu.response_text
        result.status = "ok"
        self._try_tts(result)
        return result

    def _apply_vad(self, audio: np.ndarray, result: AssistantResult) -> np.ndarray:
        speech_mask, segments, threshold_db = energy_vad(
            audio,
            sr=self.config.sample_rate,
            frame_ms=self.config.vad_frame_ms,
            hop_ms=self.config.vad_hop_ms,
            vad_margin_db=self.config.vad_margin_db,
            min_speech_ms=self.config.vad_min_speech_ms,
            pad_ms=self.config.vad_pad_ms,
        )
        result.vad_segments = segments
        result.vad_threshold_db = threshold_db
        result.vad_speech_seconds = sum(end - start for start, end in segments)

        if len(speech_mask) == 0 or not segments:
            return np.array([], dtype=np.float32)

        pieces = []
        for start_sec, end_sec in segments:
            start_sample = max(0, int(round(start_sec * self.config.sample_rate)))
            end_sample = min(len(audio), int(round(end_sec * self.config.sample_rate)))
            if start_sample < end_sample:
                pieces.append(audio[start_sample:end_sample])

        if not pieces:
            return np.array([], dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32)

    def _try_kws(self, result: AssistantResult, speech_path: Path) -> None:
        checkpoint = self.config.kws_checkpoint
        if checkpoint is None:
            result.errors.append("KWS checkpoint is not configured.")
            return
        if not checkpoint.exists():
            result.errors.append(f"KWS checkpoint not found: {checkpoint}")
            return

        try:
            from src.kws.pipeline import detect_keyword

            kws_result = detect_keyword(
                audio_path=speech_path,
                checkpoint_path=checkpoint,
                threshold=self.config.kws_threshold,
                device=self.config.device,
            )
        except Exception as exc:
            result.errors.append(f"KWS failed: {exc}")
            return

        result.kws_available = True
        result.kws_probability = float(kws_result["probability"])
        result.wake = bool(kws_result["wake"])

    def _try_tts(self, result: AssistantResult) -> None:
        if not result.response_text or not self.config.synthesize_response:
            return

        try:
            tts_path = synthesize_with_macos_say(
                text=result.response_text,
                output_path=self.config.tts_output,
                voice=self.config.tts_voice,
                rate=self.config.tts_rate,
            )
            result.tts_path = str(tts_path)
        except Exception as exc:
            result.errors.append(f"TTS failed: {exc}")
            return

        if not self.config.play_response:
            return

        try:
            result.played = play_audio_file(tts_path)
        except Exception as exc:
            result.errors.append(f"Playback failed: {exc}")
