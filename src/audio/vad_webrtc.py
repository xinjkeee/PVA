import numpy as np
import webrtcvad

from .vad import collect_speech_segments, frames_to_seconds


VALID_SAMPLE_RATES = {8000, 16000, 32000, 48000}
VALID_FRAME_MS = {10, 20, 30}


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def validate_webrtc_params(sr: int, mode: int, frame_ms: int) -> None:
    if sr not in VALID_SAMPLE_RATES:
        raise ValueError(f"WebRTC VAD supports sample rates: {sorted(VALID_SAMPLE_RATES)}")
    if frame_ms not in VALID_FRAME_MS:
        raise ValueError(f"WebRTC VAD supports frame sizes: {sorted(VALID_FRAME_MS)} ms")
    if mode not in {0, 1, 2, 3}:
        raise ValueError("WebRTC VAD mode must be 0, 1, 2, or 3")


def frame_generator(audio: np.ndarray, sr: int, frame_ms: int = 30):
    frame_len = int(sr * frame_ms / 1000)

    for i in range(0, len(audio) - frame_len + 1, frame_len):
        yield audio[i : i + frame_len]


def webrtc_vad(audio: np.ndarray, sr: int = 16000, mode: int = 2, frame_ms: int = 30) -> np.ndarray:
    validate_webrtc_params(sr, mode, frame_ms)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return np.array([], dtype=bool)

    vad = webrtcvad.Vad(mode)
    audio_int16 = float_to_int16(audio)
    return np.array(
        [vad.is_speech(frame.tobytes(), sr) for frame in frame_generator(audio_int16, sr, frame_ms)],
        dtype=bool,
    )


def webrtc_vad_segments(
    audio: np.ndarray,
    sr: int = 16000,
    mode: int = 2,
    frame_ms: int = 30,
    min_speech_ms: float = 200.0,
    pad_ms: float = 100.0,
) -> list[tuple[float, float]]:
    
    speech_mask = webrtc_vad(audio, sr=sr, mode=mode, frame_ms=frame_ms)
    frame_len = int(sr * frame_ms / 1000)
    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    pad_frames = max(0, int(pad_ms / frame_ms))

    frame_segments = collect_speech_segments(speech_mask, min_speech_frames, pad_frames)
    return frames_to_seconds(frame_segments, frame_len, frame_len, sr, len(audio))


def trim_silence_webrtc(
    audio: np.ndarray,
    sr: int = 16000,
    mode: int = 2,
    frame_ms: int = 30,
    min_speech_ms: float = 200.0,
    pad_ms: float = 100.0,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    segments = webrtc_vad_segments(
        audio,
        sr=sr,
        mode=mode,
        frame_ms=frame_ms,
        min_speech_ms=min_speech_ms,
        pad_ms=pad_ms,
    )

    pieces = []
    for start_sec, end_sec in segments:
        start_sample = max(0, int(round(start_sec * sr)))
        end_sample = min(len(audio), int(round(end_sec * sr)))
        if start_sample < end_sample:
            pieces.append(audio[start_sample:end_sample])

    if not pieces:
        return np.array([], dtype=np.float32)

    return np.concatenate(pieces).astype(np.float32)
