import numpy as np


def frame_audio(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return np.empty((0, frame_length), dtype=np.float32)

    if len(audio) < frame_length:
        audio = np.pad(audio, (0, frame_length - len(audio)))
    else:
        remainder = (len(audio) - frame_length) % hop_length
        if remainder:
            audio = np.pad(audio, (0, hop_length - remainder))

    num_frames = 1 + (len(audio) - frame_length) // hop_length
    return np.stack([
        audio[i * hop_length : i * hop_length + frame_length]
        for i in range(num_frames)
    ])


def frame_rms(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)


def to_db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(x, 1e-12))


def collect_speech_segments(
    speech_mask: np.ndarray,
    min_speech_frames: int,
    pad_frames: int = 0,
) -> list[tuple[int, int]]:
    segments = []
    in_speech = False
    start = 0

    for i, is_speech in enumerate(speech_mask):
        if is_speech and not in_speech:
            in_speech = True
            start = i
        elif not is_speech and in_speech:
            end = i
            if end - start >= min_speech_frames:
                segments.append((
                    max(0, start - pad_frames),
                    min(len(speech_mask), end + pad_frames),
                ))
            in_speech = False

    if in_speech:
        end = len(speech_mask)
        if end - start >= min_speech_frames:
            segments.append((
                max(0, start - pad_frames),
                min(len(speech_mask), end + pad_frames),
            ))

    return merge_segments(segments)


def merge_segments(segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def frames_to_seconds(
    segments: list[tuple[int, int]],
    hop_length: int,
    frame_length: int,
    sr: int,
    audio_len: int,
) -> list[tuple[float, float]]:
    return [
        (
            start * hop_length / sr,
            min((end - 1) * hop_length + frame_length, audio_len) / sr,
        )
        for start, end in segments
        if end > start
    ]


def energy_vad(
    audio: np.ndarray,
    sr: int = 16000,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    noise_percentile: float = 20.0,
    vad_margin_db: float = 10.0,
    min_speech_ms: float = 200.0,
    pad_ms: float = 100.0,
):
    if sr <= 0:
        raise ValueError("sr must be positive")

    audio = np.asarray(audio, dtype=np.float32)
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    frames = frame_audio(audio, frame_length, hop_length)
    if len(frames) == 0:
        return np.array([], dtype=bool), [], float("-inf")

    rms = to_db(frame_rms(frames))
    noise_floor_db = np.percentile(rms, noise_percentile)
    threshold_db = noise_floor_db + vad_margin_db
    speech_mask = rms > threshold_db

    min_speech_frames = max(1, int(min_speech_ms / hop_ms))
    pad_frames = max(0, int(pad_ms / hop_ms))
    segments = collect_speech_segments(speech_mask, min_speech_frames, pad_frames)
    segments_sec = frames_to_seconds(segments, hop_length, frame_length, sr, len(audio))

    return speech_mask, segments_sec, threshold_db


def trim_silence(
    audio: np.ndarray,
    sr: int = 16000,
    **vad_kwargs,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    _, segments_sec, _ = energy_vad(audio, sr=sr, **vad_kwargs)

    if not segments_sec:
        return np.array([], dtype=np.float32)

    pieces = []
    for start_sec, end_sec in segments_sec:
        start_sample = max(0, int(round(start_sec * sr)))
        end_sample = min(len(audio), int(round(end_sec * sr)))

        if start_sample < end_sample:
            pieces.append(audio[start_sample:end_sample])

    if not pieces:
        return np.array([], dtype=np.float32)

    return np.concatenate(pieces).astype(np.float32)
