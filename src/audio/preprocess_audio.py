from dataclasses import dataclass
import random
from pathlib import Path

import numpy as np

from .audio_io import find_audio_files, load_audio, save_audio
from .transforms import TARGET_SR, process_audio_with_augmentations


@dataclass(frozen=True)
class ProcessedAudioFile:
    source_path: Path
    output_path: Path


def process_file(
    input_file: Path,
    output_dir: Path,
    augment: bool = False,
    rng: np.random.Generator | None = None,
    target_sr: int = TARGET_SR,
) -> ProcessedAudioFile:
    audio, sr = load_audio(str(input_file))
    audio, target_sr = process_audio_with_augmentations(
        audio,
        sr,
        target_sr=target_sr,
        augment=augment,
        rng=rng,
    )

    out_path = output_dir / f"{input_file.stem}.wav"
    save_audio(str(out_path), audio, target_sr)
    return ProcessedAudioFile(source_path=input_file, output_path=out_path)


def process_random_files(
    input_dir: str | Path,
    output_dir: str | Path,
    n: int,
    seed: int,
    augment: bool = False,
    target_sr: int = TARGET_SR,
) -> list[ProcessedAudioFile]:
    if n < 1:
        raise ValueError("--n must be at least 1")

    input_files = find_audio_files(input_dir)
    if not input_files:
        raise RuntimeError(f"No audio files found in {input_dir}")

    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    chosen = py_rng.sample(input_files, k=min(n, len(input_files)))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    return [
        process_file(
            file_path,
            output_path,
            augment=augment,
            rng=rng,
            target_sr=target_sr,
        )
        for file_path in chosen
    ]
