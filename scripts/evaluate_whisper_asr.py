from pathlib import Path
import whisper

from src.asr.manifest import read_jsonl, write_jsonl
from src.asr.metrics import error_rates
from src.audio.decode import load_audio


WHISPER_SAMPLE_RATE = 16000


def resolve_audio_path(audio_path: str, project_root: Path) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return project_root / path


def evaluate_manifest(
    manifest_path: Path,
    output_path: Path,
    model_name: str,
    download_root: Path,
    language: str,
    limit: int | None,
    fp16: bool,
) -> dict[str, float | int]:
    project_root = Path.cwd()
    records = read_jsonl(manifest_path, limit=limit)
    download_root.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(model_name, download_root=str(download_root))

    predictions = []
    total_word_errors = 0
    total_words = 0
    total_char_errors = 0
    total_chars = 0

    for idx, record in enumerate(records, start=1):
        audio_path = resolve_audio_path(record["audio"], project_root)
        audio = load_audio(audio_path, sample_rate=WHISPER_SAMPLE_RATE)
        result = model.transcribe(
            audio,
            language=language,
            task="transcribe",
            fp16=fp16,
        )
        hypothesis = result.get("text", "").strip()
        metrics = error_rates(record["text"], hypothesis)

        total_word_errors += int(metrics["word_errors"])
        total_words += int(metrics["word_count"])
        total_char_errors += int(metrics["char_errors"])
        total_chars += int(metrics["char_count"])

        predictions.append({
            "id": record["id"],
            "audio": record["audio"],
            "reference": record["text"],
            "reference_normalized": metrics["reference_normalized"],
            "hypothesis": hypothesis,
            "hypothesis_normalized": metrics["hypothesis_normalized"],
            "wer": metrics["wer"],
            "cer": metrics["cer"],
        })

        print(
            f"[{idx}/{len(records)}] {record['id']} "
            f"WER={metrics['wer']:.3f} CER={metrics['cer']:.3f}"
        )

    write_jsonl(output_path, predictions)

    return {
        "samples": len(records),
        "wer": total_word_errors / total_words if total_words else 0.0,
        "cer": total_char_errors / total_chars if total_chars else 0.0,
        "word_errors": total_word_errors,
        "words": total_words,
        "char_errors": total_char_errors,
        "chars": total_chars,
    }
