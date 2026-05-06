import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.asr.manifest import write_jsonl
from src.asr.text import normalize_text


SPLIT_FILES = {
    "train": "train.tsv",
    "val": "dev.tsv",
    "test": "test.tsv",
}


def read_clip_durations(path: Path) -> dict[str, int]:
    durations = {}
    if not path.exists():
        return durations

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip = row.get("clip")
            duration = row.get("duration[ms]")
            if not clip or not duration:
                continue
            durations[clip] = int(float(duration))
    return durations


def load_split_records(
    split: str,
    tsv_path: Path,
    clips_dir: Path,
    durations_ms: dict[str, int],
    min_duration_ms: int,
    max_duration_ms: int | None,
    limit: int | None,
) -> tuple[list[dict[str, Any]], Counter]:
    records = []
    skipped = Counter()

    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip_name = row.get("path", "").strip()
            text = row.get("sentence", "").strip()
            normalized = normalize_text(text)

            if not clip_name:
                skipped["missing_path"] += 1
                continue
            if not normalized:
                skipped["empty_text"] += 1
                continue

            audio_path = clips_dir / clip_name
            if not audio_path.exists():
                skipped["missing_audio"] += 1
                continue

            duration_ms = durations_ms.get(clip_name)
            if duration_ms is not None:
                if duration_ms < min_duration_ms:
                    skipped["too_short"] += 1
                    continue
                if max_duration_ms is not None and duration_ms > max_duration_ms:
                    skipped["too_long"] += 1
                    continue

            records.append({
                "id": Path(clip_name).stem,
                "split": split,
                "audio": str(audio_path),
                "text": text,
                "text_normalized": normalized,
                "duration_ms": duration_ms,
                "duration_sec": duration_ms / 1000 if duration_ms is not None else None,
                "sentence_id": row.get("sentence_id", ""),
                "speaker_id": row.get("client_id", ""),
                "locale": row.get("locale", ""),
            })

            if limit is not None and len(records) >= limit:
                break

    return records, skipped


def summarize(records: list[dict[str, Any]], skipped: Counter) -> dict[str, Any]:
    durations = [r["duration_ms"] for r in records if r["duration_ms"] is not None]
    text_counts = Counter(r["text_normalized"] for r in records)
    duplicate_texts = sum(1 for count in text_counts.values() if count > 1)

    return {
        "records": len(records),
        "duration_hours": round(sum(durations) / 3_600_000, 3),
        "missing_duration": len(records) - len(durations),
        "duplicate_normalized_texts": duplicate_texts,
        "skipped": dict(skipped),
    }


def prepare_dataset(
    metadata_dir: Path,
    clips_dir: Path,
    output_dir: Path,
    min_duration_ms: int,
    max_duration_ms: int | None,
    limit_per_split: int | None,
) -> dict[str, Any]:
    durations = read_clip_durations(metadata_dir / "clip_durations.tsv")
    all_records = []
    stats = {}

    for split, file_name in SPLIT_FILES.items():
        records, skipped = load_split_records(
            split=split,
            tsv_path=metadata_dir / file_name,
            clips_dir=clips_dir,
            durations_ms=durations,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            limit=limit_per_split,
        )
        write_jsonl(output_dir / f"manifest_{split}.jsonl", records)
        all_records.extend(records)
        stats[split] = summarize(records, skipped)

    write_jsonl(output_dir / "manifest_all.jsonl", all_records)
    stats["all"] = summarize(all_records, Counter())

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats
