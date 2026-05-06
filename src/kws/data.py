import random
from pathlib import Path
from typing import Any

from src.asr.manifest import read_jsonl, write_jsonl
from src.asr.text import normalize_text


def build_kws_manifest_from_asr(
    source_manifest: Path,
    keyword: str,
    output_path: Path,
    max_positives: int | None = None,
    max_negatives: int | None = None,
    seed: int = 42,
    match: str = "contains",
) -> dict[str, int | str]:
    keyword_normalized = normalize_text(keyword)
    if not keyword_normalized:
        raise ValueError("keyword must not be empty after normalization")

    positives = []
    negatives = []
    for record in read_jsonl(source_manifest):
        text_normalized = record.get("text_normalized") or normalize_text(record.get("text", ""))
        is_positive = (
            text_normalized == keyword_normalized
            if match == "exact"
            else keyword_normalized in text_normalized
        )
        target = positives if is_positive else negatives
        target.append({
            "id": record["id"],
            "audio": record["audio"],
            "label": int(is_positive),
            "keyword": keyword,
            "text": record.get("text", ""),
            "text_normalized": text_normalized,
            "source_split": record.get("split", ""),
        })

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if max_positives is not None:
        positives = positives[:max_positives]
    if max_negatives is not None:
        negatives = negatives[:max_negatives]

    if not positives:
        raise ValueError(f"No positive examples found for keyword: {keyword!r}")
    if not negatives:
        raise ValueError("No negative examples found")

    records = positives + negatives
    rng.shuffle(records)
    write_jsonl(output_path, records)
    return {
        "keyword": keyword,
        "match": match,
        "positives": len(positives),
        "negatives": len(negatives),
        "total": len(records),
        "output": str(output_path),
    }


def split_kws_manifest(
    source_manifest: Path,
    train_output: Path,
    val_output: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, int | str]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    records = read_jsonl(source_manifest)
    if len(records) < 2:
        raise ValueError("Need at least two KWS records to create a validation split")

    by_label = {0: [], 1: []}
    for record in records:
        by_label[int(record["label"])].append(record)

    rng = random.Random(seed)
    train_records = []
    val_records = []
    for label_records in by_label.values():
        rng.shuffle(label_records)
        if len(label_records) <= 1:
            train_records.extend(label_records)
            continue

        val_count = max(1, round(len(label_records) * val_ratio))
        val_count = min(val_count, len(label_records) - 1)
        val_records.extend(label_records[:val_count])
        train_records.extend(label_records[val_count:])

    if not val_records:
        raise ValueError("Could not create a non-empty validation split")

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    write_jsonl(train_output, train_records)
    write_jsonl(val_output, val_records)
    return {
        "source": str(source_manifest),
        "train_output": str(train_output),
        "val_output": str(val_output),
        "train_records": len(train_records),
        "val_records": len(val_records),
    }
