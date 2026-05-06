from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.asr.manifest import read_jsonl
from src.audio.decode import load_audio
from src.kws.features import DEFAULT_FEATURE_CONFIG, audio_to_log_mel
from src.kws.model import SmallKWSCNN


class KWSDataset(Dataset):
    def __init__(self, manifest_path: Path, feature_config: dict, limit: int | None = None) -> None:
        self.records = read_jsonl(manifest_path, limit=limit)
        self.feature_config = feature_config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        audio = load_audio(record["audio"], sample_rate=int(self.feature_config["sample_rate"]))
        features = audio_to_log_mel(audio, self.feature_config).unsqueeze(0)
        label = torch.tensor(float(record["label"]), dtype=torch.float32)
        return features, label


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def evaluate_kws_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            predictions = (torch.sigmoid(logits) >= 0.5).float()

            total += labels.numel()
            correct += int((predictions == labels).sum().item())
            true_positive += int(((predictions == 1) & (labels == 1)).sum().item())
            false_positive += int(((predictions == 1) & (labels == 0)).sum().item())
            false_negative += int(((predictions == 0) & (labels == 1)).sum().item())
            total_loss += float(loss.item()) * labels.numel()

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "precision": precision,
        "recall": recall,
    }


def train_kws_model(
    train_manifest: Path,
    val_manifest: Path,
    checkpoint_path: Path,
    keyword: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "auto",
    limit_train: int | None = None,
    limit_val: int | None = None,
    feature_config: dict | None = None,
) -> dict[str, Any]:
    config = dict(DEFAULT_FEATURE_CONFIG)
    if feature_config:
        config.update(feature_config)

    torch_device = resolve_device(device)
    train_loader = DataLoader(
        KWSDataset(train_manifest, config, limit=limit_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        KWSDataset(val_manifest, config, limit=limit_val),
        batch_size=batch_size,
        shuffle=False,
    )

    model = SmallKWSCNN().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for features, labels in train_loader:
            features = features.to(torch_device)
            labels = labels.to(torch_device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.numel()
            total += labels.numel()

        val_metrics = evaluate_kws_model(model, val_loader, torch_device)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch} train_loss={epoch_metrics['train_loss']:.4f} "
            f"val_acc={epoch_metrics['val_accuracy']:.3f} "
            f"val_precision={epoch_metrics['val_precision']:.3f} "
            f"val_recall={epoch_metrics['val_recall']:.3f}"
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "keyword": keyword,
            "feature_config": config,
            "history": history,
        },
        checkpoint_path,
    )
    return {"checkpoint": str(checkpoint_path), "history": history, "device": str(torch_device)}


def detect_keyword(
    audio_path: Path,
    checkpoint_path: Path,
    threshold: float = 0.5,
    device: str = "auto",
) -> dict[str, Any]:
    torch_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    config = checkpoint["feature_config"]
    model = SmallKWSCNN().to(torch_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    audio = load_audio(audio_path, sample_rate=int(config["sample_rate"]))
    features = audio_to_log_mel(audio, config).unsqueeze(0).unsqueeze(0).to(torch_device)
    with torch.no_grad():
        probability = float(torch.sigmoid(model(features)).item())

    return {
        "keyword": checkpoint.get("keyword", ""),
        "probability": probability,
        "threshold": threshold,
        "wake": probability >= threshold,
    }
