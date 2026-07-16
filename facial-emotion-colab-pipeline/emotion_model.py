"""Shared model definition and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn


DEFAULT_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


class EmotionCNN(nn.Module):
    """Compact CNN designed for 48x48 grayscale face crops."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    labels: Sequence[str],
    epoch: int,
    val_accuracy: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "labels": list(labels),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "input_size": 48,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[EmotionCNN, list[str], dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    labels = checkpoint.get("labels", DEFAULT_LABELS)
    model = EmotionCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, list(labels), checkpoint

