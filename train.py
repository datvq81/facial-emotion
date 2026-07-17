"""Train the facial-expression classifier from ImageFolder datasets."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from emotion_model import ResNet18Emotion, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình nhận diện cảm xúc")
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/val"))
    parser.add_argument("--output", type=Path, default=Path("models/emotion_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Khởi tạo ResNet-18 từ trọng số ImageNet",
    )
    parser.add_argument(
        "--augmentation",
        choices=("none", "light", "strong"),
        default="light",
        help="Mức tăng cường dữ liệu cho tập train",
    )
    parser.add_argument(
        "--balance-strategy",
        choices=("weighted_loss", "sampler", "none"),
        default="weighted_loss",
        help="Cách xử lý mất cân bằng lớp",
    )
    parser.add_argument(
        "--balance-power",
        type=float,
        default=0.5,
        help="Số mũ trọng số lớp: 0=không cân bằng, 0.5=căn nghịch đảo, 1=nghịch đảo",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Số batch mỗi worker chuẩn bị trước; chỉ dùng khi num-workers > 0",
    )
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint .pt để tiếp tục huấn luyện (epochs là tổng số epoch)",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dùng mixed precision khi huấn luyện bằng CUDA",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng PyTorch không thấy GPU tương thích.")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_train_transform(level: str, input_size: int) -> transforms.Compose:
    operations: list = [
        transforms.Resize((input_size, input_size)),
    ]
    if level in ("light", "strong"):
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7 if level == "light" else 12),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05) if level == "light" else (0.10, 0.10),
                    scale=(0.95, 1.05) if level == "light" else (0.88, 1.12),
                ),
            ]
        )
    if level == "strong":
        operations.append(
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.25)], p=0.5
            )
        )
    # ImageFolder loads every image as RGB.  Retaining three channels lets the
    # model benefit from ImageNet pretraining even when FER images are grayscale.
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    if level == "strong":
        operations.append(
            transforms.RandomErasing(
                p=0.2, scale=(0.02, 0.10), ratio=(0.5, 2.0), value=0.0
            )
        )
    return transforms.Compose(operations)


def build_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, list[str], torch.Tensor | None]:
    for directory in (args.train_dir, args.val_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {directory}")

    train_transform = build_train_transform(args.augmentation, args.input_size)
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_set = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(args.val_dir, transform=val_transform)
    if train_set.classes != val_set.classes:
        raise ValueError(
            f"Nhãn train và validation không khớp: {train_set.classes} != {val_set.classes}"
        )

    counts = np.bincount(train_set.targets, minlength=len(train_set.classes))
    if np.any(counts == 0):
        raise ValueError("Mỗi lớp phải có ít nhất một ảnh huấn luyện.")
    if not 0.0 <= args.balance_power <= 1.0:
        raise ValueError("--balance-power phải nằm trong đoạn [0, 1].")

    raw_class_weights = (counts.sum() / (len(counts) * counts)) ** args.balance_power
    raw_class_weights = raw_class_weights / raw_class_weights.mean()
    class_weights = torch.tensor(raw_class_weights, dtype=torch.float32)

    sampler = None
    shuffle = True
    loss_weights: torch.Tensor | None = None
    if args.balance_strategy == "sampler":
        sample_weights = raw_class_weights[np.asarray(train_set.targets)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
    elif args.balance_strategy == "weighted_loss":
        loss_weights = class_weights
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        common["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_set, sampler=sampler, shuffle=shuffle, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    print("Số ảnh train theo lớp:", dict(zip(train_set.classes, counts.tolist())))
    print(
        f"Cân bằng: {args.balance_strategy} (power={args.balance_power}) | Trọng số:",
        {label: round(float(weight), 3) for label, weight in zip(train_set.classes, class_weights)},
    )
    print(f"Augmentation: {args.augmentation}")
    return train_loader, val_loader, train_set.classes, loss_weights


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += (logits.argmax(1) == targets).sum().item()
            total_samples += targets.size(0)
    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    train_loader, val_loader, labels, loss_weights = build_loaders(args)
    model = ResNet18Emotion(num_classes=len(labels), pretrained=args.pretrained).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    start_epoch = 1
    best_accuracy = -1.0
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint để tiếp tục: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        checkpoint_labels = checkpoint.get("labels", labels)
        if list(checkpoint_labels) != list(labels):
            raise ValueError(
                f"Nhãn checkpoint không khớp dữ liệu: {checkpoint_labels} != {labels}"
            )
        if checkpoint.get("architecture") != "resnet18":
            raise ValueError(
                "Checkpoint --resume là CNN cũ. Hãy train ResNet với file --output mới "
                "và không truyền --resume ở lần đầu."
            )
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_accuracy = float(checkpoint.get("val_accuracy", -1.0))
        print(f"Tiếp tục từ {args.resume} (epoch {start_epoch - 1}, acc {best_accuracy:.2%})")

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights.to(device) if loss_weights is not None else None,
        label_smoothing=0.05,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Thiết bị: {device} | AMP: {use_amp} | Số lớp: {len(labels)} | Nhãn: {labels}")
    epochs_without_improvement = 0
    if start_epoch > args.epochs:
        print(f"Checkpoint đã đạt epoch {start_epoch - 1}; không còn epoch nào để chạy.")
        return
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.2%} | "
            f"val loss={val_loss:.4f} acc={val_acc:.2%}"
        )
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                args.output,
                model,
                labels,
                epoch,
                val_acc,
                architecture="resnet18",
                input_size=args.input_size,
            )
            print(f"  Đã lưu checkpoint tốt nhất: {args.output}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Dừng sớm sau {epoch} epoch.")
                break
    print(f"Hoàn tất. Validation accuracy tốt nhất: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
