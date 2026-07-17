"""Evaluate a trained emotion model on a held-out dataset or one face image."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from emotion_model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm thử model nhận diện cảm xúc")
    parser.add_argument("--model", type=Path, default=Path("models/emotion_cnn.pt"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--test-dir",
        type=Path,
        help="Thư mục test dạng test/<label>/*.jpg",
    )
    source.add_argument("--image", type=Path, help="Một ảnh khuôn mặt cần dự đoán")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--confusion-csv",
        type=Path,
        default=None,
        help="Đường dẫn lưu confusion matrix dạng CSV",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã chọn CUDA nhưng PyTorch không tìm thấy GPU.")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def preprocessing(input_size: int, architecture: str | None) -> transforms.Compose:
    """Must match the deterministic validation transform used during training."""
    if architecture == "resnet18":
        return transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose([transforms.Grayscale(1), transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def predict_image(
    image_path: Path,
    model: torch.nn.Module,
    labels: list[str],
    transform: transforms.Compose,
    device: torch.device,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

    ranking = torch.argsort(probabilities, descending=True)
    print(f"Ảnh: {image_path}")
    print(f"Dự đoán: {labels[ranking[0]]} ({probabilities[ranking[0]].item():.2%})")
    print("\nXác suất:")
    for index in ranking.tolist():
        print(f"  {labels[index]:<12} {probabilities[index].item():>7.2%}")


def evaluate_dataset(
    test_dir: Path,
    model: torch.nn.Module,
    checkpoint_labels: list[str],
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy tập test: {test_dir}")
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    if dataset.classes != checkpoint_labels:
        raise ValueError(
            "Nhãn hoặc thứ tự nhãn không khớp checkpoint:\n"
            f"  checkpoint: {checkpoint_labels}\n"
            f"  test:       {dataset.classes}"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    class_count = len(checkpoint_labels)
    confusion = torch.zeros((class_count, class_count), dtype=torch.int64)
    total_loss = total_correct = total = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            total_loss += criterion(logits, targets).item()
            total_correct += (predictions == targets).sum().item()
            total += targets.numel()
            flat_indices = (targets * class_count + predictions).cpu()
            confusion += torch.bincount(
                flat_indices, minlength=class_count * class_count
            ).reshape(class_count, class_count)

    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    print(f"Thiết bị: {device} | Số ảnh test: {total}")
    print(f"Test loss: {total_loss / total:.4f}")
    print(f"Accuracy:  {total_correct / total:.2%}")
    print(f"Macro F1:  {f1.mean().item():.2%}")
    print("\nTheo từng lớp:")
    print(f"{'Lớp':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for index, label in enumerate(checkpoint_labels):
        print(
            f"{label:<12} {precision[index].item():>9.2%} "
            f"{recall[index].item():>9.2%} {f1[index].item():>9.2%} "
            f"{int(support[index].item()):>10}"
        )

    print("\nConfusion matrix (hàng=nhãn thật, cột=dự đoán):")
    header = "".ljust(12) + " ".join(f"{label[:7]:>7}" for label in checkpoint_labels)
    print(header)
    for label, row in zip(checkpoint_labels, confusion.tolist()):
        print(f"{label:<12}" + " ".join(f"{value:>7}" for value in row))
    return confusion


def save_confusion_csv(path: Path, labels: list[str], confusion: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, confusion.tolist()):
            writer.writerow([label, *row])
    print(f"\nĐã lưu confusion matrix: {path}")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if not args.model.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {args.model}")
    model, labels, checkpoint = load_checkpoint(args.model, device)
    transform = preprocessing(int(checkpoint.get("input_size", 48)), checkpoint.get("architecture"))

    if args.image is not None:
        predict_image(args.image, model, labels, transform, device)
        return

    confusion = evaluate_dataset(
        args.test_dir,
        model,
        labels,
        transform,
        device,
        args.batch_size,
        args.num_workers,
    )
    if args.confusion_csv is not None:
        save_confusion_csv(args.confusion_csv, labels, confusion)


if __name__ == "__main__":
    main()
