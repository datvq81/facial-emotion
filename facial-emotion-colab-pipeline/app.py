"""Tkinter webcam interface for real-time facial-expression recognition."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
import torch

from emotion_model import load_checkpoint


VI_LABELS = {
    "angry": "Tức giận",
    "disgust": "Ghê sợ",
    "fear": "Sợ hãi",
    "happy": "Vui vẻ",
    "neutral": "Bình thường",
    "sad": "Buồn",
    "surprise": "Ngạc nhiên",
}


class EmotionApp:
    def __init__(self, root: tk.Tk, model_path: Path, smoothing: int) -> None:
        self.root = root
        self.root.title("Nhận diện cảm xúc khuôn mặt")
        self.root.geometry("1000x700")
        self.root.minsize(820, 600)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.labels, checkpoint = load_checkpoint(model_path, self.device)
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_detector.empty():
            raise RuntimeError("Không tải được bộ phát hiện khuôn mặt của OpenCV.")

        self.capture: cv2.VideoCapture | None = None
        self.running = False
        self.photo: ImageTk.PhotoImage | None = None
        self.probability_history: deque[np.ndarray] = deque(maxlen=max(1, smoothing))
        self.last_time = time.perf_counter()
        self.fps = 0.0

        self.status = tk.StringVar(value=f"Sẵn sàng • Model epoch {checkpoint.get('epoch', '?')}")
        self.prediction = tk.StringVar(value="Chưa có khuôn mặt")
        self.camera_index = tk.IntVar(value=0)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Prediction.TLabel", font=("Segoe UI", 18, "bold"))

        header = ttk.Frame(self.root, padding=(18, 14))
        header.pack(fill="x")
        ttk.Label(header, text="Nhận diện cảm xúc", style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.status).pack(side="right")

        content = ttk.Frame(self.root, padding=(18, 0, 18, 12))
        content.pack(fill="both", expand=True)
        self.video_label = ttk.Label(content, anchor="center")
        self.video_label.pack(side="left", fill="both", expand=True)

        panel = ttk.Frame(content, padding=(18, 12), width=260)
        panel.pack(side="right", fill="y")
        ttk.Label(panel, text="Kết quả hiện tại").pack(anchor="w")
        ttk.Label(panel, textvariable=self.prediction, style="Prediction.TLabel", wraplength=240).pack(
            anchor="w", pady=(5, 20)
        )

        ttk.Label(panel, text="Xác suất theo lớp").pack(anchor="w")
        self.bars: list[tuple[ttk.Progressbar, ttk.Label]] = []
        for label in self.labels:
            row = ttk.Frame(panel)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=VI_LABELS.get(label, label), width=13).pack(side="left")
            bar = ttk.Progressbar(row, maximum=100, length=105)
            bar.pack(side="left", padx=4)
            value = ttk.Label(row, text="0%", width=5)
            value.pack(side="left")
            self.bars.append((bar, value))

        controls = ttk.Frame(self.root, padding=(18, 8, 18, 18))
        controls.pack(fill="x")
        ttk.Label(controls, text="Camera:").pack(side="left")
        ttk.Spinbox(controls, from_=0, to=9, width=4, textvariable=self.camera_index).pack(
            side="left", padx=(6, 14)
        )
        ttk.Button(controls, text="Bắt đầu", command=self.start).pack(side="left", padx=4)
        ttk.Button(controls, text="Dừng", command=self.stop).pack(side="left", padx=4)
        ttk.Label(
            controls,
            text="Kết quả chỉ phản ánh biểu cảm bề mặt, không phải cảm xúc chắc chắn.",
        ).pack(side="right")

    def start(self) -> None:
        if self.running:
            return
        self.capture = cv2.VideoCapture(self.camera_index.get(), cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror("Lỗi camera", "Không mở được camera đã chọn.")
            return
        self.running = True
        self.probability_history.clear()
        self.status.set(f"Đang chạy • {self.device}")
        self._update_frame()

    def stop(self) -> None:
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.status.set("Đã dừng")

    def _predict(self, gray_face: np.ndarray) -> np.ndarray:
        face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(face).float().div(127.5).sub(1.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        return probabilities.cpu().numpy()

    def _update_frame(self) -> None:
        if not self.running or self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.stop()
            messagebox.showerror("Lỗi camera", "Không đọc được khung hình từ camera.")
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60)
        )
        largest = max(faces, key=lambda box: box[2] * box[3], default=None)
        if largest is not None:
            x, y, w, h = largest
            margin = int(0.08 * max(w, h))
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(gray.shape[1], x + w + margin), min(gray.shape[0], y + h + margin)
            probabilities = self._predict(gray[y1:y2, x1:x2])
            self.probability_history.append(probabilities)
            smoothed = np.mean(self.probability_history, axis=0)
            best = int(smoothed.argmax())
            label = self.labels[best]
            confidence = float(smoothed[best])
            color = (62, 201, 120) if confidence >= 0.6 else (40, 180, 240)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label}: {confidence:.0%}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            self.prediction.set(f"{VI_LABELS.get(label, label)} • {confidence:.0%}")
            for probability, (bar, value) in zip(smoothed, self.bars):
                percent = float(probability * 100)
                bar["value"] = percent
                value.configure(text=f"{percent:.0f}%")
        else:
            self.probability_history.clear()
            self.prediction.set("Chưa thấy khuôn mặt")

        now = time.perf_counter()
        instantaneous_fps = 1.0 / max(now - self.last_time, 1e-6)
        self.fps = instantaneous_fps if self.fps == 0 else 0.9 * self.fps + 0.1 * instantaneous_fps
        self.last_time = now
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(480, self.video_label.winfo_width())
        max_h = max(360, self.video_label.winfo_height())
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo)
        self.root.after(15, self._update_frame)

    def close(self) -> None:
        self.stop()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Giao diện nhận diện cảm xúc qua webcam")
    parser.add_argument("--model", type=Path, default=Path("models/emotion_cnn.pt"))
    parser.add_argument("--smoothing", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy model {args.model}. Hãy chạy train.py trước hoặc chỉ định --model."
        )
    root = tk.Tk()
    EmotionApp(root, args.model, args.smoothing)
    root.mainloop()


if __name__ == "__main__":
    main()

