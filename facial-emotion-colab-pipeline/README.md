# Nhận diện cảm xúc khuôn mặt thời gian thực

Dự án gồm hai phần:

- `train.py`: huấn luyện CNN với ảnh khuôn mặt xám 48x48.
- `app.py`: giao diện webcam, phát hiện khuôn mặt và dự đoán cảm xúc theo thời gian thực.

Các lớp mặc định: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

## 1. Cài đặt

Khuyến nghị Python 3.10-3.12. Trên Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu máy có GPU NVIDIA, cài PyTorch theo hướng dẫn tại https://pytorch.org/get-started/locally/ trước khi chạy lệnh cài requirements.

## 2. Chuẩn bị dữ liệu

Cấu trúc dữ liệu dùng chuẩn `ImageFolder`:

```text
data/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  val/
    angry/
    ...
```

Mỗi thư mục lớp chứa ảnh `.jpg` hoặc `.png`. Nếu bộ dữ liệu chỉ có `train` và `test`, có thể dùng `test` làm validation bằng tham số `--val-dir data/test`.

## 3. Huấn luyện

```powershell
python train.py --train-dir data/train --val-dir data/val --epochs 30
```

Checkpoint tốt nhất được lưu tại `models/emotion_cnn.pt`. Ví dụ chạy nhanh để kiểm tra pipeline:

```powershell
python train.py --train-dir data/train --val-dir data/val --epochs 1 --batch-size 32
```

Các tùy chọn hữu ích:

```text
--device auto|cpu|cuda   Thiết bị huấn luyện
--num-workers N          Số tiến trình đọc dữ liệu
--patience N             Dừng sớm sau N epoch không cải thiện
--output PATH            Nơi lưu checkpoint
```

### Huấn luyện bằng Google Colab

Mở `colab_train.ipynb` bằng Google Colab và chạy lần lượt các cell. Notebook sẽ:

1. Gắn Google Drive để model không mất khi phiên Colab ngắt.
2. Giải nén dataset `.zip` nếu cần và kiểm tra cấu trúc lớp.
3. Huấn luyện bằng GPU với mixed precision.
4. Lưu model tốt nhất vào Drive và tải file `.pt` về máy.

Hướng dẫn chi tiết và cấu trúc Drive nằm trong `COLAB.md`.

## 4. Chạy giao diện webcam

```powershell
python app.py --model models/emotion_cnn.pt
```

Chọn camera trong giao diện, bấm **Bắt đầu**, và bấm **Dừng** trước khi đổi camera. Tham số `--smoothing` điều chỉnh số khung hình dùng để làm mượt dự đoán.

## Lưu ý chất lượng

Đây là mô hình biểu cảm khuôn mặt, không thể biết chắc cảm xúc nội tâm. Kết quả phụ thuộc mạnh vào ánh sáng, góc mặt, độ cân bằng dữ liệu và khác biệt nhân khẩu học. Không nên dùng kết quả cho quyết định y tế, tuyển dụng, kỷ luật hoặc giám sát con người.
