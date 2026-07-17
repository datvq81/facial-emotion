# Pipeline Google Colab → máy local

## Chuẩn bị trên Google Drive

1. Giải nén dự án vào `MyDrive/facial-emotion-recognition/`.
2. Đưa dataset lên Drive theo một trong hai cách:
   - thư mục `data/train` và `data/val`; hoặc
   - file `fer2013.zip` chứa `train/val` hoặc `train/test`.
3. Mở `colab_train.ipynb` bằng Google Colab.
4. Chọn **Runtime → Change runtime type → T4 GPU**.

Cấu trúc khuyến nghị:

```text
MyDrive/facial-emotion-recognition/
  app.py
  emotion_model.py
  train.py
  colab_train.ipynb
  fer2013.zip             # có thể xóa sau khi giải nén
  data/
    train/<label>/*.jpg
    val/<label>/*.jpg      # hoặc test/<label>/*.jpg
  models/
```

## Chạy notebook

Sửa các biến trong cell **Cấu hình** nếu đường dẫn của bạn khác, sau đó chọn **Runtime → Run all**. Model tốt nhất được lưu vào:

```text
MyDrive/facial-emotion-recognition/models/emotion_cnn.pt
```

Để GPU không phải chờ Google Drive, hãy copy file ZIP sang `/content` rồi giải nén
dataset tại đó. Chỉ checkpoint cần lưu trên Drive. Với T4, thử batch size 1024 trước,
sau đó tăng lên 2048 nếu không bị CUDA out of memory.

Vì checkpoint nằm trên Drive, file vẫn còn nếu Colab ngắt kết nối. Để tiếp tục từ checkpoint, đặt `RESUME = True` trong cell cấu hình rồi chạy lại. `EPOCHS` là tổng số epoch mong muốn, không phải số epoch chạy thêm.

## Dùng model trên máy local

Tải `emotion_cnn.pt` từ notebook hoặc Google Drive và đặt tại:

```text
models/emotion_cnn.pt
```

Sau đó cài dependencies và chạy:

```powershell
pip install -r requirements.txt
python app.py --model models/emotion_cnn.pt
```

Checkpoint dùng `map_location`, vì vậy model huấn luyện bằng GPU Colab vẫn chạy được trên CPU local.
