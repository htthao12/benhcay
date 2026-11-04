# HƯỚNG DẪN HUẤN LUYỆN 100 EPOCHS TRÊN GOOGLE COLAB (GPU MIỄN PHÍ)

Tài liệu này hướng dẫn chi tiết cách tải file gói zip lên Google Colab để huấn luyện 3 mô hình (`DSCNet`, `ResNet18`, `SE-ResNet18`) với **100 epochs** sử dụng GPU Tesla T4 hoàn toàn miễn phí.

---

### 1. File zip chuẩn bị:
File nén **`plant_disease_colab.zip`** nằm ngay tại thư mục:
`D:\AIRB\Nam5\hocsaunangcao\midterm\plant_disease_colab.zip`

Bên trong file zip đã có đầy đủ:
- `data/processed/`: 8.400 ảnh phân chia đều 7 lớp (`Healthy`, `Powdery_Mildew`, `Rust`, `Anthracnose`, `Aphids`, `Scorch_Sunburn`, `Yellowing_Wilt`).
- `src/`: Toàn bộ mã nguồn 3 mô hình, engine huấn luyện (hỗ trợ AMP fp16 tăng tốc GPU), đánh giá, xuất ONNX.
- `train_colab.ipynb`: Notebook Jupyter chạy 1-click.
- `requirements.txt`: Các thư viện cần thiết.

---

### 2. Các bước thực hiện trên Google Colab:

#### 🔹 Bước 1: Mở Google Colab và kích hoạt GPU
1. Mở trình duyệt truy cập: [https://colab.research.google.com](https://colab.research.google.com)
2. Bấm **New Notebook** (Sổ tay mới) hoặc bấm tab **Upload** và tải file `train_colab.ipynb` lên.
3. Bật GPU: Vào thanh menu trên cùng: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Phần **Hardware accelerator** chọn **T4 GPU** $\rightarrow$ Bấm **Save**.

#### 🔹 Bước 2: Tải file `plant_disease_colab.zip` lên Colab
- Ở thanh công cụ bên trái giao diện Colab, bấm vào biểu tượng **Thư mục (Files)**.
- Kéo thả file `plant_disease_colab.zip` từ máy tính của bạn vào khung Files (hoặc bấm nút Upload có hình trang giấy có mũi tên đi lên).
- *Mẹo nếu mạng chậm*: Bạn cũng có thể upload file zip này lên Google Drive cá nhân của bạn, rồi trong Colab chỉ cần chạy 2 dòng lệnh để giải nén trực tiếp từ Drive cực kỳ nhanh:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  !unzip -q "/content/drive/MyDrive/plant_disease_colab.zip" -d /content/project
  %cd /content/project
  ```

#### 🔹 Bước 3: Chạy huấn luyện 3 mô hình (100 Epochs mỗi mô hình)
Chạy lần lượt các ô lệnh trong file notebook:
1. Cài thư viện:
   ```bash
   !pip install -r requirements.txt
   ```
2. Huấn luyện mô hình **DSCNet** (Siêu nhẹ, ~6-8 phút là xong 100 epochs trên GPU T4):
   ```bash
   !python src/training/train.py --model dscnet --epochs 100 --batch_size 32
   ```
3. Huấn luyện mô hình **ResNet18**:
   ```bash
   !python src/training/train.py --model resnet18 --epochs 100 --batch_size 32
   ```
4. Huấn luyện mô hình **SE-ResNet18**:
   ```bash
   !python src/training/train.py --model se_resnet18 --epochs 100 --batch_size 32
   ```

#### 🔹 Bước 4: Tải kết quả và file trọng số về máy tính
Sau khi train xong 100 epochs, các file trọng số `.pth`, file biểu đồ `.png` và báo cáo phân loại sẽ nằm trong thư mục `weights/`. Chạy ô lệnh cuối:
```python
!zip -r trained_weights_and_results.zip weights/
from google.colab import files
files.download('trained_weights_and_results.zip')
```
Sau đó bạn tải file `trained_weights_and_results.zip` về máy, giải nén ghi đè vào thư mục `D:\AIRB\Nam5\hocsaunangcao\midterm\weights\`.

Lúc này, bạn chỉ việc gõ `streamlit run app/app.py` trên máy tính là ứng dụng Real-time sẽ sử dụng ngay trọng số 100 epochs đỉnh cao này để quét camera trực tiếp!
