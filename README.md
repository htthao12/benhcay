# Hệ Thống Nhận Diện Bệnh Cây Cảnh Thời Gian Thực (Real-Time Ornamental Plant Disease Diagnosis)
> **Đồ án môn học:** Học sâu nâng cao (Advanced Deep Learning)  
> **Hạng mục:** Báo cáo Đồ án Cuối kỳ & Kết thúc học phần (Final Course Project)  
> **Trường:** Đại học Phenikaa (Phenikaa University)  
> **Tác giả:** Hoàng Thị Thảo (MSV: 22011077)  
> **Repository:** [https://github.com/htthao12/benhcay](https://github.com/htthao12/benhcay)

---

## 📌 Giới Thiệu Đề Tài
Đề tài nghiên cứu và phát triển giải pháp thị giác máy tính ứng dụng học sâu (Deep Learning) nhằm **chẩn đoán bệnh trên lá cây cảnh theo thời gian thực (Real-time)** với độ chính xác cao và độ trễ thấp, tối ưu hóa để triển khai trên các thiết bị rìa (Edge Devices, Mobile, WebAssembly/ONNX).

Hệ thống phân loại chính xác **7 lớp trạng thái lá cây cảnh**:
1. 🌿 **Healthy** (Lá khỏe mạnh)
2. 🍂 **Anthracnose** (Bệnh Thán thư)
3. 🐛 **Aphids** (Rệp sáp / Sâu hại chích hút)
4. ❄️ **Powdery_Mildew** (Bệnh Phấn trắng)
5. 🟤 **Rust** (Bệnh Rỉ sắt)
6. ☀️ **Scorch_Sunburn** (Cháy nắng / Khô mép lá)
7. 🍂 **Yellowing_Wilt** (Vàng lá / Héo rũ / Úng rễ)

---

## 🏗️ Kiến Trúc 3 Mô Hình Học Sâu

Dự án thiết kế, huấn luyện và so sánh thực nghiệm 3 kiến trúc mạng nơ-ron tích chập (CNN) tối ưu phần cứng:

| Kiến trúc | Đặc điểm nổi bật | Số tham số | FLOPs (M) | MACs (M) | Dung lượng (.pth) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ResNet-18** *(Compact Bottleneck)* | Kiến trúc Residual gọn nhẹ với Bottleneck 1x1, độ khái quát cao nhất | **120,279** | 106.64 | 51.78 | 1.51 MB |
| **SE-ResNet-18** | Tích hợp cơ chế Channel Attention (Squeeze-and-Excitation) | **128,599** | 106.65 | 51.79 | 1.63 MB |
| **DSCNet** | Sử dụng Depthwise Separable Convolutions siêu nhẹ | **53,055** | **57.46** | **26.76** | **0.72 MB** |

---

## 📊 Kết Quả Thực Nghiệm & Đánh Giá (Tập Test: 1,400 ảnh độc lập)

| Hạng | Mô hình | Best Epoch | Val Acc (%) | Test Top-1 Acc (%) | Test Top-3 Acc (%) | Test Macro-F1 | CPU Latency (ms) | CPU FPS | Lựa chọn triển khai |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **ResNet-18** | **71** | **98.79%** | **98.07%** | **99.71%** | **0.9807** | **3.15 ms** | **317.4 FPS** | ⭐ **Best Model (App)** |
| 🥈 | **SE-ResNet-18** | 61 | 98.79% | 98.00% | 99.71% | 0.9800 | 3.23 ms | 309.2 FPS | Dự phòng |
| 🥉 | **DSCNet** | 87 | 98.86% | 97.79% | 99.93% | 0.9779 | **1.85 ms** | **541.1 FPS** | Thiết bị siêu nhẹ |

### Chi tiết F1-Score & Accuracy từng lớp bệnh (ResNet-18 - Best Model):
* **Anthracnose (Thán thư):** F1 = `0.9524` | Accuracy = `95.00%`
* **Aphids (Rệp sáp / Sâu hại):** F1 = `0.9848` | Accuracy = `97.00%`
* **Healthy (Lá khỏe mạnh):** F1 = `0.9779` | Accuracy = `99.50%`
* **Powdery_Mildew (Phấn trắng):** F1 = `0.9926` | Accuracy = `100.00%`
* **Rust (Rỉ sắt):** F1 = `0.9901` | Accuracy = `100.00%`
* **Scorch_Sunburn (Cháy nắng):** F1 = `0.9773` | Accuracy = `97.00%`
* **Yellowing_Wilt (Vàng lá):** F1 = `0.9899` | Accuracy = `98.00%`

---

## 🔍 Trực Quan Hóa XAI (Grad-CAM & Grad-CAM++)
Để giải thích quyết định của mô hình (Explainable AI), dự án áp dụng kỹ thuật **Grad-CAM** và **Grad-CAM++** tại lớp tích chập cuối cùng:
- Bản đồ nhiệt (Heatmap) tập trung chính xác vào vùng vết bệnh thực sự (đốm hoại tử thán thư, mảng phấn trắng, ổ rỉ sắt) thay vì học vẹt hậu cảnh.
- Chi tiết hình ảnh trực quan hóa được lưu tại thư mục [results/](results/).

---

## 📁 Cấu Trúc Thư Mục Repository

```text
benhcay/
├── app/
│   ├── app.py                     # FastAPI Real-time Streaming Server + Grad-CAM API
│   └── mobile_app.html            # Giao diện Web App tương thích Mobile (Camera / Upload)
├── src/
│   ├── data/
│   │   └── dataset.py             # DataLoader nạp ảnh đa nguồn, Data Augmentation
│   ├── models/
│   │   ├── dscnet.py              # Depthwise Separable Convolution Network
│   │   ├── resnet18.py            # ResNet-18 Compact Bottleneck
│   │   └── se_resnet18.py         # Squeeze-and-Excitation ResNet-18
│   ├── training/
│   │   └── train.py               # Script huấn luyện với Early Stopping, ReduceLROnPlateau
│   ├── evaluation/
│   │   ├── evaluate.py            # Đánh giá toàn diện (Acc, F1, Matrix, Latency, FPS)
│   │   └── gradcam.py             # Trích xuất bản đồ nhiệt Grad-CAM & Grad-CAM++
│   └── export/
│       └── export_model.py        # Xuất mô hình sang định dạng ONNX
├── results/                       # Báo cáo, đồ thị Loss/Accuracy, Confusion Matrix, Grad-CAM
│   ├── 1_dscnet/
│   ├── 2_resnet18/
│   ├── 3_se_resnet18/
│   └── 4_comparison_all_models/   # Báo cáo tổng hợp đối sánh 3 mô hình
├── weights/                       # Trọng số tốt nhất (.pth) và ONNX
│   ├── resnet18_best.pth          # Best Model (1.5 MB)
│   ├── se_resnet18_best.pth       # (1.6 MB)
│   ├── dscnet_best.pth            # (0.7 MB)
│   ├── best_model_info.json
│   └── *.onnx
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 hoặc 3.11:
```bash
git clone https://github.com/htthao12/benhcay.git
cd benhcay
pip install -r requirements.txt
```

### 2. Chạy ứng dụng chẩn đoán thời gian thực (Web / Mobile App)
Khởi động máy chủ ứng dụng:
```bash
python app/app.py
```
* **Giao diện Web:** Mở trình duyệt truy cập `http://localhost:8000`
* **Điện thoại di động (cùng Wi-Fi/LAN):** Truy cập địa chỉ IP hiển thị trên terminal (ví dụ: `http://192.168.x.x:8000`).

### 3. Huấn luyện mô hình (Training)
```bash
# Huấn luyện ResNet-18
python src/training/train.py --model resnet18 --epochs 100 --batch-size 64 --lr 0.001

# Huấn luyện SE-ResNet-18
python src/training/train.py --model se_resnet18 --epochs 100 --batch-size 64 --lr 0.001

# Huấn luyện DSCNet
python src/training/train.py --model dscnet --epochs 100 --batch-size 64 --lr 0.001
```

### 4. Đánh giá mô hình (Evaluation & Benchmarking)
```bash
python src/evaluation/evaluate.py --all --data-dir data/test
```

### 5. Xuất mô hình ONNX
```bash
python src/export/export_model.py --model resnet18
```

---

## 📜 Giấy Phép & Bản Quyền
Dự án được xây dựng phục vụ nghiên cứu học thuật tại Đại học Phenikaa.
Mọi thắc mắc xin liên hệ tác giả qua email sinh viên: `22011077@st.phenikaa-uni.edu.vn`.
