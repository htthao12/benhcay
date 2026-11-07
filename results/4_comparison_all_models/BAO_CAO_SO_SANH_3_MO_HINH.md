# BÁO CÁO TỔNG HỢP & ĐỐI SÁNH KẾT QUẢ 3 MÔ HÌNH HỌC SÂU (PCOPD)

**Mô hình xuất sắc nhất (được chọn cho ứng dụng Real-Time):** **ResNet-18 (Compact Bottleneck 1x1) (`RESNET18`)**
- **Test Macro-F1:** `0.9807`
- **Test Top-1 Accuracy:** `98.07%`
- **Test Top-3 Accuracy:** `99.71%`
- **Tổng số tham số:** `120,279` (`1.51 MB`)
- **Độ phức tạp tính toán:** `106.64 MFLOPs` (`51.78 MMACs`)
- **Tốc độ suy luận CPU (Batch=1):** `3.15 ms` (`317.4 FPS`)

---

## 1. Bảng Đối Sánh Tổng Hợp 3 Mô Hình Trên Tập Test

| Hạng | Mô hình | Số tham số | FLOPs (M) | MACs (M) | Dung lượng (MB) | Best Epoch | Val Acc (%) | Val Macro-F1 | Test Top-1 Acc (%) | Test Top-3 Acc (%) | Test Macro-F1 | CPU Latency (ms) | CPU FPS | Chọn cho App Real-Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **RESNET18** (ResNet-18 (Compact Bottleneck 1x1)) | 120,279 | 106.64 | 51.78 | 1.51 | 71 | 98.79% | 0.9878 | **98.07%** | 99.71% | **0.9807** | 3.15 | 317.4 | **BEST MODEL** |
| 2 | **SE_RESNET18** (SE-ResNet-18 (Channel Attention)) | 128,599 | 106.65 | 51.79 | 1.63 | 61 | 98.79% | 0.9879 | **98.00%** | 99.71% | **0.9800** | 3.23 | 309.2 | — |
| 3 | **DSCNET** (DSCNet (Compact Depthwise Separable)) | 53,055 | 57.46 | 26.76 | 0.72 | 87 | 98.86% | 0.9886 | **97.79%** | 99.93% | **0.9779** | 1.85 | 541.1 | — |

---

## 2. Bảng Đối Sánh F1-Score & Độ Chính Xác Từng Lớp Bệnh Giữa 3 Mô Hình

| Lớp bệnh (Disease Class) | DSCNet F1 | ResNet-18 F1 | SE-ResNet-18 F1 | DSCNet Acc (%) | ResNet-18 Acc (%) | SE-ResNet-18 Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Anthracnose** | 0.9719 | 0.9524 | 0.9674 | 95.00% | 95.00% | 96.50% |
| **Aphids** | 0.9723 | 0.9848 | 0.9848 | 96.50% | 97.00% | 97.00% |
| **Healthy** | 0.9561 | 0.9779 | 0.9526 | 98.00% | 99.50% | 95.50% |
| **Powdery_Mildew** | 0.9975 | 0.9926 | 0.9926 | 100.00% | 100.00% | 100.00% |
| **Rust** | 0.9925 | 0.9901 | 0.9926 | 99.50% | 100.00% | 100.00% |
| **Scorch_Sunburn** | 0.9653 | 0.9773 | 0.9851 | 97.50% | 97.00% | 99.50% |
| **Yellowing_Wilt** | 0.9899 | 0.9899 | 0.9848 | 98.00% | 98.00% | 97.50% |