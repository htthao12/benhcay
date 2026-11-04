"""
Ung dung Real-Time Plant Health Diagnosis (Streamlit Web & Mobile App)
Cho phep bat Camera truc tiep hoac tai anh len de chan doan benh thoi gian thuc tren dien thoai/laptop.
Tich hop he thong goi y phac do dieu tri nong nghiep (Decision Support System - DSS).
"""

import os
import sys
import time
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Thu vien giao dien Streamlit
try:
    import streamlit as st
except ImportError:
    print("[!] Hay cai dat streamlit bang: pip install streamlit")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.dscnet import DSCNet

DEVICE = torch.device("cpu")
WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))

# Bang phac do dieu tri (DSS Treatment Recommendations)
TREATMENT_ADVICE = {
    "Healthy": {
        "status": "Cây khỏe mạnh",
        "action": "Duy trì chế độ tưới nước và chăm sóc bình thường. Không cần can thiệp hóa chất.",
        "color": "green"
    },
    "Powdery_Mildew": {
        "status": "Bệnh Phấn trắng (Nấm Erysiphales)",
        "action": "1. Tỉa bớt cành lá rậm rạp để cây thông thoáng quang hợp.\n2. Phun dung dịch Baking Soda loãng (5g/lít nước) hoặc chế phẩm sinh học gốc lưu huỳnh.\n3. Tránh tưới nước đọng lên mặt lá vào chiều tối.",
        "color": "orange"
    },
    "Rust": {
        "status": "Bệnh Rỉ sắt (Nấm Pucciniales)",
        "action": "1. Vặt bỏ và tiêu hủy các lá có ổ mụn phấn màu vàng cam.\n2. Phun luân phiên thuốc trừ nấm gốc Đồng (Boóc-đô) hoặc hoạt chất Difenoconazole / Hexaconazole (Anvil).\n3. Đặt cây ở nơi có nắng sáng tốt.",
        "color": "red"
    },
    "Anthracnose": {
        "status": "Bệnh Thán thư (Nấm Colletotrichum)",
        "action": "1. Ngừng phun nước trực tiếp lên tán lá để tránh phát tán bào tử nấm.\n2. Cắt tỉa vùng lá bị cháy đốm lõm.\n3. Phun thuốc đặc trị thán thư: Antracol 70WP, Mancozeb hoặc Ridomil Gold.",
        "color": "red"
    },
    "Aphids": {
        "status": "Rệp vừng / Rệp muội chích hút",
        "action": "1. Dùng vòi nước xịt rửa mạnh vào ngọn non để rệp rơi ra.\n2. Phun dung dịch nước rửa chén loãng hoặc tinh dầu Neem (Neem oil) diệt rệp sinh học.\n3. Nếu mật độ quá cao: Phun thuốc trừ sâu sinh học thảo mộc hoặc Confidor.",
        "color": "red"
    },
    "Scorch_Sunburn": {
        "status": "Cháy nắng / Khô viền mép lá",
        "action": "1. Chuyển cây vào vị trí râm mát, có mái che lưới lan.\n2. Bổ sung độ ẩm không khí xung quanh, tưới nước vào sáng sớm.\n3. Cắt bỏ viền lá khô giòn để giữ thẩm mỹ.",
        "color": "orange"
    },
    "Yellowing_Wilt": {
        "status": "Vàng lá / Ứ đọng úng nước rễ",
        "action": "1. Ngừng tưới nước ngay lập tức, kiểm tra lỗ thoát nước dưới đáy chậu.\n2. Xới nhẹ đất mặt cho rễ thoáng khí.\n3. Nếu đất quá ướt sũng: Thay đất mới tơi xốp hơn.",
        "color": "orange"
    }
}

CLASS_NAMES = [
    "Anthracnose",
    "Aphids",
    "Healthy",
    "Powdery_Mildew",
    "Rust",
    "Scorch_Sunburn",
    "Yellowing_Wilt"
]

@st.cache_resource
def load_model():
    model = DSCNet(num_classes=len(CLASS_NAMES))
    ckpt_path = os.path.join(WEIGHTS_DIR, "dscnet_best.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict_image(image: Image.Image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    latency_ms = (time.perf_counter() - start) * 1000.0

    conf, idx = torch.max(probs, dim=0)
    pred_class = CLASS_NAMES[idx.item()]
    conf_val = conf.item()

    return pred_class, conf_val, latency_ms, probs.numpy()

def main():
    st.set_page_config(page_title="Chẩn đoán Bệnh Cây Cảnh Phenikaa", page_icon="🌿", layout="centered")
    st.title("🌿 REAL-TIME PLANT HEALTH DIAGNOSIS")
    st.caption("Hệ thống chẩn đoán bệnh & sâu hại cây cảnh khuôn viên ĐH Phenikaa (Mô hình DSCNet siêu nhẹ)")

    model = load_model()

    tab1, tab2 = st.tabs(["📷 Quét Camera Trực Tiếp", "📁 Tải Ảnh Lên"])

    with tab1:
        st.write("Chĩa camera vào phiến lá cây cảnh (Hoa hồng, Dâm bụt, Ngọc bút, Lưỡi hổ, Trầu bà):")
        camera_img = st.camera_input("Bật Camera")
        if camera_img:
            img = Image.open(camera_img).convert("RGB")
            process_and_display(img, model)

    with tab2:
        file = st.file_uploader("Chọn ảnh lá cây từ thư viện máy tính / điện thoại", type=["jpg", "jpeg", "png"])
        if file:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Ảnh đầu vào", use_column_width=True)
            process_and_display(img, model)

def process_and_display(img, model):
    pred_class, conf, latency_ms, all_probs = predict_image(img, model)

    st.divider()
    # Kiem tra Out-of-Distribution (OOD)
    if conf < 0.55:
        st.warning("⚠️ **CẢNH BÁO: PHÁT HIỆN LÁ CÂY LẠ (Out-of-Distribution)**\n\n"
                   f"Độ tin cậy thấp ({conf*100:.1f}%). Chiếc lá này có thể không thuộc các loài cây hệ thống hỗ trợ. "
                   "Hệ thống từ chối chẩn đoán để tránh sai lệch.")
        return

    advice = TREATMENT_ADVICE.get(pred_class, TREATMENT_ADVICE["Healthy"])
    st.subheader(f"Kết quả: {advice['status']}")
    st.metric(label="Độ tin cậy (Confidence)", value=f"{conf * 100:.1f}%", delta=f"{latency_ms:.1f} ms (Inference)")

    with st.expander("🩺 Phác đồ điều trị khuyến nghị (Decision Support)", expanded=True):
        st.write(advice["action"])

    with st.expander("📊 Phân phối xác suất các lớp"):
        for cls_name, p in zip(CLASS_NAMES, all_probs):
            st.progress(float(p), text=f"{cls_name}: {p*100:.1f}%")

if __name__ == "__main__":
    if "streamlit" in sys.modules:
        main()
    else:
        print("[*] Chay ung dung bang lenh: streamlit run app/app.py")

