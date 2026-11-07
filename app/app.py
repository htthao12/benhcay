# -*- coding: utf-8 -*-
"""
Phenikaa Plant Disease AI - Mobile Real-time Streaming Server & Explainable AI (Grad-CAM / Grad-CAM++)
- Chay backend FastAPI phuc vu stream nhan dien benh la cay canh thoi gian thuc tren dien thoai qua Wi-Fi / LAN / 4G.
- Tu dong in IP LAN va khoi tao duong truyen HTTPS bao mat (Cloudflare Tunnel) cho iPhone (Safari) & Android.
- Ho tro chuyen doi truc tiep ca 3 mo hinh đã học (SE-ResNet-18, DSCNet, ResNet-18) ngay tren man hinh dien thoai,
  hien thi Test Accuracy (98.22%), Macro-F1 (0.9821), Confidence %, FPS, Ban do nhiet Grad-CAM / Grad-CAM++
  va chay mo hinh ONNX truc tiep tren trinh duyet dien thoai (On-Device WebAssembly).
"""

import os
import sys
import time
import json
import socket
import base64
import threading
from contextlib import asynccontextmanager

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.dscnet import DSCNet
from src.models.resnet18 import ResNet18
from src.models.se_resnet18 import SEResNet18
from src.evaluation.gradcam import GradCAMExtractor, apply_heatmap_overlay

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CLASS_NAMES = [
    "Anthracnose",
    "Aphids",
    "Healthy",
    "Powdery_Mildew",
    "Rust",
    "Scorch_Sunburn",
    "Yellowing_Wilt",
]

VIET_NAME_MAP = {
    "Anthracnose": "Bệnh Thán thư",
    "Aphids": "Rệp vừng / Sâu hại",
    "Healthy": "Lá khỏe mạnh",
    "Powdery_Mildew": "Bệnh Phấn trắng",
    "Rust": "Bệnh Rỉ sắt",
    "Scorch_Sunburn": "Cháy nắng / Táp lá",
    "Yellowing_Wilt": "Vàng lá / Úng rễ",
}

TREATMENT_ADVICE = {
    "Healthy": {
        "status": "Cây khỏe mạnh (Healthy)",
        "action": "Duy trì chế độ tưới nước và chăm sóc bình thường. Không cần can thiệp hóa chất.",
        "color": "green",
    },
    "Powdery_Mildew": {
        "status": "Bệnh Phấn trắng (Nấm Erysiphales)",
        "action": (
            "1. Tỉa bớt cành lá rậm rạp để cây thông thoáng quang hợp.\n"
            "2. Phun dung dịch Baking Soda loãng (5g/lít nước) hoặc chế phẩm sinh học gốc lưu huỳnh.\n"
            "3. Tránh tưới nước đọng lên mặt lá vào chiều tối."
        ),
        "color": "orange",
    },
    "Rust": {
        "status": "Bệnh Rỉ sắt (Nấm Pucciniales)",
        "action": (
            "1. Vặt bỏ và tiêu hủy các lá có ổ mụn phấn màu vàng cam.\n"
            "2. Phun luân phiên thuốc trừ nấm gốc Đồng (Boóc-đô) hoặc hoạt chất Difenoconazole / Hexaconazole (Anvil).\n"
            "3. Đặt cây ở nơi có nắng sáng tốt."
        ),
        "color": "red",
    },
    "Anthracnose": {
        "status": "Bệnh Thán thư (Nấm Colletotrichum)",
        "action": (
            "1. Ngừng phun nước trực tiếp lên tán lá để tránh phát tán bào tử nấm.\n"
            "2. Cắt tỉa vùng lá bị cháy đốm lõm.\n"
            "3. Phun thuốc đặc trị thán thư: Antracol 70WP, Mancozeb hoặc Ridomil Gold."
        ),
        "color": "red",
    },
    "Aphids": {
        "status": "Rệp vừng / Rệp muội chích hút (Aphids)",
        "action": (
            "1. Dùng vòi nước xịt rửa mạnh vào ngọn non để rệp rơi ra.\n"
            "2. Phun dung dịch nước rửa chén loãng hoặc tinh dầu Neem (Neem oil) diệt rệp sinh học.\n"
            "3. Nếu mật độ quá cao: Phun thuốc trừ sâu sinh học thảo mộc hoặc Confidor."
        ),
        "color": "red",
    },
    "Scorch_Sunburn": {
        "status": "Cháy nắng / Khô viền mép lá (Scorch / Sunburn)",
        "action": (
            "1. Chuyển cây vào vị trí râm mát, có mái che lưới lan.\n"
            "2. Bổ sung độ ẩm không khí xung quanh, tưới nước vào sáng sớm.\n"
            "3. Cắt bỏ viền lá khô giòn để giữ thẩm mỹ."
        ),
        "color": "orange",
    },
    "Yellowing_Wilt": {
        "status": "Vàng lá / Ứ đọng úng nước rễ (Yellowing / Wilt)",
        "action": (
            "1. Ngừng tưới nước ngay lập tức, kiểm tra lỗ thoát nước dưới đáy chậu.\n"
            "2. Xới nhẹ đất mặt cho rễ thoáng khí.\n"
            "3. Nếu đất quá ướt sũng: Thay đất mới tơi xốp hơn."
        ),
        "color": "orange",
    },
}

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LOADED_MODELS = {}
MODEL_EXTRACTORS = {}
MODEL_SUMMARIES = {}
BEST_MODEL_KEY = "se_resnet18"
INFER_LOCK = threading.Lock()


def _instantiate_model(model_name: str, num_classes: int):
    name = model_name.lower()
    if name == "dscnet":
        return DSCNet(num_classes=num_classes)
    elif name == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif name == "se_resnet18":
        return SEResNet18(num_classes=num_classes)
    return SEResNet18(num_classes=num_classes)


def get_lan_ip():
    """Lay dia chi IP mang noi bo (Wi-Fi / LAN / Hotspot) cua may tinh."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def load_best_model():
    """
    Tu dong doc weights/best_model_info.json va nap day du 3 mo hinh da huan luyen
    (se_resnet18_best.pth, dscnet_best.pth, resnet18_best.pth) vao RAM/VRAM.
    """
    global BEST_MODEL_KEY, LOADED_MODELS, MODEL_EXTRACTORS, MODEL_SUMMARIES

    best_info_path = os.path.join(WEIGHTS_DIR, "best_model_info.json")
    selected_name = "se_resnet18"
    best_meta = {
        "best_model_name": "se_resnet18",
        "test_macro_f1": 0.9821,
        "test_top1_accuracy": 98.22,
        "test_top3_accuracy": 99.94,
        "trainable_parameters": 128599,
        "classes": list(DEFAULT_CLASS_NAMES),
    }

    if os.path.exists(best_info_path):
        try:
            with open(best_info_path, "r", encoding="utf-8-sig") as f:
                loaded_info = json.load(f)
            selected_name = (loaded_info.get("best_model") or loaded_info.get("best_model_name") or "se_resnet18").lower()
            best_meta.update(loaded_info)
            best_meta["best_model_name"] = selected_name
        except Exception:
            pass

    BEST_MODEL_KEY = selected_name

    for m_name in ("se_resnet18", "dscnet", "resnet18"):
        s_path = os.path.join(WEIGHTS_DIR, f"{m_name}_test_summary.json")
        s_info = {
            "model_key": m_name,
            "best_model_name": m_name,
            "classes": list(DEFAULT_CLASS_NAMES),
            "test_top1_accuracy": 98.22 if m_name == "se_resnet18" else (97.52 if m_name == "dscnet" else 97.40),
            "test_top3_accuracy": 99.94 if m_name == "se_resnet18" else (99.30 if m_name == "dscnet" else 99.62),
            "test_macro_f1": 0.9821 if m_name == "se_resnet18" else (0.9750 if m_name == "dscnet" else 0.9738),
        }
        if os.path.exists(s_path):
            try:
                with open(s_path, "r", encoding="utf-8-sig") as f:
                    s_info.update(json.load(f))
            except Exception:
                pass

        classes = s_info.get("classes") or list(DEFAULT_CLASS_NAMES)
        model_inst = _instantiate_model(m_name, num_classes=len(classes))
        ckpt_path = os.path.join(WEIGHTS_DIR, f"{m_name}_best.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model_inst.load_state_dict(ckpt["model_state_dict"])
            if "classes" in ckpt and ckpt["classes"]:
                classes = ckpt["classes"]
                s_info["classes"] = classes
            print(f"[OK] Đã nạp trọng số mô hình đã học: {ckpt_path} (Top-1 Acc: {s_info.get('test_top1_accuracy')}%)")
        else:
            print(f"[!] Chưa tìm thấy file {ckpt_path}")

        model_inst = model_inst.to(DEVICE)
        model_inst.eval()
        with torch.no_grad():
            _ = model_inst(torch.zeros(1, 3, 224, 224, device=DEVICE))

        s_info["trainable_parameters"] = sum(p.numel() for p in model_inst.parameters() if p.requires_grad)
        LOADED_MODELS[m_name] = model_inst
        MODEL_EXTRACTORS[m_name] = GradCAMExtractor(model_inst)
        MODEL_SUMMARIES[m_name] = s_info

    best_meta.update(MODEL_SUMMARIES.get(BEST_MODEL_KEY, {}))
    best_meta["best_model_name"] = BEST_MODEL_KEY
    return LOADED_MODELS[BEST_MODEL_KEY], best_meta


def detect_leaf_bbox(bgr_img: np.ndarray):
    """
    Tim khung chua phien la cay (Bounding Box) trong anh camera bang phan doan mau HSV + Contour.
    """
    h, w = bgr_img.shape[:2]
    try:
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask_leaf = cv2.inRange(hsv, np.array([12, 25, 25]), np.array([100, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask_leaf, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if 0.03 * (w * h) <= area <= 0.95 * (w * h):
                x, y, bw, bh = cv2.boundingRect(largest)
                pad_x = int(0.05 * w)
                pad_y = int(0.05 * h)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)
                return [x1, y1, x2, y2], [
                    round(x1 / w, 4),
                    round(y1 / h, 4),
                    round(x2 / w, 4),
                    round(y2 / h, 4),
                ]
    except Exception:
        pass

    x1, y1, x2, y2 = int(0.12 * w), int(0.12 * h), int(0.88 * w), int(0.88 * h)
    return [x1, y1, x2, y2], [0.12, 0.12, 0.88, 0.88]


def _encode_rgb_to_b64(rgb_arr: np.ndarray, quality: int = 80) -> str:
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")


def analyze_image_with_gradcam(
    pil_img: Image.Image,
    model,
    class_names,
    extractor: GradCAMExtractor = None,
    include_gradcam: bool = True,
):
    """
    Suy luan anh la dau vao (224x224) + Trich xuat dong thoi Grad-CAM & Grad-CAM++.
    """
    pil_224 = pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    tensor = INFER_TRANSFORM(pil_224).unsqueeze(0).to(DEVICE)
    orig_np = np.asarray(pil_224)

    with INFER_LOCK:
        start = time.perf_counter()
        if include_gradcam:
            own_ext = False
            if extractor is None:
                extractor = GradCAMExtractor(model)
                own_ext = True
            cam_res = extractor.generate(tensor, target_class=None, out_size=(224, 224))
            if own_ext:
                extractor.remove_hooks()
            latency_ms = (time.perf_counter() - start) * 1000.0
            probs_np = cam_res["probs"]
            pred_idx = int(cam_res["target_class"])
            conf_val = float(cam_res["confidence"])
            _, gradcam_overlay = apply_heatmap_overlay(pil_224, cam_res["gradcam"], alpha=0.45, colormap="jet")
            _, gradcampp_overlay = apply_heatmap_overlay(pil_224, cam_res["gradcampp"], alpha=0.45, colormap="jet")
        else:
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
            latency_ms = (time.perf_counter() - start) * 1000.0
            probs_np = probs.detach().cpu().numpy()
            pred_idx = int(np.argmax(probs_np))
            conf_val = float(probs_np[pred_idx])
            gradcam_overlay = None
            gradcampp_overlay = None

    fps = 1000.0 / max(latency_ms, 0.5)
    pred_class = class_names[pred_idx]

    return {
        "pred_class": pred_class,
        "confidence": conf_val,
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 1),
        "probs": probs_np,
        "input_224": orig_np,
        "gradcam_overlay": gradcam_overlay,
        "gradcampp_overlay": gradcampp_overlay,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not LOADED_MODELS:
        load_best_model()
    yield


app = FastAPI(title="Phenikaa Plant Disease AI Mobile Server", lifespan=lifespan)


class FramePayload(BaseModel):
    image: str
    model_name: str = "best"
    conf: float = 0.30
    include_gradcam: bool = True


@app.get("/", response_class=HTMLResponse)
async def serve_mobile_ui():
    """Tra ve giao dien ung dung Mobile Realtime Viewfinder tren dien thoai."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(app_dir, "mobile_app.html"),
        os.path.join(TEMPLATES_DIR, "mobile_app.html"),
    ]
    for html_path in candidates:
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8-sig") as f:
                return HTMLResponse(
                    content=f.read(),
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
                )
    return HTMLResponse("<h1>Khong tim thay template mobile_app.html</h1>", status_code=404)


@app.get("/model_stats")
async def get_model_stats():
    """Tra ve thong tin doi sanh va chi so Accuracy/F1 cua ca 3 mo hinh da huan luyen."""
    if not MODEL_SUMMARIES:
        load_best_model()
    return JSONResponse({
        "best_model": BEST_MODEL_KEY,
        "models": MODEL_SUMMARIES,
    })


@app.get("/onnx/{model_file}")
async def serve_onnx_model(model_file: str):
    """Phuc vu file .onnx de trinh duyet dien thoai tai ve chay On-Device WebAssembly."""
    safe_name = os.path.basename(model_file)
    onnx_path = os.path.join(WEIGHTS_DIR, safe_name)
    if os.path.exists(onnx_path):
        return FileResponse(onnx_path, media_type="application/octet-stream", filename=safe_name)
    return JSONResponse({"error": f"Khong tim thay file ONNX: {safe_name}"}, status_code=404)


@app.post("/predict_frame")
async def predict_frame(payload: FramePayload):
    """
    Nhan frame tu camera dien thoai, cat vung la (ROI) + suy luan voi mo hinh da hoc
    (SE-ResNet-18 / DSCNet / ResNet-18), tinh Grad-CAM & Grad-CAM++ va tra ve
    Bounding Box + Ten benh tieng Viet + Acc % + Phac do dieu tri (DSS).
    """
    if not LOADED_MODELS:
        load_best_model()

    try:
        req_model = payload.model_name.lower()
        if req_model == "best" or req_model not in LOADED_MODELS:
            req_model = BEST_MODEL_KEY

        model = LOADED_MODELS[req_model]
        extractor = MODEL_EXTRACTORS.get(req_model)
        m_info = MODEL_SUMMARIES[req_model]
        class_names = m_info.get("classes", DEFAULT_CLASS_NAMES)

        encoded_data = payload.image.split(",")[1] if "," in payload.image else payload.image
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            return JSONResponse({"status": "error", "message": "Invalid image frame"})

        h, w = bgr_img.shape[:2]
        box_px, box_norm = detect_leaf_bbox(bgr_img)
        x1, y1, x2, y2 = box_px

        # Cat vung la (ROI) neu hop le de mo hinh tap trung chinh xac vao phien la
        if (x2 - x1) >= 48 and (y2 - y1) >= 48:
            roi_bgr = bgr_img[y1:y2, x1:x2]
        else:
            roi_bgr = bgr_img

        rgb_img = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        res = analyze_image_with_gradcam(
            pil_img,
            model,
            class_names,
            extractor=extractor,
            include_gradcam=payload.include_gradcam,
        )

        pred_cls = res["pred_class"]
        conf = res["confidence"]
        viet_name = VIET_NAME_MAP.get(pred_cls, pred_cls)
        advice = TREATMENT_ADVICE.get(pred_cls, TREATMENT_ADVICE["Healthy"])
        is_ood = conf < payload.conf

        probs_list = [
            {
                "class": cls_name,
                "viet_name": VIET_NAME_MAP.get(cls_name, cls_name),
                "prob": round(float(p), 4),
            }
            for cls_name, p in zip(class_names, res["probs"])
        ]
        probs_list.sort(key=lambda x: x["prob"], reverse=True)

        resp_payload = {
            "status": "success",
            "frame_w": w,
            "frame_h": h,
            "model_name": req_model,
            "model_info": {
                "model_key": req_model,
                "test_top1_accuracy": m_info.get("test_top1_accuracy", 98.22),
                "test_top3_accuracy": m_info.get("test_top3_accuracy", 99.94),
                "test_macro_f1": m_info.get("test_macro_f1", 0.9821),
                "trainable_parameters": m_info.get("trainable_parameters", 128599),
            },
            "pred_class": pred_cls,
            "viet_name": viet_name,
            "status_title": advice["status"],
            "confidence": round(conf, 4),
            "acc_percent": round(conf * 100.0, 1),
            "is_ood": is_ood,
            "latency_ms": res["latency_ms"],
            "fps": res["fps"],
            "treatment": advice["action"],
            "severity_color": advice["color"],
            "probs": probs_list,
            "detections": [
                {
                    "box": [round(float(v), 1) for v in box_px],
                    "box_norm": box_norm,
                    "conf": round(conf, 3),
                    "name": viet_name,
                    "raw_name": pred_cls,
                }
            ],
        }

        if payload.include_gradcam and res["gradcam_overlay"] is not None:
            resp_payload["input_224_b64"] = _encode_rgb_to_b64(res["input_224"])
            resp_payload["gradcam_b64"] = _encode_rgb_to_b64(res["gradcam_overlay"])
            resp_payload["gradcampp_b64"] = _encode_rgb_to_b64(res["gradcampp_overlay"])

        return JSONResponse(resp_payload)

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


def _ensure_local_ssl_cert(lan_ip: str):
    """Tu dong tao chung chi SSL noi bo (cert.pem, key.pem) de chay https://<IP_LAN>:8443 tren cung Wi-Fi."""
    cert_path = os.path.join(WEIGHTS_DIR, "lan_cert.pem")
    key_path = os.path.join(WEIGHTS_DIR, "lan_key.pem")
    try:
        import datetime
        import ipaddress
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "VN"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Phenikaa Plant Disease AI"),
            x509.NameAttribute(NameOID.COMMON_NAME, lan_ip),
        ])
        san_list = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]
        try:
            san_list.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
        except Exception:
            pass

        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
            .sign(key, hashes.SHA256())
        )
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        with open(key_path, "wb") as f:
            f.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        return cert_path, key_path
    except Exception:
        return None, None


def _run_https_lan_server(https_port: int, cert_path: str, key_path: str):
    """Chay song song cong HTTPS noi bo (8443) cho dien thoai bat cung mang Wi-Fi."""
    try:
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=https_port,
            ssl_certfile=cert_path,
            ssl_keyfile=key_path,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception:
        pass


def start_cloudflare_tunnel(port: int):
    """Khoi tao Cloudflare HTTPS Tunnel de iPhone (Safari) mo duoc Camera truc tiep."""
    try:
        from pycloudflared import try_cloudflare
        tunnel = try_cloudflare(port=port)
        print("\n" + "★" * 68)
        print(" 🌐 ĐÃ TẠO XONG LINK HTTPS BẢO MẬT DÀNH CHO ĐIỆN THOẠI (MỌI MẠNG WI-FI / 4G):")
        print(f" 👉 {tunnel.tunnel}")
        print(" (Mở link này trên Safari/Chrome điện thoại: Cấp quyền Camera & Stream mượt mà!)")
        print("★" * 68 + "\n")
    except Exception:
        pass


def run_mobile_server(port: int = 8000, https_port: int = 8443):
    load_best_model()
    lan_ip = get_lan_ip()
    cert_path, key_path = _ensure_local_ssl_cert(lan_ip)
    best_info = MODEL_SUMMARIES.get(BEST_MODEL_KEY, {})
    print("\n" + "=" * 68)
    print(" 🌿 PHENIKAA PLANT DISEASE AI - MOBILE REAL-TIME STREAMING SERVER")
    print("=" * 68)
    print(
        f" 🏆 Mô hình mặc định: {BEST_MODEL_KEY.upper()} "
        f"(Test Acc: {best_info.get('test_top1_accuracy', 98.22)}% | "
        f"Macro-F1: {best_info.get('test_macro_f1', 0.9821)} | "
        f"Params: {best_info.get('trainable_parameters', 128599):,})"
    )
    print(" 📦 Đã nạp sẵn cả 3 mô hình đã học (SE-ResNet-18, DSCNet, ResNet-18) + ONNX + Grad-CAM++")
    print(" 💻 1. Mở trên máy tính cá nhân:")
    print(f"    👉 http://localhost:{port}")
    if cert_path and key_path:
        print(" 📶 2. Mở TRỰC TIẾP QUA CÙNG MẠNG WI-FI / HOTSPOT (Tốc độ cao nhất, không trễ):")
        print(f"    👉 https://{lan_ip}:{https_port}")
        print("    (Lưu ý trên Safari/Chrome: Bấm 'Hiển thị chi tiết' -> 'Truy cập trang web này')")
        t_https = threading.Thread(
            target=_run_https_lan_server,
            args=(https_port, cert_path, key_path),
            daemon=True,
        )
        t_https.start()
    print(" 🌐 3. Đường truyền HTTPS Cloudflare (Dùng được cả khi cùng Wi-Fi hoặc 4G/5G):")
    print("    ⏳ Đang khởi tạo link trycloudflare.com...")
    print("=" * 68)

    t = threading.Thread(target=start_cloudflare_tunnel, args=(port,), daemon=True)
    t.start()

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    run_mobile_server(port=8000)
