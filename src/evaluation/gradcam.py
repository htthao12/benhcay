"""
Module Giai thich Mo hinh Hoc sau (Explainable AI - XAI):
Cai dat thuan PyTorch ca 2 thuat toan:
1. Grad-CAM (Selvaraju et al., ICCV 2017)
2. Grad-CAM++ (Chattopadhay et al., WACV 2018)
Ho tro day du 3 kien truc: DSCNet, ResNet18, SE-ResNet18.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.dataset import get_data_transforms, resolve_data_dir, IMAGE_SIZE
from src.models.dscnet import DSCNet
from src.models.resnet18 import ResNet18
from src.models.se_resnet18 import SEResNet18

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEST_DATA_DIR = os.path.join(resolve_data_dir(), "test")

MODEL_FOLDER_MAP = {
    "dscnet": "1_dscnet",
    "resnet18": "2_resnet18",
    "se_resnet18": "3_se_resnet18",
}

DEFAULT_CLASSES = [
    "Anthracnose",
    "Aphids",
    "Healthy",
    "Powdery_Mildew",
    "Rust",
    "Scorch_Sunburn",
    "Yellowing_Wilt",
]


def get_target_layer(model):
    """
    Xac dinh tang Tich chap cuoi cung (Target Layer) cho Grad-CAM / Grad-CAM++:
    - DSCNet: model.head_conv
    - ResNet18 / SEResNet18: model.stage4[-1]
    """
    if hasattr(model, "head_conv"):
        return model.head_conv
    if hasattr(model, "stage4"):
        return model.stage4[-1]
    raise ValueError(f"Khong tim thay target layer phu hop cho mo hinh: {type(model).__name__}")


def _normalize_cam(cam_tensor: torch.Tensor, out_size=(224, 224)) -> np.ndarray:
    """Noi suy bilinear len out_size va chuan hoa min-max ve [0, 1]."""
    cam_resized = F.interpolate(
        cam_tensor,
        size=out_size,
        mode="bilinear",
        align_corners=False,
    )
    cam_np = cam_resized.squeeze().detach().cpu().numpy()
    cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
    if cam_max - cam_min > 1e-8:
        cam_np = (cam_np - cam_min) / (cam_max - cam_min)
    else:
        cam_np = np.zeros_like(cam_np, dtype=np.float32)
    return cam_np.astype(np.float32)


class GradCAMExtractor:
    """
    Trich xuat dong thoi Grad-CAM va Grad-CAM++ chi trong 1 lan forward + backward.
    """

    def __init__(self, model: torch.nn.Module, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else get_target_layer(model)
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, input_tensor: torch.Tensor, target_class=None, out_size=(IMAGE_SIZE, IMAGE_SIZE)):
        """
        Tinh dong thoi ban do nhiet Grad-CAM va Grad-CAM++ trong 1 lan backward.
        Tra ve dict chua:
        - 'gradcam': np.ndarray (H, W) in [0, 1]
        - 'gradcampp': np.ndarray (H, W) in [0, 1]
        - 'target_class': int
        - 'confidence': float
        - 'probs': np.ndarray (num_classes,)
        """
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            inp = input_tensor.clone().detach().requires_grad_(True)
            logits = self.model(inp)
            probs = F.softmax(logits, dim=1)

            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())

            score = logits[0, target_class]
            score.backward(retain_graph=False)

        A = self.activations  # (1, C, h, w)
        G = self.gradients    # (1, C, h, w)

        # 1. Grad-CAM: weights_gc = mean(G, dim=(2, 3))
        weights_gc = torch.mean(G, dim=(2, 3), keepdim=True)
        cam_gc = F.relu(torch.sum(weights_gc * A, dim=1, keepdim=True))

        # 2. Grad-CAM++:
        G_2 = G ** 2
        G_3 = G ** 3
        sum_A_G3 = torch.sum(A * G_3, dim=(2, 3), keepdim=True)
        alpha_pp = G_2 / (2.0 * G_2 + sum_A_G3 + 1e-7)
        weights_gcpp = torch.sum(alpha_pp * F.relu(G), dim=(2, 3), keepdim=True)
        cam_gcpp = F.relu(torch.sum(weights_gcpp * A, dim=1, keepdim=True))

        cam_gc_np = _normalize_cam(cam_gc, out_size=out_size)
        cam_gcpp_np = _normalize_cam(cam_gcpp, out_size=out_size)
        probs_np = probs[0].detach().cpu().numpy()
        conf = float(probs_np[target_class])

        return {
            "gradcam": cam_gc_np,
            "gradcampp": cam_gcpp_np,
            "target_class": target_class,
            "confidence": conf,
            "probs": probs_np,
        }


def apply_heatmap_overlay(pil_img: Image.Image, cam_map: np.ndarray, alpha=0.45, colormap="jet"):
    """
    Chuyen ban do nhiet [0, 1] thanh anh RGB heatmap va anh chong lop (overlay) len anh goc.
    Tra ve (heatmap_rgb, overlay_rgb) kieu uint8.
    """
    h, w = cam_map.shape[:2]
    img_resized = pil_img.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
    img_np = np.asarray(img_resized, dtype=np.float32)

    cmap = plt.get_cmap(colormap) if hasattr(plt, "get_cmap") else cm.get_cmap(colormap)
    heatmap_rgba = cmap(np.clip(cam_map, 0.0, 1.0))
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255.0).astype(np.uint8)

    overlay_float = (1.0 - alpha) * img_np + alpha * heatmap_rgb.astype(np.float32)
    overlay_rgb = np.clip(overlay_float, 0, 255).astype(np.uint8)
    return heatmap_rgb, overlay_rgb


def _build_model_instance(model_name: str, num_classes: int):
    name = model_name.lower()
    if name == "dscnet":
        return DSCNet(num_classes=num_classes)
    elif name == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif name == "se_resnet18":
        return SEResNet18(num_classes=num_classes)
    raise ValueError(f"Khong ho tro model: {model_name}")


def _load_model_for_gradcam(model_name: str, device):
    ckpt_path = os.path.join(WEIGHTS_DIR, f"{model_name.lower()}_best.pth")
    class_names = list(DEFAULT_CLASSES)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if "classes" in ckpt and ckpt["classes"]:
            class_names = ckpt["classes"]
        model = _build_model_instance(model_name, num_classes=len(class_names))
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        if os.path.exists(TEST_DATA_DIR):
            subdirs = sorted([d for d in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, d))])
            if subdirs:
                class_names = subdirs
        model = _build_model_instance(model_name, num_classes=len(class_names))
    model = model.to(device)
    model.eval()
    return model, class_names


def _get_representative_test_images(class_names, data_dir=None):
    """Lay anh dai dien dau tien cua moi lop trong data/test/<Class>/."""
    test_dir = os.path.join(resolve_data_dir(data_dir), "test") if data_dir else TEST_DATA_DIR
    rep_images = []
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if not files:
            continue
        rep_images.append((cls_idx, cls_name, os.path.join(cls_dir, files[0])))
    return rep_images


def generate_gradcam_for_model(model_name: str, data_dir=None):
    """
    Xuat bo anh Grad-CAM va Grad-CAM++ cho mo hinh model_name:
    - 7 anh chi tiet tung lop (1x5 khung) vao results/<folder>/gradcam/<Class>_gradcam_vs_gradcampp.png
    - 1 anh tong hop 7 lop (7x5 khung) vao results/<folder>/gradcam/<model>_gradcam_overview_7classes.png
    """
    name = model_name.lower()
    if name not in MODEL_FOLDER_MAP:
        raise ValueError(f"Model khong hop le: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = _load_model_for_gradcam(name, device)
    _, val_tf = get_data_transforms()

    rep_images = _get_representative_test_images(class_names, data_dir=data_dir)
    if not rep_images:
        print(f"[!] Khong tim thay anh test tai {TEST_DATA_DIR} de ve Grad-CAM.")
        return

    folder = MODEL_FOLDER_MAP[name]
    gradcam_dir = os.path.join(RESULTS_DIR, folder, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    extractor = GradCAMExtractor(model)
    overview_rows = []

    for cls_idx, cls_name, img_path in rep_images:
        pil_img = Image.open(img_path).convert("RGB")
        pil_224 = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        inp_tensor = val_tf(pil_img).unsqueeze(0).to(device)

        res = extractor.generate(inp_tensor, target_class=None, out_size=(IMAGE_SIZE, IMAGE_SIZE))
        pred_cls = class_names[res["target_class"]]
        conf_pct = res["confidence"] * 100.0

        gc_hm, gc_ov = apply_heatmap_overlay(pil_224, res["gradcam"], alpha=0.45, colormap="jet")
        gcpp_hm, gcpp_ov = apply_heatmap_overlay(pil_224, res["gradcampp"], alpha=0.45, colormap="jet")

        overview_rows.append({
            "cls_name": cls_name,
            "pred_cls": pred_cls,
            "conf_pct": conf_pct,
            "orig": np.asarray(pil_224),
            "gc_hm": gc_hm,
            "gc_ov": gc_ov,
            "gcpp_hm": gcpp_hm,
            "gcpp_ov": gcpp_ov,
        })

        # Xuat anh chi tiet 1x5 khung cho tung lop
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
        panels = [
            (np.asarray(pil_224), f"Original: {cls_name}"),
            (gc_hm, "Grad-CAM Heatmap"),
            (gc_ov, f"Grad-CAM Overlay\nPred: {pred_cls} ({conf_pct:.1f}%)"),
            (gcpp_hm, "Grad-CAM++ Heatmap"),
            (gcpp_ov, f"Grad-CAM++ Overlay\nPred: {pred_cls} ({conf_pct:.1f}%)"),
        ]
        for ax, (im_arr, title) in zip(axes, panels):
            ax.imshow(im_arr)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.suptitle(f"{name.upper()} — Explainable AI (Grad-CAM vs Grad-CAM++) — Class: {cls_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        per_class_path = os.path.join(gradcam_dir, f"{cls_name}_gradcam_vs_gradcampp.png")
        plt.savefig(per_class_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Xuat anh tong hop 7x5 khung
    n_rows = len(overview_rows)
    fig, axes = plt.subplots(n_rows, 5, figsize=(18, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Original (224x224)",
        "Grad-CAM Heatmap",
        "Grad-CAM Overlay",
        "Grad-CAM++ Heatmap",
        "Grad-CAM++ Overlay",
    ]

    for r_idx, row in enumerate(overview_rows):
        imgs = [row["orig"], row["gc_hm"], row["gc_ov"], row["gcpp_hm"], row["gcpp_ov"]]
        for c_idx, im_arr in enumerate(imgs):
            ax = axes[r_idx, c_idx]
            ax.imshow(im_arr)
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=11, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(
                    f"{row['cls_name']}\n({row['pred_cls']} {row['conf_pct']:.1f}%)",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f"{name.upper()} — Tong hop Grad-CAM & Grad-CAM++ tren 7 Lop Benh Cay Canh", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    overview_path = os.path.join(gradcam_dir, f"{name}_gradcam_overview_7classes.png")
    plt.savefig(overview_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    extractor.remove_hooks()
    print(f"[+] Da xuat bo anh Grad-CAM & Grad-CAM++ cho {name.upper()} tai: {gradcam_dir}")


def generate_3models_gradcam_comparison(data_dir=None):
    """
    Xuat anh luoi 7 hang (7 lop benh) x 7 cot:
    (Anh goc | DSCNet Grad-CAM | DSCNet Grad-CAM++ | ResNet18 Grad-CAM | ResNet18 Grad-CAM++ | SE-ResNet18 Grad-CAM | SE-ResNet18 Grad-CAM++)
    luu tai results/4_comparison_all_models/comparison_gradcam_3models.png.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_tf = get_data_transforms()

    model_names = ["dscnet", "resnet18", "se_resnet18"]
    models_dict = {}
    extractors_dict = {}
    class_names = list(DEFAULT_CLASSES)

    for m_name in model_names:
        model, cls_list = _load_model_for_gradcam(m_name, device)
        if cls_list:
            class_names = cls_list
        models_dict[m_name] = model
        extractors_dict[m_name] = GradCAMExtractor(model)

    rep_images = _get_representative_test_images(class_names, data_dir=data_dir)
    if not rep_images:
        for ext in extractors_dict.values():
            ext.remove_hooks()
        print(f"[!] Khong tim thay anh test tai {resolve_data_dir(data_dir)} de ve doi sanh Grad-CAM 3 mo hinh.")
        return

    comp_dir = os.path.join(RESULTS_DIR, "4_comparison_all_models")
    os.makedirs(comp_dir, exist_ok=True)

    n_rows = len(rep_images)
    fig, axes = plt.subplots(n_rows, 7, figsize=(24, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_headers = [
        "Anh goc (Original)",
        "DSCNet\nGrad-CAM",
        "DSCNet\nGrad-CAM++",
        "ResNet18\nGrad-CAM",
        "ResNet18\nGrad-CAM++",
        "SE-ResNet18\nGrad-CAM",
        "SE-ResNet18\nGrad-CAM++",
    ]

    for r_idx, (cls_idx, cls_name, img_path) in enumerate(rep_images):
        pil_img = Image.open(img_path).convert("RGB")
        pil_224 = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        inp_tensor = val_tf(pil_img).unsqueeze(0).to(device)

        row_panels = [(np.asarray(pil_224), cls_name)]
        for m_name in model_names:
            res = extractors_dict[m_name].generate(inp_tensor, target_class=cls_idx, out_size=(IMAGE_SIZE, IMAGE_SIZE))
            _, gc_ov = apply_heatmap_overlay(pil_224, res["gradcam"], alpha=0.45, colormap="jet")
            _, gcpp_ov = apply_heatmap_overlay(pil_224, res["gradcampp"], alpha=0.45, colormap="jet")
            pred_name = class_names[res["target_class"]]
            conf_pct = res["confidence"] * 100.0
            sub_lbl = f"{pred_name} ({conf_pct:.0f}%)"
            row_panels.append((gc_ov, sub_lbl))
            row_panels.append((gcpp_ov, sub_lbl))

        for c_idx, (im_arr, sub_lbl) in enumerate(row_panels):
            ax = axes[r_idx, c_idx]
            ax.imshow(im_arr)
            if r_idx == 0:
                ax.set_title(col_headers[c_idx], fontsize=11, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(cls_name, fontsize=11, fontweight="bold")
            else:
                ax.set_xlabel(sub_lbl, fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])

    for ext in extractors_dict.values():
        ext.remove_hooks()

    plt.suptitle(
        "Doi sanh Vung Chu y (Grad-CAM vs Grad-CAM++) giua 3 Mo hinh: DSCNet, ResNet18, SE-ResNet18",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(comp_dir, "comparison_gradcam_3models.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Da luu anh doi sanh Grad-CAM 3 mo hinh (7x7) tai: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trich xuat Grad-CAM va Grad-CAM++ cho cac mo hinh")
    parser.add_argument("--model", type=str, default="all", choices=["dscnet", "resnet18", "se_resnet18", "all"])
    parser.add_argument("--data_dir", type=str, default=None, help="Duong dan thu muc data (mac dinh tu dong nhan dien d:/AIRB/.../data)")
    args = parser.parse_args()

    if args.model == "all":
        for m in ["dscnet", "resnet18", "se_resnet18"]:
            generate_gradcam_for_model(m, data_dir=args.data_dir)
        generate_3models_gradcam_comparison(data_dir=args.data_dir)
    else:
        generate_gradcam_for_model(args.model, data_dir=args.data_dir)

