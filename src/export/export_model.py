"""
Script chuyen doi mo hinh tot nhat (Best Model: DSCNet / ResNet18 / SE-ResNet18) sang ONNX
Chuan bi dong goi vao Mobile App / Web Camera Real-time
Tuong thich PyTorch 2.6+ bang cach kiem tra inspect.signature(torch.onnx.export) va truyen dynamo=False.
"""

import os
import sys
import json
import inspect
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.dscnet import DSCNet
from src.models.resnet18 import ResNet18
from src.models.se_resnet18 import SEResNet18

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")


def _build_model(model_name: str, num_classes: int = 7):
    name = model_name.lower()
    if name == "dscnet":
        return DSCNet(num_classes=num_classes)
    elif name == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif name == "se_resnet18":
        return SEResNet18(num_classes=num_classes)
    raise ValueError(f"Khong ho tro kien truc mo hinh: {model_name}")


def resolve_best_model_name() -> str:
    """Doc ten Best Model tu weights/best_model_info.json hoac quet checkpoint co val_f1 cao nhat."""
    best_info_path = os.path.join(WEIGHTS_DIR, "best_model_info.json")
    if os.path.exists(best_info_path):
        try:
            with open(best_info_path, "r", encoding="utf-8-sig") as f:
                info = json.load(f)
            best_name = (info.get("best_model") or info.get("best_model_name") or "").lower()
            if best_name in ("dscnet", "resnet18", "se_resnet18"):
                return best_name
        except Exception:
            pass

    best_f1 = -1.0
    best_candidate = "dscnet"
    for m_name in ("dscnet", "resnet18", "se_resnet18"):
        ckpt_path = os.path.join(WEIGHTS_DIR, f"{m_name}_best.pth")
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                f1_val = float(ckpt.get("val_f1", 0.0))
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_candidate = m_name
            except Exception:
                continue
    return best_candidate


def export_to_onnx(model_name="best", num_classes=7):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    resolved_name = resolve_best_model_name() if model_name.lower() == "best" else model_name.lower()

    ckpt_path = os.path.join(WEIGHTS_DIR, f"{resolved_name}_best.pth")
    onnx_path = os.path.join(WEIGHTS_DIR, f"{resolved_name}.onnx")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "classes" in checkpoint and checkpoint["classes"]:
            num_classes = len(checkpoint["classes"])
        model = _build_model(resolved_name, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[*] Da nap trong so tot nhat ({resolved_name.upper()}) tu: {ckpt_path}")
    else:
        model = _build_model(resolved_name, num_classes=num_classes)
        print(f"[!] Chua co checkpoint '{ckpt_path}', xuat mo hinh {resolved_name.upper()} voi trong so khoi tao.")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    export_kwargs = {
        "export_params": True,
        "opset_version": 14,
        "do_constant_folding": True,
        "input_names": ["input_image"],
        "output_names": ["class_probabilities"],
        "dynamic_axes": {
            "input_image": {0: "batch_size"},
            "class_probabilities": {0: "batch_size"},
        },
    }

    # Kiem tra inspect.signature(torch.onnx.export): neu co tham so 'dynamo' thi truyen dynamo=False
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        export_kwargs["dynamo"] = False

    print(f"[*] Dang xuat mo hinh {resolved_name.upper()} sang dinh dang ONNX tai: {onnx_path} ...")
    torch.onnx.export(model, dummy_input, onnx_path, **export_kwargs)
    size_mb = os.path.getsize(onnx_path) / (1024.0 * 1024.0)
    print(f"[+] Xuat thanh cong file ONNX: {onnx_path} (Dung luong: {size_mb:.2f} MB)")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xuat mo hinh sang dinh dang ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="best",
        choices=["best", "dscnet", "resnet18", "se_resnet18", "all"],
        help="Chon kien truc xuat ONNX (mac dinh: best doc tu weights/best_model_info.json)",
    )
    args = parser.parse_args()

    if args.model == "all":
        for m in ("dscnet", "resnet18", "se_resnet18"):
            export_to_onnx(m)
    else:
        export_to_onnx(args.model)
