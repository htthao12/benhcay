"""
Script chuyen doi mo hinh DSCNet tot nhat sang ONNX
Chuan bi dong goi vao Mobile App / Web Camera Real-time
"""

import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.dscnet import DSCNet

WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "weights"))

def export_to_onnx(model_name="dscnet", num_classes=7):
    ckpt_path = os.path.join(WEIGHTS_DIR, f"{model_name}_best.pth")
    onnx_path = os.path.join(WEIGHTS_DIR, f"{model_name}.onnx")

    model = DSCNet(num_classes=num_classes)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[*] Da nap trong so tot nhat tu: {ckpt_path}")
    else:
        print("[!] Chua co checkpoint, xuat mo hinh voi trong so khoi tao de kiem thu pipeline.")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    print(f"[*] Dang xuat mo hinh sang dinh dang ONNX tai: {onnx_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['class_probabilities'],
        dynamic_axes={'input_image': {0: 'batch_size'}, 'class_probabilities': {0: 'batch_size'}}
    )
    print(f"[+] Xuat thanh cong file ONNX: {onnx_path} (Dung luong: {os.path.getsize(onnx_path)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    export_to_onnx()

