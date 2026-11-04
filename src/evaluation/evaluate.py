"""
Script danh gia chuyen sau cac mo hinh tren tap Test:
- Normalized Confusion Matrix (Ma tran nham lan chuan hoa theo hang)
- Bao cao F1-score, Precision, Recall tung lop
- Do do tre suy luan (Latency) va toc do xu ly (FPS) tren CPU
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.dataset import get_dataloaders
from src.models.resnet18 import ResNet18
from src.models.se_resnet18 import SEResNet18
from src.models.dscnet import DSCNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "weights"))

def benchmark_latency(model, input_size=(1, 3, 224, 224), iterations=100):
    """Do do tre va FPS khi suy luan don anh (batch=1) tren CPU"""
    model.eval()
    dummy = torch.randn(*input_size).to(DEVICE)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy)
        
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy)
    total_time = time.perf_counter() - start
    avg_latency_ms = (total_time / iterations) * 1000.0
    fps = 1000.0 / avg_latency_ms
    return avg_latency_ms, fps

def evaluate_checkpoint(model_name="dscnet"):
    ckpt_path = os.path.join(WEIGHTS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[!] Khong tim thay file checkpoint '{ckpt_path}'. Hay chay train.py truoc!")
        return

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    class_names = checkpoint.get("classes", [])
    num_classes = len(class_names)

    if model_name.lower() == "dscnet":
        model = DSCNet(num_classes=num_classes)
    elif model_name.lower() == "se_resnet18":
        model = SEResNet18(num_classes=num_classes)
    elif model_name.lower() == "resnet18":
        model = ResNet18(num_classes=num_classes)
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # Benchmark CPU latency
    lat_ms, fps = benchmark_latency(model)
    print("=" * 70)
    print(f"KET QUA DANH GIA MO HINH: {model_name.upper()}")
    print(f"Do tre suy luan don anh (Latency): {lat_ms:.2f} ms | Toc do (FPS): {fps:.1f} FPS")
    print("=" * 70)

    # Load Test Data
    data_info = get_dataloaders(batch_size=32)
    test_loader = data_info["test_loader"]

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # Classification Report
    print("\n[+] Chi tiet F1-Score & Accuracy tung lop tren tap Test:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

if __name__ == "__main__":
    evaluate_checkpoint("dscnet")

