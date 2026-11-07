"""
Script danh gia chuyen sau va doi sanh toan dien 3 mo hinh (DSCNet, ResNet18, SE-ResNet18) tren tap Test.
Quy hoach ket qua vao cau truc thu muc:
- results/1_dscnet/
- results/2_resnet18/
- results/3_se_resnet18/
- results/4_comparison_all_models/
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import csv
import time
import json
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.dataset import get_dataloaders
from src.models.dscnet import DSCNet
from src.models.resnet18 import ResNet18
from src.models.se_resnet18 import SEResNet18
from src.evaluation.gradcam import (
    MODEL_FOLDER_MAP,
    generate_gradcam_for_model,
    generate_3models_gradcam_comparison,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_MODELS = ["dscnet", "resnet18", "se_resnet18"]


def _create_model(model_name: str, num_classes: int):
    name = model_name.lower()
    if name == "dscnet":
        return DSCNet(num_classes=num_classes)
    elif name == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif name == "se_resnet18":
        return SEResNet18(num_classes=num_classes)
    raise ValueError(f"Khong ho tro model '{model_name}'")


def benchmark_latency(model, input_size=(1, 3, 224, 224), iterations=100, target_device="cpu"):
    """Do do tre (ms) va toc do xu ly (FPS) khi suy luan don anh (Batch=1) tren target_device."""
    dev = torch.device(target_device)
    orig_device = next(model.parameters()).device
    model = model.to(dev)
    model.eval()
    dummy = torch.randn(*input_size, device=dev)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start

    model.to(orig_device)
    avg_latency_ms = (total_time / iterations) * 1000.0
    fps = 1000.0 / max(avg_latency_ms, 1e-6)
    return float(avg_latency_ms), float(fps)


def compute_flops_and_macs(model, input_size=(1, 3, 224, 224)):
    """
    Tinh chinh xac so phep nhan-cong tich luy (MACs) va phep tinh dau phay dong (FLOPs)
    cua mo hinh voi kich thuoc dau vao (1, 3, 224, 224) bang PyTorch forward hooks.
    Tra ve: (flops_m, macs_m) don vi Trieu (Million - MFLOPs, MMACs).
    """
    total_macs = 0
    total_flops = 0
    hooks = []

    def conv_hook(module, inp, out):
        nonlocal total_macs, total_flops
        batch_size, out_c, out_h, out_w = out.shape
        k_h, k_w = module.kernel_size
        in_c = module.in_channels
        groups = module.groups
        macs_per_elem = (in_c // groups) * k_h * k_w
        out_elems = batch_size * out_c * out_h * out_w
        layer_macs = out_elems * macs_per_elem
        layer_flops = 2 * layer_macs + (out_elems if module.bias is not None else 0)
        total_macs += layer_macs
        total_flops += layer_flops

    def linear_hook(module, inp, out):
        nonlocal total_macs, total_flops
        batch_size = out.shape[0] if out.dim() > 1 else 1
        layer_macs = batch_size * module.in_features * module.out_features
        layer_flops = 2 * layer_macs + (batch_size * module.out_features if module.bias is not None else 0)
        total_macs += layer_macs
        total_flops += layer_flops

    def bn_hook(module, inp, out):
        nonlocal total_macs, total_flops
        elems = inp[0].numel()
        total_macs += elems
        total_flops += 2 * elems

    def act_pool_hook(module, inp, out):
        nonlocal total_flops
        total_flops += out.numel()

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(bn_hook))
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Sigmoid, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d)):
            hooks.append(m.register_forward_hook(act_pool_hook))

    orig_device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_size, device=orig_device)
        _ = model(dummy)

    for h in hooks:
        h.remove()
    if was_training:
        model.train()

    flops_m = float(total_flops) / 1e6
    macs_m = float(total_macs) / 1e6
    return flops_m, macs_m


def plot_standalone_loss_and_acc_curves(history: dict, model_name: str, save_dirs: list, best_epoch: int = None):
    """
    Xuat 2 do thi doc lap (300 DPI) phuc vu chen truc tiep vao Slide/Bao cao:
    1. {name}_loss_curve.png (Train Loss vs Validation Loss qua cac Epoch)
    2. {name}_accuracy_curve.png (Train Accuracy vs Validation Accuracy qua cac Epoch)
    """
    if not history or "train_loss" not in history or not history["train_loss"]:
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Do thi Loss Curve rieng biet
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2.2)
    ax.plot(epochs, history["val_loss"], label="Validation Loss", color="#d62728", linewidth=2.2)
    if best_epoch:
        ax.axvline(best_epoch, color="#ff7f0e", linestyle="--", linewidth=1.8, label=f"Best Epoch ({best_epoch})")
    ax.set_title(f"{model_name.upper()} — Đồ thị Hàm mất mát (Train vs Validation Loss)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch (Vòng lặp huấn luyện)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, f"{model_name}_loss_curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Do thi Accuracy Curve rieng biet
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(epochs, history["train_acc"], label="Train Top-1 Accuracy (%)", color="#1f77b4", linewidth=2.2)
    ax.plot(epochs, history["val_acc"], label="Validation Top-1 Accuracy (%)", color="#d62728", linewidth=2.2)
    if "val_top3_acc" in history and history["val_top3_acc"]:
        ax.plot(epochs, history["val_top3_acc"], label="Validation Top-3 Accuracy (%)", color="#2ca02c", linestyle="-.", linewidth=1.8)
    if best_epoch:
        ax.axvline(best_epoch, color="#ff7f0e", linestyle="--", linewidth=1.8, label=f"Best Epoch ({best_epoch})")
    ax.set_title(f"{model_name.upper()} — Đồ thị Độ chính xác (Train vs Validation Accuracy)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch (Vòng lặp huấn luyện)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, f"{model_name}_accuracy_curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_qualitative_inference_grids(records: list, class_names: list, model_name: str, model_res_dir: str):
    """
    Xuat bang chi tiet du doan tren toan bo tap Test ({name}_test_predictions_detail.csv)
    va 2 anh truc quan hoa ket qua suy luan thuc te (300 DPI):
    1. {name}_qualitative_correct_samples.png: Luoi 2x7 (14 anh thuc te tren 7 lop benh) du doan DUNG
       kem nhan True, Pred va Confidence Score (%).
    2. {name}_qualitative_error_analysis.png: Luoi phan tich cac mau du doan SAI (Misclassified)
       hoac co do tin cay thap nhat (Borderline/Hard Cases) de phan tich nguyen nhan khach quan trong bao cao.
    """
    os.makedirs(model_res_dir, exist_ok=True)

    # 1. Xuat CSV chi tiet tung anh Test
    detail_csv = os.path.join(model_res_dir, f"{model_name}_test_predictions_detail.csv")
    fields = [
        "Index",
        "Image_File",
        "True_Class",
        "Predicted_Class",
        "Is_Correct",
        "Confidence (%)",
        "Second_Best_Class",
        "Second_Best_Conf (%)",
        "Image_Path",
    ]
    with open(detail_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "Index": r["index"],
                "Image_File": os.path.basename(r["img_path"]),
                "True_Class": r["true_cls"],
                "Predicted_Class": r["pred_cls"],
                "Is_Correct": "CORRECT" if r["correct"] else "WRONG",
                "Confidence (%)": round(r["conf_pct"], 2),
                "Second_Best_Class": r["second_cls"],
                "Second_Best_Conf (%)": round(r["second_conf_pct"], 2),
                "Image_Path": r["img_path"],
            })

    # 2. Luoi 2x7 (14 mau) du doan DUNG dai dien cho 7 lop benh (2 mau/lop)
    selected_correct = []
    for cls_name in class_names:
        cls_correct = [r for r in records if r["true_cls"] == cls_name and r["correct"]]
        cls_correct.sort(key=lambda x: x["conf_pct"], reverse=True)
        selected_correct.extend(cls_correct[:2])

    if selected_correct:
        n_cols = 7
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 7.2))
        axes_flat = axes.flatten()
        for idx, ax in enumerate(axes_flat):
            col_cls_idx = idx % n_cols
            row_slot = idx // n_cols
            cls_name = class_names[col_cls_idx] if col_cls_idx < len(class_names) else ""
            cls_items = [r for r in selected_correct if r["true_cls"] == cls_name]
            if row_slot < len(cls_items):
                item = cls_items[row_slot]
                img = Image.open(item["img_path"]).convert("RGB").resize((224, 224))
                ax.imshow(np.asarray(img))
                ax.set_title(
                    f"[DUNG] True: {item['true_cls']}\nPred: {item['pred_cls']} ({item['conf_pct']:.1f}%)",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#15803d",
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor("#16a34a")
                    spine.set_linewidth(2.8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")

        plt.suptitle(
            f"{model_name.upper()} — Kết quả Suy luận Thực tế Dự đoán Đúng trên Tập Test (7 Lớp Bệnh Cây Cảnh Phenikaa)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_res_dir, f"{model_name}_qualitative_correct_samples.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 3. Luoi phan tich loi sai & ca kho (Misclassified + Lowest-Confidence Borderline Samples)
    wrong_records = [r for r in records if not r["correct"]]
    wrong_records.sort(key=lambda x: x["conf_pct"], reverse=True)

    correct_low_conf = [r for r in records if r["correct"]]
    correct_low_conf.sort(key=lambda x: x["conf_pct"])

    max_panels = 12
    analysis_items = []
    for r in wrong_records[:max_panels]:
        analysis_items.append(("MISCLASSIFIED (DỰ ĐOÁN SAI)", r, "#dc2626"))
    if len(analysis_items) < max_panels:
        needed = max_panels - len(analysis_items)
        for r in correct_low_conf[:needed]:
            analysis_items.append(("HARD CASE (ĐỘ TIN CẬY THẤP)", r, "#d97706"))

    if analysis_items:
        n_cols = 4
        n_rows = int(np.ceil(len(analysis_items) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows))
        axes_flat = np.array(axes).reshape(-1)
        for idx, ax in enumerate(axes_flat):
            if idx < len(analysis_items):
                tag, item, color_hex = analysis_items[idx]
                img = Image.open(item["img_path"]).convert("RGB").resize((224, 224))
                ax.imshow(np.asarray(img))
                ax.set_title(
                    f"[{tag}]\nTrue: {item['true_cls']}\nPred: {item['pred_cls']} ({item['conf_pct']:.1f}%)\n"
                    f"2nd: {item['second_cls']} ({item['second_conf_pct']:.1f}%)",
                    fontsize=8.5,
                    fontweight="bold",
                    color=color_hex,
                )
                ax.set_xlabel(os.path.basename(item["img_path"]), fontsize=7.5)
                for spine in ax.spines.values():
                    spine.set_edgecolor(color_hex)
                    spine.set_linewidth(2.8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")

        n_err = len(wrong_records)
        plt.suptitle(
            f"{model_name.upper()} — Phân tích Mẫu Dự đoán Sai ({n_err}/{len(records)} ảnh) & Ca Khó (Hard Cases) trên Tập Test Thực tế",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_res_dir, f"{model_name}_qualitative_error_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_confusion_matrix_figure(cm_counts: np.ndarray, class_names, model_name: str, save_paths):
    """
    Ve Ma tran nham lan kep (300 DPI):
    - Khung trai: So luong mau thuc te (Blues)
    - Khung phai: Ty le % chuan hoa theo hang (Greens)
    """
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_pct = np.divide(
        cm_counts.astype(np.float64) * 100.0,
        row_sums,
        out=np.zeros_like(cm_counts, dtype=np.float64),
        where=row_sums != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    n_cls = len(class_names)

    # Khung trai: So luong mau thuc te (Blues)
    im0 = axes[0].imshow(cm_counts, interpolation="nearest", cmap="Blues")
    axes[0].set_title(f"{model_name.upper()} — Ma tran Nham lan (So luong mau)", fontsize=12, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    thresh0 = cm_counts.max() / 2.0 if cm_counts.size else 0
    for i in range(n_cls):
        for j in range(n_cls):
            axes[0].text(
                j,
                i,
                f"{int(cm_counts[i, j])}",
                ha="center",
                va="center",
                color="white" if cm_counts[i, j] > thresh0 else "black",
                fontsize=9,
                fontweight="bold" if i == j else "normal",
            )

    # Khung phai: Ty le % chuan hoa theo hang (Greens)
    im1 = axes[1].imshow(cm_pct, interpolation="nearest", cmap="Greens", vmin=0.0, vmax=100.0)
    axes[1].set_title(f"{model_name.upper()} — Ma tran Nham lan Chuan hoa (%)", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    for i in range(n_cls):
        for j in range(n_cls):
            axes[1].text(
                j,
                i,
                f"{cm_pct[i, j]:.1f}%",
                ha="center",
                va="center",
                color="white" if cm_pct[i, j] > 50.0 else "black",
                fontsize=8.5,
                fontweight="bold" if i == j else "normal",
            )

    for ax in axes:
        ax.set_xticks(np.arange(n_cls))
        ax.set_yticks(np.arange(n_cls))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Nhan du doan (Predicted Label)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Nhan thuc te (True Label)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    for sp in save_paths:
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        plt.savefig(sp, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_checkpoint(model_name="dscnet", batch_size=32, run_gradcam=True, data_dir=None):
    """
    Danh gia toan dien 1 mo hinh tren tap Test va luu day du bao cao vao results/<folder>/ va weights/.
    """
    name = model_name.lower()
    if name not in MODEL_FOLDER_MAP:
        raise ValueError(f"Model khong hop le: {model_name}")

    folder = MODEL_FOLDER_MAP[name]
    model_res_dir = os.path.join(RESULTS_DIR, folder)
    os.makedirs(model_res_dir, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    ckpt_path = os.path.join(WEIGHTS_DIR, f"{name}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[!] Khong tim thay file checkpoint '{ckpt_path}'. Hay chay train.py truoc!")
        return None

    data_info = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    test_loader = data_info["test_loader"]
    test_samples = getattr(test_loader.dataset, "samples", [])

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    class_names = checkpoint.get("classes", data_info["classes"])
    num_classes = len(class_names)
    best_epoch = int(checkpoint.get("epoch", 0))

    model = _create_model(name, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024.0 * 1024.0)
    flops_m, macs_m = compute_flops_and_macs(model, input_size=(1, 3, 224, 224))

    # 1. Benchmark Latency tren CPU va CUDA (neu co)
    cpu_lat_ms, cpu_fps = benchmark_latency(model, target_device="cpu")
    cuda_lat_ms, cuda_fps = (None, None)
    if torch.cuda.is_available():
        cuda_lat_ms, cuda_fps = benchmark_latency(model, target_device="cuda")

    # 2. Danh gia tren tap Test + Thu thap du doan chi tiet tung anh phuc vu Error Analysis
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    sample_records = []
    sample_cursor = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct_top1 += preds.eq(targets).sum().item()

            maxk = min(3, outputs.size(1))
            topk_probs, topk_preds = probs.topk(maxk, 1, True, True)
            correct_top3 += (
                topk_preds.t()
                .eq(targets.view(1, -1).expand_as(topk_preds.t()))[:maxk]
                .reshape(-1)
                .float()
                .sum()
                .item()
            )

            bs = targets.size(0)
            for b_idx in range(bs):
                global_idx = sample_cursor + b_idx
                img_path = test_samples[global_idx][0] if global_idx < len(test_samples) else f"sample_{global_idx}.jpg"
                t_idx = int(targets[b_idx].item())
                p_idx = int(topk_preds[b_idx, 0].item())
                p_conf = float(topk_probs[b_idx, 0].item()) * 100.0
                sec_idx = int(topk_preds[b_idx, 1].item()) if maxk > 1 else p_idx
                sec_conf = float(topk_probs[b_idx, 1].item()) * 100.0 if maxk > 1 else 0.0
                sample_records.append({
                    "index": global_idx + 1,
                    "img_path": img_path,
                    "true_idx": t_idx,
                    "true_cls": class_names[t_idx],
                    "pred_idx": p_idx,
                    "pred_cls": class_names[p_idx],
                    "correct": (t_idx == p_idx),
                    "conf_pct": p_conf,
                    "second_cls": class_names[sec_idx],
                    "second_conf_pct": sec_conf,
                })

            sample_cursor += bs
            total_samples += bs
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    test_loss = float(total_loss / max(1, total_samples))
    test_top1_acc = float((correct_top1 / max(1, total_samples)) * 100.0)
    test_top3_acc = float((correct_top3 / max(1, total_samples)) * 100.0)
    test_macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    test_macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    test_macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    test_weighted_prec = float(precision_score(all_targets, all_preds, average="weighted", zero_division=0))
    test_weighted_rec = float(recall_score(all_targets, all_preds, average="weighted", zero_division=0))

    # 3. Ma tran nham lan kep
    cm_counts = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    cm_paths = [
        os.path.join(model_res_dir, f"{name}_confusion_matrix.png"),
        os.path.join(WEIGHTS_DIR, f"{name}_confusion_matrix.png"),
    ]
    plot_confusion_matrix_figure(cm_counts, class_names, name, cm_paths)

    # 3b. Xuat anh Qualitative Correct Samples, Error Analysis & CSV chi tiet
    generate_qualitative_inference_grids(sample_records, class_names, name, model_res_dir)

    # 4. Bang chi so tung lop (per_class_metrics.csv)
    precs, recs, f1s, supports = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(num_classes)), zero_division=0
    )
    per_class_rows = []
    per_class_dict = {}
    for idx, cls_name in enumerate(class_names):
        cls_total = int(supports[idx])
        cls_correct = int(cm_counts[idx, idx])
        cls_acc_pct = (cls_correct / cls_total * 100.0) if cls_total > 0 else 0.0
        row_item = {
            "Class": cls_name,
            "Precision": round(float(precs[idx]), 4),
            "Recall": round(float(recs[idx]), 4),
            "F1_Score": round(float(f1s[idx]), 4),
            "Class_Accuracy (%)": round(float(cls_acc_pct), 2),
            "Support": cls_total,
        }
        per_class_rows.append(row_item)
        per_class_dict[cls_name] = row_item

    macro_row = {
        "Class": "MACRO_AVERAGE",
        "Precision": round(test_macro_prec, 4),
        "Recall": round(test_macro_rec, 4),
        "F1_Score": round(test_macro_f1, 4),
        "Class_Accuracy (%)": round(
            float(np.mean([r["Class_Accuracy (%)"] for r in per_class_rows])), 2
        ),
        "Support": int(total_samples),
    }
    weighted_row = {
        "Class": "WEIGHTED_AVERAGE",
        "Precision": round(test_weighted_prec, 4),
        "Recall": round(test_weighted_rec, 4),
        "F1_Score": round(test_weighted_f1, 4),
        "Class_Accuracy (%)": round(test_top1_acc, 2),
        "Support": int(total_samples),
    }

    per_class_csv_path = os.path.join(model_res_dir, f"{name}_per_class_metrics.csv")
    fieldnames = ["Class", "Precision", "Recall", "F1_Score", "Class_Accuracy (%)", "Support"]
    with open(per_class_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_class_rows:
            writer.writerow(r)
        writer.writerow(macro_row)
        writer.writerow(weighted_row)

    # Dong bo history/learning_curves tu weights/ sang results/<folder>/ va ve do thi Loss/Acc doc lap
    for suffix in ["_history.json", "_learning_curves.png"]:
        w_file = os.path.join(WEIGHTS_DIR, f"{name}{suffix}")
        r_file = os.path.join(model_res_dir, f"{name}{suffix}")
        if os.path.exists(w_file) and not os.path.exists(r_file):
            shutil.copy2(w_file, r_file)

    train_loss_best = checkpoint.get("train_loss")
    train_acc_best = checkpoint.get("train_acc")
    val_loss_best = checkpoint.get("val_loss")
    val_acc_best = checkpoint.get("val_acc")
    val_f1_best = checkpoint.get("val_f1")

    hist_path = os.path.join(model_res_dir, f"{name}_history.json")
    if not os.path.exists(hist_path):
        hist_path = os.path.join(WEIGHTS_DIR, f"{name}_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8-sig") as f:
            hist_data = json.load(f)
        plot_standalone_loss_and_acc_curves(hist_data, name, [model_res_dir, WEIGHTS_DIR], best_epoch=best_epoch)
        idx_ep = max(0, best_epoch - 1)
        if train_loss_best is None and idx_ep < len(hist_data.get("train_loss", [])):
            train_loss_best = hist_data["train_loss"][idx_ep]
        if train_acc_best is None and idx_ep < len(hist_data.get("train_acc", [])):
            train_acc_best = hist_data["train_acc"][idx_ep]
        if val_loss_best is None and idx_ep < len(hist_data.get("val_loss", [])):
            val_loss_best = hist_data["val_loss"][idx_ep]
        if val_acc_best is None and idx_ep < len(hist_data.get("val_acc", [])):
            val_acc_best = hist_data["val_acc"][idx_ep]
        if val_f1_best is None and idx_ep < len(hist_data.get("val_f1", [])):
            val_f1_best = hist_data["val_f1"][idx_ep]

    # 5. Bao cao van ban classification_report.txt
    cls_report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0)
    report_txt_path = os.path.join(model_res_dir, f"{name}_classification_report.txt")
    with open(report_txt_path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write(f"BAO CAO DANH GIA MO HINH: {name.upper()}\n")
        f.write("=" * 72 + "\n")
        f.write(f"Best Epoch:             {best_epoch}\n")
        f.write(f"Trainable Parameters:   {trainable_params:,} ({trainable_params / 1e6:.3f}M)\n")
        f.write(f"Checkpoint Size (MB):   {ckpt_size_mb:.2f} MB\n")
        f.write(f"Computational FLOPs:    {flops_m:.2f} MFLOPs ({macs_m:.2f} MMACs)\n")
        if train_loss_best is not None and val_loss_best is not None:
            f.write(f"Train Loss / Acc:       {float(train_loss_best):.4f} / {float(train_acc_best):.2f}%\n")
            f.write(f"Val Loss / Acc / F1:    {float(val_loss_best):.4f} / {float(val_acc_best):.2f}% / {float(val_f1_best):.4f}\n")
        f.write(f"Test Loss:              {test_loss:.4f}\n")
        f.write(f"Test Top-1 Accuracy:    {test_top1_acc:.2f}%\n")
        f.write(f"Test Top-3 Accuracy:    {test_top3_acc:.2f}%\n")
        f.write(f"Test Macro-F1:          {test_macro_f1:.4f}\n")
        f.write(f"Test Weighted-F1:       {test_weighted_f1:.4f}\n")
        f.write(f"Test Macro-Precision:   {test_macro_prec:.4f}\n")
        f.write(f"Test Macro-Recall:      {test_macro_rec:.4f}\n")
        f.write(f"CPU Latency (Batch=1):  {cpu_lat_ms:.2f} ms ({cpu_fps:.1f} FPS)\n")
        if cuda_lat_ms is not None:
            f.write(f"GPU Latency (Batch=1):  {cuda_lat_ms:.2f} ms ({cuda_fps:.1f} FPS)\n")
        f.write("=" * 72 + "\n\n")
        f.write(cls_report_str + "\n")

    # 6. Luu test_summary.json tai results/<folder>/ va weights/
    summary_data = {
        "model_name": name,
        "best_epoch": best_epoch,
        "trainable_parameters": trainable_params,
        "checkpoint_size_mb": round(ckpt_size_mb, 3),
        "flops_m": round(flops_m, 2),
        "macs_m": round(macs_m, 2),
        "train_loss": round(float(train_loss_best), 4) if train_loss_best is not None else None,
        "train_accuracy": round(float(train_acc_best), 2) if train_acc_best is not None else None,
        "val_loss": round(float(val_loss_best), 4) if val_loss_best is not None else None,
        "val_accuracy": round(float(val_acc_best), 2) if val_acc_best is not None else None,
        "val_macro_f1": round(float(val_f1_best), 4) if val_f1_best is not None else None,
        "test_loss": round(test_loss, 4),
        "test_top1_accuracy": round(test_top1_acc, 2),
        "test_top3_accuracy": round(test_top3_acc, 2),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "test_macro_precision": round(test_macro_prec, 4),
        "test_macro_recall": round(test_macro_rec, 4),
        "test_weighted_precision": round(test_weighted_prec, 4),
        "test_weighted_recall": round(test_weighted_rec, 4),
        "cpu_latency_ms": round(cpu_lat_ms, 2),
        "cpu_fps": round(cpu_fps, 1),
        "cuda_latency_ms": round(cuda_lat_ms, 2) if cuda_lat_ms is not None else None,
        "cuda_fps": round(cuda_fps, 1) if cuda_fps is not None else None,
        "classes": class_names,
        "per_class_metrics": per_class_dict,
    }

    for json_out in [
        os.path.join(model_res_dir, f"{name}_test_summary.json"),
        os.path.join(WEIGHTS_DIR, f"{name}_test_summary.json"),
    ]:
        with open(json_out, "w", encoding="utf-8-sig", newline="\r\n") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print(f"KET QUA DANH GIA MO HINH: {name.upper()} (Epoch {best_epoch})")
    print(f" - Parameters & Size:   {trainable_params:,} params ({ckpt_size_mb:.2f} MB)")
    print(f" - Complexity (FLOPs):  {flops_m:.2f} MFLOPs | {macs_m:.2f} MMACs")
    print(f" - Test Loss:           {test_loss:.4f}")
    print(f" - Test Top-1 Accuracy: {test_top1_acc:.2f}% | Top-3 Accuracy: {test_top3_acc:.2f}%")
    print(f" - Test Macro-F1:       {test_macro_f1:.4f} | Weighted-F1: {test_weighted_f1:.4f}")
    print(f" - CPU Latency:         {cpu_lat_ms:.2f} ms ({cpu_fps:.1f} FPS)")
    print("=" * 72)
    print(cls_report_str)

    # 7. Goi Grad-CAM & Grad-CAM++
    if run_gradcam:
        generate_gradcam_for_model(name, data_dir=data_dir)

    return summary_data


def _plot_comparison_charts(summaries, comp_dir):
    """Ve 4 bieu do doi sanh (300 DPI) vao results/4_comparison_all_models/."""
    models = [s["model_name"].upper() for s in summaries]
    colors = ["#2b5c8f", "#d95f02", "#2ca02c"][: len(summaries)]

    # 1. comparison_metrics_bar_chart.png (Top-1 Acc, Top-3 Acc, Macro-F1 * 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summaries))
    width = 0.25
    top1_vals = [s["test_top1_accuracy"] for s in summaries]
    top3_vals = [s["test_top3_accuracy"] for s in summaries]
    f1_pct_vals = [s["test_macro_f1"] * 100.0 for s in summaries]

    b1 = ax.bar(x - width, top1_vals, width, label="Test Top-1 Accuracy (%)", color="#1f77b4")
    b2 = ax.bar(x, top3_vals, width, label="Test Top-3 Accuracy (%)", color="#ff7f0e")
    b3 = ax.bar(x + width, f1_pct_vals, width, label="Test Macro-F1 (x100 %)", color="#2ca02c")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylabel("Diem so (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_title("So sanh Hieu nang Phan loai giua 3 Mo hinh tren Tap Test Thuc te", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "comparison_metrics_bar_chart.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. comparison_per_class_f1_chart.png
    class_names = summaries[0]["classes"]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    x_cls = np.arange(len(class_names))
    w_cls = 0.8 / max(1, len(summaries))

    for i, s in enumerate(summaries):
        offset = (i - (len(summaries) - 1) / 2.0) * w_cls
        cls_f1s = [s["per_class_metrics"][c]["F1_Score"] * 100.0 for c in class_names]
        bars = ax.bar(x_cls + offset, cls_f1s, w_cls, label=s["model_name"].upper(), color=colors[i])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x_cls)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel("F1-Score (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_title("Doi sanh F1-Score tung Lop Benh giua 3 Mo hinh", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "comparison_per_class_f1_chart.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. comparison_efficiency_chart.png (4 khung: Tham so K, FLOPs M, CPU Latency ms, CPU FPS)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    params_k = [s["trainable_parameters"] / 1000.0 for s in summaries]
    flops_vals = [s.get("flops_m", 0.0) or 0.0 for s in summaries]
    lat_ms = [s["cpu_latency_ms"] for s in summaries]
    fps_vals = [s["cpu_fps"] for s in summaries]

    eff_specs = [
        (axes[0], params_k, "So luong Tham so (K Params)", "K Params", "{:.1f}K"),
        (axes[1], flops_vals, "Do phuc tap Tinh toan (FLOPs)", "MFLOPs", "{:.1f}M"),
        (axes[2], lat_ms, "Do tre Suy luan CPU (Batch=1)", "Latency (ms)", "{:.2f} ms"),
        (axes[3], fps_vals, "Toc do Xu ly Thoi gian thuc (FPS)", "Frames / sec (FPS)", "{:.1f} FPS"),
    ]
    for ax, vals, title, ylabel, fmt in eff_specs:
        bars = ax.bar(models, vals, color=colors, width=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                fmt.format(h),
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.25 if vals and max(vals) > 0 else 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[3].axhline(30.0, color="#dc2626", linestyle="--", linewidth=1.6, label="Real-time Threshold (30 FPS)")
    axes[3].legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Doi sanh Hieu nang Thoi gian thuc (Real-time Performance) & Do phuc tap Phan cung", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "comparison_efficiency_chart.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. comparison_learning_curves.png (Chong duong cong Val Loss, Val Acc, Val Macro-F1 cua ca 3 mo hinh)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    has_any_history = False
    for i, s in enumerate(summaries):
        m_name = s["model_name"]
        folder = MODEL_FOLDER_MAP[m_name]
        hist_candidates = [
            os.path.join(RESULTS_DIR, folder, f"{m_name}_history.json"),
            os.path.join(WEIGHTS_DIR, f"{m_name}_history.json"),
        ]
        hist_path = next((p for p in hist_candidates if os.path.exists(p)), None)
        if not hist_path:
            continue
        with open(hist_path, "r", encoding="utf-8-sig") as f:
            hist = json.load(f)
        epochs = range(1, len(hist.get("val_loss", [])) + 1)
        if not epochs:
            continue
        has_any_history = True
        lbl = m_name.upper()
        axes[0].plot(epochs, hist["val_loss"], label=lbl, color=colors[i], linewidth=2)
        axes[1].plot(epochs, hist["val_acc"], label=lbl, color=colors[i], linewidth=2)
        axes[2].plot(epochs, hist["val_f1"], label=lbl, color=colors[i], linewidth=2)

    if has_any_history:
        axes[0].set_title("Validation Loss", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Loss")
        axes[1].set_title("Validation Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Accuracy (%)")
        axes[2].set_title("Validation Macro-F1 Score", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("Macro-F1")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
        plt.suptitle("So sanh Duong cong Hoi tu (Learning Curves) giua 3 Mo hinh", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(comp_dir, "comparison_learning_curves.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _export_comparison_reports(summaries, best_model_info, per_class_rows, comp_dir):
    """Xuat BAO_CAO_SO_SANH_3_MO_HINH.md va BAO_CAO_SO_SANH_3_MO_HINH.html (utf-8-sig, newline='\r\n')."""
    winner_name = best_model_info["best_model_name"].upper()
    md_lines = [
        "# BÁO CÁO ĐỐI SÁNH TOÀN DIỆN 3 MÔ HÌNH HỌC SÂU (13,165 ẢNH — TEST THỰC TẾ PHENIKAA)",
        "",
        f"## 1. Mô hình Chiến thắng (Best Model): **{winner_name}**",
        f"- **Test Macro-F1:** `{best_model_info['test_macro_f1']:.4f}`",
        f"- **Test Top-1 Accuracy:** `{best_model_info['test_top1_accuracy']:.2f}%`",
        f"- **Test Top-3 Accuracy:** `{best_model_info['test_top3_accuracy']:.2f}%`",
        f"- **Số lượng tham số & Dung lượng:** `{best_model_info['trainable_parameters']:,}` tham số (`{best_model_info['checkpoint_size_mb']:.2f} MB`)",
        f"- **Độ phức tạp tính toán (FLOPs / MACs):** `{best_model_info.get('flops_m', 0.0):.2f} MFLOPs` (`{best_model_info.get('macs_m', 0.0):.2f} MMACs`)",
        f"- **Độ trễ suy luận CPU (Batch=1):** `{best_model_info['cpu_latency_ms']:.2f} ms` (`{best_model_info['cpu_fps']:.1f} FPS`)",
        f"- **Tiêu chí xếp hạng:** `{best_model_info['selection_criteria']}`",
        "",
        "## 2. Bảng Tổng hợp Hiệu năng 3 Mô hình trên Tập Test Thực tế",
        "",
        "| Hạng | Mô hình | Tham số | Dung lượng (MB) | FLOPs (M) | Test Loss | Top-1 Acc (%) | Top-3 Acc (%) | Macro-F1 | Weighted-F1 | CPU Latency (ms) | CPU FPS |",
        "| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for rank, s in enumerate(summaries, start=1):
        md_lines.append(
            f"| {rank} | **{s['model_name'].upper()}** | {s['trainable_parameters']:,} | {s['checkpoint_size_mb']:.2f} | "
            f"{s.get('flops_m', 0.0):.2f} | {s['test_loss']:.4f} | {s['test_top1_accuracy']:.2f}% | {s['test_top3_accuracy']:.2f}% | "
            f"**{s['test_macro_f1']:.4f}** | {s['test_weighted_f1']:.4f} | {s['cpu_latency_ms']:.2f} | {s['cpu_fps']:.1f} |"
        )

    md_lines.extend([
        "",
        "## 3. Bảng Đối sánh F1-Score và Accuracy Từng Lớp Bệnh",
        "",
        "| Lớp bệnh (Class) | DSCNet F1 | ResNet18 F1 | SE-ResNet18 F1 | DSCNet Acc (%) | ResNet18 Acc (%) | SE-ResNet18 Acc (%) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for r in per_class_rows:
        md_lines.append(
            f"| **{r['Class']}** | {r.get('dscnet_f1', 0.0):.4f} | {r.get('resnet18_f1', 0.0):.4f} | {r.get('se_resnet18_f1', 0.0):.4f} | "
            f"{r.get('dscnet_acc', 0.0):.2f}% | {r.get('resnet18_acc', 0.0):.2f}% | {r.get('se_resnet18_acc', 0.0):.2f}% |"
        )

    md_lines.extend([
        "",
        "## 4. Danh mục Biểu đồ & Trực quan hóa (300 DPI)",
        "- `comparison_metrics_bar_chart.png`: Biểu đồ so sánh Top-1 Accuracy, Top-3 Accuracy và Macro-F1.",
        "- `comparison_per_class_f1_chart.png`: Biểu đồ so sánh F1-Score trên từng lớp bệnh.",
        "- `comparison_efficiency_chart.png`: Biểu đồ so sánh Tham số (K), Độ phức tạp tính toán (MFLOPs), Độ trễ CPU (ms) và Tốc độ xử lý thời gian thực (FPS).",
        "- `comparison_learning_curves.png`: Biểu đồ chồng đường cong học tập Validation của cả 3 mô hình.",
        "- `comparison_gradcam_3models.png`: Ma trận 7x7 đối sánh trực quan vùng chú ý Grad-CAM và Grad-CAM++.",
        "- Trong từng thư mục mô hình (`1_dscnet`, `2_resnet18`, `3_se_resnet18`):",
        "  - `{name}_loss_curve.png` & `{name}_accuracy_curve.png`: Đồ thị Loss và Accuracy độc lập qua các Epoch.",
        "  - `{name}_confusion_matrix.png`: Ma trận nhầm lẫn kép (Số lượng & Tỷ lệ %).",
        "  - `{name}_qualitative_correct_samples.png`: Hình ảnh kết quả suy luận thực tế dự đoán đúng kèm Confidence Score.",
        "  - `{name}_qualitative_error_analysis.png`: Hình ảnh phân tích các mẫu dự đoán sai (Misclassified) và ca khó (Hard Cases).",
        "",
    ])

    md_path = os.path.join(comp_dir, "BAO_CAO_SO_SANH_3_MO_HINH.md")
    with open(md_path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write("\n".join(md_lines))

    # Xuat HTML
    summary_tr = []
    for rank, s in enumerate(summaries, start=1):
        summary_tr.append(
            f"<tr><td>{rank}</td><td><b>{s['model_name'].upper()}</b></td>"
            f"<td>{s['trainable_parameters']:,}</td><td>{s['checkpoint_size_mb']:.2f}</td>"
            f"<td>{s.get('flops_m', 0.0):.2f}</td>"
            f"<td>{s['test_loss']:.4f}</td><td>{s['test_top1_accuracy']:.2f}%</td>"
            f"<td>{s['test_top3_accuracy']:.2f}%</td><td><b>{s['test_macro_f1']:.4f}</b></td>"
            f"<td>{s['test_weighted_f1']:.4f}</td><td>{s['cpu_latency_ms']:.2f}</td><td>{s['cpu_fps']:.1f}</td></tr>"
        )

    per_cls_tr = []
    for r in per_class_rows:
        per_cls_tr.append(
            f"<tr><td><b>{r['Class']}</b></td>"
            f"<td>{r.get('dscnet_f1', 0.0):.4f}</td><td>{r.get('resnet18_f1', 0.0):.4f}</td><td>{r.get('se_resnet18_f1', 0.0):.4f}</td>"
            f"<td>{r.get('dscnet_acc', 0.0):.2f}%</td><td>{r.get('resnet18_acc', 0.0):.2f}%</td><td>{r.get('se_resnet18_acc', 0.0):.2f}%</td></tr>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo Cáo Đối Sánh 3 Mô Hình Học Sâu - PCOPD</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 32px; color: #1e293b; line-height: 1.6; }}
h1, h2 {{ color: #0f172a; }}
.winner-box {{ background: #ecfdf5; border-left: 6px solid #10b981; padding: 16px 20px; border-radius: 6px; margin-bottom: 24px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px 0; }}
th, td {{ border: 1px solid #cbd5e1; padding: 10px 12px; text-align: center; }}
th {{ background: #f1f5f9; font-weight: 700; }}
td:nth-child(2), per-cls td:first-child {{ text-align: left; }}
.charts img {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0; border-radius: 8px; margin: 12px 0 24px 0; }}
</style>
</head>
<body>
<h1>BÁO CÁO ĐỐI SÁNH TOÀN DIỆN 3 MÔ HÌNH HỌC SÂU (13,165 ẢNH — TEST THỰC TẾ PHENIKAA)</h1>
<div class="winner-box">
  <h2>🏆 Mô hình Chiến thắng (Best Model): {winner_name}</h2>
  <p><b>Test Macro-F1:</b> {best_model_info['test_macro_f1']:.4f} |
     <b>Test Top-1 Accuracy:</b> {best_model_info['test_top1_accuracy']:.2f}% |
     <b>Test Top-3 Accuracy:</b> {best_model_info['test_top3_accuracy']:.2f}% |
     <b>Tham số:</b> {best_model_info['trainable_parameters']:,} ({best_model_info['checkpoint_size_mb']:.2f} MB) |
     <b>FLOPs:</b> {best_model_info.get('flops_m', 0.0):.2f} MFLOPs |
     <b>CPU Latency:</b> {best_model_info['cpu_latency_ms']:.2f} ms ({best_model_info['cpu_fps']:.1f} FPS)</p>
</div>

<h2>1. Bảng Tổng hợp Hiệu năng 3 Mô hình trên Tập Test Thực tế</h2>
<table>
  <thead>
    <tr>
      <th>Hạng</th><th>Mô hình</th><th>Tham số</th><th>Dung lượng (MB)</th><th>FLOPs (M)</th><th>Test Loss</th>
      <th>Top-1 Acc (%)</th><th>Top-3 Acc (%)</th><th>Macro-F1</th><th>Weighted-F1</th>
      <th>CPU Latency (ms)</th><th>CPU FPS</th>
    </tr>
  </thead>
  <tbody>
    {"".join(summary_tr)}
  </tbody>
</table>

<h2>2. Bảng Đối sánh F1-Score và Accuracy Từng Lớp Bệnh</h2>
<table>
  <thead>
    <tr>
      <th>Lớp bệnh (Class)</th><th>DSCNet F1</th><th>ResNet18 F1</th><th>SE-ResNet18 F1</th>
      <th>DSCNet Acc (%)</th><th>ResNet18 Acc (%)</th><th>SE-ResNet18 Acc (%)</th>
    </tr>
  </thead>
  <tbody>
    {"".join(per_cls_tr)}
  </tbody>
</table>

<div class="charts">
  <h2>3. Biểu đồ Trực quan & Giải thích Mô hình (Grad-CAM / Grad-CAM++)</h2>
  <h3>3.1. So sánh Top-1 Accuracy, Top-3 Accuracy và Macro-F1</h3>
  <img src="comparison_metrics_bar_chart.png" alt="Metrics Comparison">
  <h3>3.2. So sánh F1-Score Từng Lớp Bệnh</h3>
  <img src="comparison_per_class_f1_chart.png" alt="Per-Class F1 Comparison">
  <h3>3.3. Hiệu quả Tính toán (Tham số, FLOPs, Độ trễ CPU, FPS)</h3>
  <img src="comparison_efficiency_chart.png" alt="Efficiency Comparison">
  <h3>3.4. Đường cong Hội tụ (Learning Curves)</h3>
  <img src="comparison_learning_curves.png" alt="Learning Curves Comparison">
  <h3>3.5. Đối sánh Grad-CAM & Grad-CAM++ trên 7 Lớp Bệnh</h3>
  <img src="comparison_gradcam_3models.png" alt="Grad-CAM 3 Models Comparison">
</div>
</body>
</html>
"""
    html_path = os.path.join(comp_dir, "BAO_CAO_SO_SANH_3_MO_HINH.html")
    with open(html_path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write(html_content)


def compare_all_models(data_dir=None):
    """
    Doc ket qua cua ca 3 mo hinh (dscnet, resnet18, se_resnet18), xep hang chon Best Model
    theo thu tu uu tien: (test_macro_f1, test_top1_accuracy, -trainable_parameters).
    Xuat bang tong hop, 4 bieu do doi sanh, Grad-CAM 3 mo hinh va bao cao MD/HTML.
    """
    comp_dir = os.path.join(RESULTS_DIR, "4_comparison_all_models")
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    summaries = []
    for m_name in ALL_MODELS:
        folder = MODEL_FOLDER_MAP[m_name]
        summary_path = os.path.join(RESULTS_DIR, folder, f"{m_name}_test_summary.json")
        if not os.path.exists(summary_path):
            w_summary_path = os.path.join(WEIGHTS_DIR, f"{m_name}_test_summary.json")
            if os.path.exists(w_summary_path):
                summary_path = w_summary_path
            elif os.path.exists(os.path.join(WEIGHTS_DIR, f"{m_name}_best.pth")):
                evaluate_checkpoint(m_name, run_gradcam=True, data_dir=data_dir)

        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8-sig") as f:
                s_obj = json.load(f)
            if "model_name" not in s_obj and "model_key" in s_obj:
                s_obj["model_name"] = s_obj["model_key"]
            if isinstance(s_obj.get("per_class_metrics"), list):
                s_obj["per_class_metrics"] = {
                    item["Class"]: item for item in s_obj["per_class_metrics"] if "Class" in item
                }
            if "flops_m" not in s_obj or s_obj.get("flops_m") is None:
                m_tmp = _create_model(s_obj["model_name"], num_classes=len(s_obj.get("classes", range(7))))
                fl_m, mc_m = compute_flops_and_macs(m_tmp)
                s_obj["flops_m"] = round(fl_m, 2)
                s_obj["macs_m"] = round(mc_m, 2)
            summaries.append(s_obj)

    if not summaries:
        print("[!] Chua co ket qua danh gia cua mo hinh nao de so sanh.")
        return None

    # Xep hang theo thu tu uu tien: (test_macro_f1, test_top1_accuracy, -trainable_parameters)
    summaries_sorted = sorted(
        summaries,
        key=lambda x: (
            float(x["test_macro_f1"]),
            float(x["test_top1_accuracy"]),
            -int(x["trainable_parameters"]),
        ),
        reverse=True,
    )

    winner = summaries_sorted[0]
    best_model_info = {
        "best_model": winner["model_name"],
        "best_model_name": winner["model_name"],
        "best_checkpoint_file": f"{winner['model_name']}_best.pth",
        "checkpoint_file": f"{winner['model_name']}_best.pth",
        "best_epoch": winner["best_epoch"],
        "trainable_parameters": winner["trainable_parameters"],
        "checkpoint_size_mb": winner["checkpoint_size_mb"],
        "flops_m": winner.get("flops_m"),
        "macs_m": winner.get("macs_m"),
        "test_loss": winner["test_loss"],
        "test_top1_accuracy": winner["test_top1_accuracy"],
        "test_top3_accuracy": winner["test_top3_accuracy"],
        "test_macro_f1": winner["test_macro_f1"],
        "test_weighted_f1": winner["test_weighted_f1"],
        "test_macro_precision": winner["test_macro_precision"],
        "test_macro_recall": winner["test_macro_recall"],
        "cpu_latency_ms": winner["cpu_latency_ms"],
        "cpu_fps": winner["cpu_fps"],
        "classes": winner["classes"],
        "selection_criteria": "Priority: (test_macro_f1, test_top1_accuracy, -trainable_parameters)",
        "ranking": [
            {
                "rank": idx + 1,
                "model_name": s["model_name"],
                "test_macro_f1": s["test_macro_f1"],
                "test_top1_accuracy": s["test_top1_accuracy"],
                "trainable_parameters": s["trainable_parameters"],
                "flops_m": s.get("flops_m"),
            }
            for idx, s in enumerate(summaries_sorted)
        ],
    }

    best_info_path = os.path.join(WEIGHTS_DIR, "best_model_info.json")
    with open(best_info_path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        json.dump(best_model_info, f, indent=2, ensure_ascii=False)

    # Luu bang tong hop 3 mo hinh (model_comparison_summary.csv va .json)
    summary_csv_path = os.path.join(comp_dir, "model_comparison_summary.csv")
    sum_fields = [
        "Rank",
        "Model",
        "Best_Epoch",
        "Trainable_Parameters",
        "Checkpoint_Size_MB",
        "FLOPs_M",
        "MACs_M",
        "Test_Loss",
        "Test_Top1_Accuracy (%)",
        "Test_Top3_Accuracy (%)",
        "Test_Macro_F1",
        "Test_Weighted_F1",
        "Test_Macro_Precision",
        "Test_Macro_Recall",
        "CPU_Latency_ms",
        "CPU_FPS",
        "CUDA_Latency_ms",
        "CUDA_FPS",
    ]
    with open(summary_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sum_fields)
        writer.writeheader()
        for idx, s in enumerate(summaries_sorted, start=1):
            writer.writerow({
                "Rank": idx,
                "Model": s["model_name"],
                "Best_Epoch": s["best_epoch"],
                "Trainable_Parameters": s["trainable_parameters"],
                "Checkpoint_Size_MB": s["checkpoint_size_mb"],
                "FLOPs_M": s.get("flops_m"),
                "MACs_M": s.get("macs_m"),
                "Test_Loss": s["test_loss"],
                "Test_Top1_Accuracy (%)": s["test_top1_accuracy"],
                "Test_Top3_Accuracy (%)": s["test_top3_accuracy"],
                "Test_Macro_F1": s["test_macro_f1"],
                "Test_Weighted_F1": s["test_weighted_f1"],
                "Test_Macro_Precision": s["test_macro_precision"],
                "Test_Macro_Recall": s["test_macro_recall"],
                "CPU_Latency_ms": s["cpu_latency_ms"],
                "CPU_FPS": s["cpu_fps"],
                "CUDA_Latency_ms": s.get("cuda_latency_ms"),
                "CUDA_FPS": s.get("cuda_fps"),
            })

    summary_json_path = os.path.join(comp_dir, "model_comparison_summary.json")
    with open(summary_json_path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        json.dump(
            {
                "best_model_info": best_model_info,
                "models": summaries_sorted,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Luu bang doi sanh F1 & Accuracy tung lop benh giua 3 mo hinh
    by_name = {s["model_name"]: s for s in summaries}
    class_names = winner["classes"]
    per_class_comp_rows = []
    for cls_name in class_names:
        row = {"Class": cls_name}
        for m in ALL_MODELS:
            if m in by_name and cls_name in by_name[m]["per_class_metrics"]:
                m_cls = by_name[m]["per_class_metrics"][cls_name]
                row[f"{m}_f1"] = m_cls["F1_Score"]
                row[f"{m}_acc"] = m_cls["Class_Accuracy (%)"]
            else:
                row[f"{m}_f1"] = 0.0
                row[f"{m}_acc"] = 0.0
        per_class_comp_rows.append(row)

    macro_comp_row = {"Class": "MACRO_AVERAGE"}
    for m in ALL_MODELS:
        if m in by_name:
            macro_comp_row[f"{m}_f1"] = by_name[m]["test_macro_f1"]
            macro_comp_row[f"{m}_acc"] = round(
                float(np.mean([r[f"{m}_acc"] for r in per_class_comp_rows])), 2
            )
        else:
            macro_comp_row[f"{m}_f1"] = 0.0
            macro_comp_row[f"{m}_acc"] = 0.0

    per_class_f1_csv = os.path.join(comp_dir, "model_comparison_per_class_f1.csv")
    pc_fields = [
        "Class",
        "dscnet_f1",
        "resnet18_f1",
        "se_resnet18_f1",
        "dscnet_acc",
        "resnet18_acc",
        "se_resnet18_acc",
    ]
    with open(per_class_f1_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pc_fields)
        writer.writeheader()
        for r in per_class_comp_rows:
            writer.writerow(r)
        writer.writerow(macro_comp_row)

    # Ve 4 bieu do doi sanh (theo thu tu ALL_MODELS neu du, hoac summaries)
    ordered_for_plots = [by_name[m] for m in ALL_MODELS if m in by_name]
    _plot_comparison_charts(ordered_for_plots, comp_dir)

    # Goi generate_3models_gradcam_comparison()
    generate_3models_gradcam_comparison(data_dir=data_dir)

    # Xuat bao cao BAO_CAO_SO_SANH_3_MO_HINH.md va .html
    _export_comparison_reports(summaries_sorted, best_model_info, per_class_comp_rows + [macro_comp_row], comp_dir)

    print("\n" + "=" * 72)
    print(f"[BEST MODEL SELECTED]: {winner['model_name'].upper()}")
    print(
        f" -> Test Macro-F1: {winner['test_macro_f1']:.4f} | "
        f"Top-1 Acc: {winner['test_top1_accuracy']:.2f}% | "
        f"Params: {winner['trainable_parameters']:,} | "
        f"FLOPs: {winner.get('flops_m', 0.0):.2f} MFLOPs"
    )
    print(f" -> Da luu thong tin Best Model tai: {best_info_path}")
    print(f" -> Da luu bo bao cao doi sanh tai:  {comp_dir}")
    print("=" * 72)
    return best_model_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danh gia va so sanh 3 mo hinh hoc sau tren tap Test")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["dscnet", "resnet18", "se_resnet18", "all"],
        help="Ten mo hinh can danh gia (mac dinh: all)",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Duong dan thu muc data (mac dinh tu dong nhan dien d:/AIRB/.../data)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size khi danh gia")
    parser.add_argument("--no_gradcam", action="store_true", help="Bo qua buoc sinh anh Grad-CAM")
    args = parser.parse_args()

    if args.model == "all":
        for m in ALL_MODELS:
            evaluate_checkpoint(m, batch_size=args.batch_size, run_gradcam=not args.no_gradcam, data_dir=args.data_dir)
        compare_all_models(data_dir=args.data_dir)
    else:
        evaluate_checkpoint(args.model, batch_size=args.batch_size, run_gradcam=not args.no_gradcam, data_dir=args.data_dir)
