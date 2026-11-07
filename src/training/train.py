"""
Engine huan luyen mo hinh hoc sau tu dau (Train From Scratch)
Ho tro: DSCNet, ResNet18, SE-ResNet18 (hoac --model all de huan luyen tuan tu ca 3 mo hinh)
Su dung: AdamW, Cosine Annealing LR Schedule, Label Smoothing 0.1, Mixed Precision (AMP), Early Stopping
Tu dong luu lich su (.json & .csv utf-8-sig), ve bieu do Learning Curves (300 DPI),
danh gia chi tiet tren tap Test + Grad-CAM / Grad-CAM++ va doi sanh 3 mo hinh.
"""

import os
import sys
import csv
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.dataset import get_dataloaders
from src.models.resnet18 import ResNet18
from src.models.se_resnet18 import SEResNet18
from src.models.dscnet import DSCNet
from src.evaluation.gradcam import MODEL_FOLDER_MAP
from src.evaluation.evaluate import (
    evaluate_checkpoint,
    compare_all_models,
    plot_standalone_loss_and_acc_curves,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_MODELS = ["dscnet", "resnet18", "se_resnet18"]


def train_one_epoch(model, dataloader, criterion, optimizer, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = scaler is not None and DEVICE.type == "cuda"

    for images, targets in dataloader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / max(1, total), (correct / max(1, total)) * 100.0


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    top3_correct = 0.0
    all_preds = []
    all_targets = []

    use_amp = DEVICE.type == "cuda"

    for images, targets in dataloader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        maxk = min(3, outputs.size(1))
        _, topk_preds = outputs.topk(maxk, 1, True, True)
        top3_correct += (
            topk_preds.t()
            .eq(targets.view(1, -1).expand_as(topk_preds.t()))[:maxk]
            .reshape(-1)
            .float()
            .sum()
            .item()
        )

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / max(1, total)
    acc = (correct / max(1, total)) * 100.0
    top3_acc = (top3_correct / max(1, total)) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, top3_acc, macro_f1, all_preds, all_targets


def plot_learning_curves(history, save_paths, model_name, best_epoch=None):
    """
    Ve bieu do 3 khung (Loss, Top-1/Top-3 Accuracy, Validation Macro-F1 co danh dau vach doc tai best_epoch)
    luu vao ca weights/ va results/<folder>/ o do phan giai 300 DPI.
    """
    if isinstance(save_paths, str):
        save_paths = [save_paths]

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # 1. Loss curve
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="crimson", linewidth=2)
    if best_epoch:
        axes[0].axvline(best_epoch, color="darkorange", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_epoch})")
    axes[0].set_title(f"{model_name.upper()} — Loss Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Top-1 / Top-3 Accuracy curve
    axes[1].plot(epochs, history["train_acc"], label="Train Top-1 Acc", color="royalblue", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Top-1 Acc", color="crimson", linewidth=2)
    if "val_top3_acc" in history and history["val_top3_acc"]:
        axes[1].plot(epochs, history["val_top3_acc"], label="Val Top-3 Acc", color="teal", linestyle="-.", linewidth=1.8)
    if best_epoch:
        axes[1].axvline(best_epoch, color="darkorange", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_epoch})")
    axes[1].set_title(f"{model_name.upper()} — Top-1 & Top-3 Accuracy (%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Validation Macro-F1 curve
    axes[2].plot(epochs, history["val_f1"], label="Val Macro-F1", color="forestgreen", linewidth=2)
    if best_epoch:
        axes[2].axvline(best_epoch, color="darkorange", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_epoch})")
    axes[2].set_title(f"{model_name.upper()} — Validation Macro-F1", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1 Score")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    for sp in save_paths:
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        plt.savefig(sp, dpi=300, bbox_inches="tight")
        print(f"[+] Da luu bieu do Learning Curves tai: {sp}")
    plt.close(fig)


def _save_history_files(history, name, model_res_dir):
    """Luu lich su huan luyen ra ca .json va .csv (utf-8-sig, newline='')."""
    for json_path in [
        os.path.join(WEIGHTS_DIR, f"{name}_history.json"),
        os.path.join(model_res_dir, f"{name}_history.json"),
    ]:
        with open(json_path, "w", encoding="utf-8-sig", newline="\r\n") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(model_res_dir, f"{name}_history.csv")
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "val_top3_acc",
        "val_f1",
        "lr",
        "epoch_time",
    ]
    n_epochs = len(history["train_loss"])
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(n_epochs):
            writer.writerow({
                "epoch": idx + 1,
                "train_loss": round(float(history["train_loss"][idx]), 6),
                "val_loss": round(float(history["val_loss"][idx]), 6),
                "train_acc": round(float(history["train_acc"][idx]), 4),
                "val_acc": round(float(history["val_acc"][idx]), 4),
                "val_top3_acc": round(float(history["val_top3_acc"][idx]), 4),
                "val_f1": round(float(history["val_f1"][idx]), 6),
                "lr": round(float(history["lr"][idx]), 8),
                "epoch_time": round(float(history["epoch_time"][idx]), 3),
            })


def train_model(model_name="dscnet", epochs=100, batch_size=32, lr=2e-3, patience=20, data_dir=None):
    name = model_name.lower()
    if name not in MODEL_FOLDER_MAP:
        raise ValueError(f"Khong ho tro model '{model_name}'")

    folder = MODEL_FOLDER_MAP[name]
    model_res_dir = os.path.join(RESULTS_DIR, folder)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(model_res_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"BAT DAU HUAN LUYEN: {name.upper()} | EPOCHS: {epochs} | DEVICE: {DEVICE}")
    print("=" * 75)

    # 1. Load Data
    data_info = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    train_loader = data_info["train_loader"]
    val_loader = data_info["val_loader"]
    test_loader = data_info["test_loader"]
    num_classes = data_info["num_classes"]
    class_names = data_info["classes"]
    print(f"[*] So luong lop: {num_classes} ({class_names})")
    print(
        f"[*] So anh: Train = {len(train_loader.dataset):,} | "
        f"Val = {len(val_loader.dataset):,} | Test = {len(test_loader.dataset):,}"
    )

    # 2. Khoi tao Model
    if name == "dscnet":
        model = DSCNet(num_classes=num_classes)
    elif name == "se_resnet18":
        model = SEResNet18(num_classes=num_classes)
    elif name == "resnet18":
        model = ResNet18(num_classes=num_classes)

    model = model.to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Tong tham so mo hinh ({name.upper()}): {param_count:,}")

    # 3. Loss & Optimizer (AdamW + Label Smoothing 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_top3_acc": [],
        "val_f1": [],
        "lr": [],
        "epoch_time": [],
    }

    save_path = os.path.join(WEIGHTS_DIR, f"{name}_best.pth")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_top3, val_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - start_time

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_top3_acc"].append(float(val_top3))
        history["val_f1"].append(float(val_f1))
        history["lr"].append(current_lr)
        history["epoch_time"].append(float(elapsed))

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] ({elapsed:.1f}s) | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% Top3: {val_top3:.2f}% F1: {val_f1:.4f}"
        )

        # Check Early Stopping & Save Best Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_f1": float(val_f1),
                    "val_acc": float(val_acc),
                    "val_top3_acc": float(val_top3),
                    "trainable_parameters": param_count,
                    "classes": class_names,
                },
                save_path,
            )
            print(f"  ==> [SAVED] Luu checkpoint tot nhat tai Epoch {epoch} (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] Early Stopping tai Epoch {epoch} vi Val F1 khong cai thien sau {patience} epochs.")
                break

    print(
        f"\n[+] Huan luyen {name.upper()} hoan tat! "
        f"Checkpoint tot nhat o Epoch {best_epoch} (Val F1: {best_val_f1:.4f}) luu tai: {save_path}"
    )

    # Luu lich su (.json & .csv) va ve bieu do Learning Curves vao ca weights/ va results/<folder>/
    _save_history_files(history, name, model_res_dir)
    curve_paths = [
        os.path.join(WEIGHTS_DIR, f"{name}_learning_curves.png"),
        os.path.join(model_res_dir, f"{name}_learning_curves.png"),
    ]
    plot_learning_curves(history, curve_paths, name, best_epoch=best_epoch)
    plot_standalone_loss_and_acc_curves(history, name, [WEIGHTS_DIR, model_res_dir], best_epoch=best_epoch)

    # 4. Danh gia kiem thu doc lap tren tap TEST + Grad-CAM / Grad-CAM++ + Qualitative Samples
    evaluate_checkpoint(name, batch_size=batch_size, run_gradcam=True, data_dir=data_dir)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huan luyen va doi sanh 3 mo hinh nhan dien benh la cay")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "dscnet", "resnet18", "se_resnet18"],
        help="Ten mo hinh can huan luyen (mac dinh: all -> dscnet, resnet18, se_resnet18)",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Duong dan thu muc data (mac dinh tu dong nhan dien d:/AIRB/.../data)")
    parser.add_argument("--epochs", type=int, default=100, help="So epochs (mac dinh 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Kich thuoc batch")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    args = parser.parse_args()

    target_models = ALL_MODELS if args.model == "all" else [args.model]
    for m_name in target_models:
        train_model(
            model_name=m_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            data_dir=args.data_dir,
        )

    # Sau khi hoan tat vong lap train, tu dong goi compare_all_models()
    compare_all_models(data_dir=args.data_dir)
