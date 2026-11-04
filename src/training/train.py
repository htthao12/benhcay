"""
Engine huấn luyện mô hình học sâu từ đầu (Train From Scratch)
Hỗ trợ: DSCNet, ResNet18, SE-ResNet18
Sử dụng: AdamW, Cosine Annealing LR Schedule, Label Smoothing 0.1, Mixed Precision (AMP), Early Stopping
Đánh giá đầy đủ: Train/Val/Test Loss, Accuracy, Top-3 Accuracy, Macro-F1 và vẽ biểu đồ Learning Curves
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.dataset import get_dataloaders
from src.models.resnet18 import ResNet18
from src.models.se_resnet18 import SEResNet18
from src.models.dscnet import DSCNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "weights"))

def train_one_epoch(model, dataloader, criterion, optimizer, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = (scaler is not None and DEVICE.type == "cuda")

    for images, targets in dataloader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

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

    return total_loss / total, (correct / total) * 100.0

@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    top3_correct = 0.0
    all_preds = []
    all_targets = []

    use_amp = (DEVICE.type == "cuda")

    for images, targets in dataloader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        
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

        # Top-3 Accuracy
        maxk = min(3, outputs.size(1))
        _, topk_preds = outputs.topk(maxk, 1, True, True)
        top3_correct += topk_preds.t().eq(targets.view(1, -1).expand_as(topk_preds.t()))[:maxk].reshape(-1).float().sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / total
    acc = (correct / total) * 100.0
    top3_acc = (top3_correct / total) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, top3_acc, macro_f1, all_preds, all_targets

def plot_learning_curves(history, save_path, model_name):
    """Vẽ và lưu biểu đồ Loss, Accuracy, F1 qua các epoch"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="crimson")
    axes[0].set_title(f"{model_name.upper()} - Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy curve
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="royalblue")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", color="crimson")
    axes[1].set_title(f"{model_name.upper()} - Accuracy Curve (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    # Macro-F1 curve
    axes[2].plot(epochs, history["val_f1"], label="Val Macro-F1", color="forestgreen")
    axes[2].set_title(f"{model_name.upper()} - Validation Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Da luu bieu do Learning Curves tai: {save_path}")

def train_model(model_name="dscnet", epochs=100, batch_size=32, lr=2e-3, patience=20):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print("=" * 70)
    print(f"BAT DAU HUAN LUYEN: {model_name.upper()} | EPOCHS: {epochs} | DEVICE: {DEVICE}")
    print("=" * 70)

    # 1. Load Data
    data_info = get_dataloaders(batch_size=batch_size)
    train_loader = data_info["train_loader"]
    val_loader = data_info["val_loader"]
    test_loader = data_info["test_loader"]
    num_classes = data_info["num_classes"]
    class_names = data_info["classes"]
    print(f"[*] So luong lop: {num_classes} ({class_names})")
    print(f"[*] So anh: Train = {len(train_loader.dataset)} | Val = {len(val_loader.dataset)} | Test = {len(test_loader.dataset)}")

    # 2. Khởi tạo Model
    if model_name.lower() == "dscnet":
        model = DSCNet(num_classes=num_classes)
    elif model_name.lower() == "se_resnet18":
        model = SEResNet18(num_classes=num_classes)
    elif model_name.lower() == "resnet18":
        model = ResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Khong ho tro model '{model_name}'")

    model = model.to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Tong tham so mo hinh: {param_count:,}")

    # 3. Loss & Optimizer (AdamW + Label Smoothing 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    save_path = os.path.join(WEIGHTS_DIR, f"{model_name}_best.pth")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_top3, val_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch [{epoch:03d}/{epochs:03d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% Top3: {val_top3:.2f}% F1: {val_f1:.4f}")

        # Check Early Stopping & Save Best Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "classes": class_names
            }, save_path)
            print(f"  ==> [SAVED] Luu checkpoint tot nhat tai Epoch {epoch} (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] Early Stopping tai Epoch {epoch} vi Val F1 khong cai thien sau {patience} epochs.")
                break

    print(f"\n[+] Huan luyen hoan tat! Checkpoint tot nhat o Epoch {best_epoch} (Val F1: {best_val_f1:.4f}) luu tai: {save_path}")

    # Lưu lịch sử và vẽ biểu đồ
    with open(os.path.join(WEIGHTS_DIR, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_learning_curves(history, os.path.join(WEIGHTS_DIR, f"{model_name}_learning_curves.png"), model_name)

    # 4. Đánh giá kiểm thử độc lập trên tập TEST
    print("\n" + "=" * 70)
    print(f"DANH GIA CUOI CUNG TREN TAP TEST VOI CHECKPOINT TOT NHAT ({model_name.upper()} - Epoch {best_epoch}):")
    print("=" * 70)
    best_checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_acc, test_top3, test_f1, test_preds, test_targets = evaluate(model, test_loader, criterion)
    print(f"[*] Test Loss: {test_loss:.4f}")
    print(f"[*] Test Accuracy (Top-1): {test_acc:.2f}%")
    print(f"[*] Test Top-3 Accuracy:   {test_top3:.2f}%")
    print(f"[*] Test Macro-F1 Score:   {test_f1:.4f}")
    print("\nChi tiet tung lop tren tap Test:")
    print(classification_report(test_targets, test_preds, target_names=class_names, digits=4))

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huan luyen mo hinh nhan dien benh la cay")
    parser.add_argument("--model", type=str, default="dscnet", choices=["dscnet", "resnet18", "se_resnet18"], help="Ten mo hinh")
    parser.add_argument("--epochs", type=int, default=100, help="So epochs (mac dinh 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Kich thuoc batch")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    args = parser.parse_args()

    train_model(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)
