"""
PyTorch Dataset & DataLoader pipeline cho bo du lieu PCOPD (13,165 anh)
Tu dong nhan dien duong dan data/ (train, val, test) hoac data/processed/
Tich hop Data Augmentation manh me va chuan hoa anh 224x224
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Kich thuoc anh chuan hoc sau (224x224 de bat chi tiet vet benh nho tren la)
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Chuan hoa Z-Score theo ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_data_dir(data_dir=None) -> str:
    """
    Tu dong phat hien thu muc chua 3 tap train/, val/, test/:
    Uu tien:
    1. Duong dan data_dir truyen vao truc tiep (hoac data_dir/processed)
    2. d:/AIRB/Nam5/hocsaunangcao/midterm/data (cau truc moi)
    3. d:/AIRB/Nam5/hocsaunangcao/midterm/data/processed
    4. Duong dan tren Kaggle / Colab (/kaggle/input/...)
    """
    if data_dir:
        if os.path.exists(os.path.join(data_dir, "train")):
            return os.path.abspath(data_dir)
        if os.path.exists(os.path.join(data_dir, "processed", "train")):
            return os.path.abspath(os.path.join(data_dir, "processed"))
        if os.path.exists(os.path.join(data_dir, "data", "train")):
            return os.path.abspath(os.path.join(data_dir, "data"))

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "processed"),
        "/kaggle/input/data/data",
        "/kaggle/input/data",
        "/kaggle/input/phenikaa-plant-disease/data",
        "/kaggle/input/phenikaa-plant-disease",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train")):
            return os.path.abspath(c)

    return os.path.join(base_dir, "data")


def get_data_transforms():
    """Dinh nghia cac phep bien doi va tang cuong du lieu (Augmentation)"""
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Random Affine: Xoay goc ±30 do, scale phong to thu nho 0.8 - 1.2, nghieng shear 10 do
        transforms.RandomAffine(
            degrees=30,
            scale=(0.8, 1.2),
            shear=10,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        # Photometric Jitter: Do sang, tuong phan, do bao hoa mo phong anh sang tu nhien san truong
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        # Z-score Normalization
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Random Erasing mo phong vet rach, la che khuat mot phan
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transforms, val_test_transforms


def get_dataloaders(data_dir=None, batch_size=BATCH_SIZE, num_workers=2):
    """Khoi tao DataLoaders cho tap Train, Validation va Test"""
    resolved_dir = resolve_data_dir(data_dir)
    train_tf, val_tf = get_data_transforms()

    train_path = os.path.join(resolved_dir, "train")
    val_path = os.path.join(resolved_dir, "val")
    test_path = os.path.join(resolved_dir, "test")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Khong tim thay thu muc train tai '{train_path}'. "
            f"Hay kiem tra thu muc 'data/' hoac truyen --data_dir!"
        )

    train_dataset = datasets.ImageFolder(train_path, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_path, transform=val_tf)
    test_dataset = datasets.ImageFolder(test_path, transform=val_tf)

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )

    return {
        "data_dir": resolved_dir,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "num_classes": len(train_dataset.classes)
    }


if __name__ == "__main__":
    print("[*] Kiem tra pipeline Dataset & DataLoader...")
    info = get_dataloaders()
    print(f" -> Duong dan Data: {info['data_dir']}")
    print(f" -> Cac lop benh ({info['num_classes']}): {info['classes']}")
    print(
        f" -> Train: {len(info['train_dataset']):,} | "
        f"Val: {len(info['val_dataset']):,} | "
        f"Test: {len(info['test_dataset']):,}"
    )
