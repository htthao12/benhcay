"""
PyTorch Dataset & DataLoader pipeline cho bo du lieu PCOPD
Tich hop Data Augmentation manh me va chuan hoa anh 224x224
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# Kich thuoc anh chuan hoc sau (nang cap tu 160 len 224 de bat chi tiet vet benh nho)
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Chuan hoa Z-Score theo ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_data_transforms():
    """Dinh nghia cac phep bien doi va tang cuong du lieu (Augmentation)"""
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Random Affine: Xoay goc ±30 do, scale phong to thu nho 0.8 - 1.2
        transforms.RandomAffine(
            degrees=30,
            scale=(0.8, 1.2),
            shear=10,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        # Photometric Jitter: Do sang, tuong phan, do bao hoa
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        # Z-score Normalization
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Random Erasing mo phong vet rach, la che khuat
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transforms, val_test_transforms

def get_dataloaders(data_dir=None, batch_size=BATCH_SIZE, num_workers=0):
    """Khoi tao DataLoaders cho tap Train, Validation va Test"""
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))

    train_tf, val_tf = get_data_transforms()

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Chua tim thay thu muc '{train_path}'. Hay chay 'build_pcopd_dataset.py' truoc!")

    train_dataset = datasets.ImageFolder(train_path, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_path, transform=val_tf)
    test_dataset = datasets.ImageFolder(test_path, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "num_classes": len(train_dataset.classes)
    }

if __name__ == "__main__":
    print("[*] Kiem tra pipeline Dataset & DataLoader...")
    train_tf, val_tf = get_data_transforms()
    print(" -> Data transformations khoi tao thanh cong!")

