"""
Script gom, quy chuan hoa va chia tap Train/Val/Test cho bo du lieu lai
PCOPD (Phenikaa Campus Ornamental Plant Disease)
Bao gom 7 nhan: Healthy, Powdery_Mildew, Rust, Anthracnose, Aphids, Scorch_Sunburn, Yellowing_Wilt
"""

import os
import shutil
import random
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CUSTOM_DIR = os.path.join(BASE_DIR, "data", "phenikaa_custom")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Danh muc chuan hoa 7 trang thai benh & suc khoe
TARGET_CONDITIONS = [
    "Healthy",
    "Powdery_Mildew",
    "Rust",
    "Anthracnose",
    "Aphids",
    "Scorch_Sunburn",
    "Yellowing_Wilt"
]

def scan_and_collect_images():
    """Gom anh tu tat ca cac nguon va phan loai theo 7 nhan benh chuan hoa"""
    collected = defaultdict(list)
    
    # 1. Quet thu muc anh tu chup Phenikaa
    print("[*] 1. Dang quet thu muc anh tu chup Phenikaa (data/phenikaa_custom)...")
    if os.path.exists(CUSTOM_DIR):
        for species in os.listdir(CUSTOM_DIR):
            species_path = os.path.join(CUSTOM_DIR, species)
            if not os.path.isdir(species_path):
                continue
            for cond in os.listdir(species_path):
                cond_path = os.path.join(species_path, cond)
                if not os.path.isdir(cond_path):
                    continue
                
                matched_cond = None
                for tc in TARGET_CONDITIONS:
                    if tc.lower() in cond.lower():
                        matched_cond = tc
                        break
                
                if matched_cond:
                    for fname in os.listdir(cond_path):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                            collected[matched_cond].append({
                                "path": os.path.join(cond_path, fname),
                                "source": "phenikaa_custom",
                                "species": species
                            })

    # 2. Quet cac thu muc raw tai tu Kaggle
    print("[*] 2. Dang quet thu muc nguon Kaggle (data/raw)...")
    if os.path.exists(RAW_DIR):
        for root, dirs, files in os.walk(RAW_DIR):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, fname)
                    path_lower = full_path.lower()
                    
                    matched_cond = None
                    
                    # Phan loai benh
                    if "powdery" in path_lower or "mildew" in path_lower:
                        matched_cond = "Powdery_Mildew"
                    elif "rust" in path_lower or "red_spot" in path_lower:
                        matched_cond = "Rust"
                    elif "anthracnose" in path_lower or "blight" in path_lower:
                        matched_cond = "Anthracnose"
                    elif "aphid" in path_lower or "insect" in path_lower or "sawfly" in path_lower or "bite" in path_lower:
                        matched_cond = "Aphids"
                    elif "scorch" in path_lower or "sunburn" in path_lower:
                        matched_cond = "Scorch_Sunburn"
                    elif "yellow" in path_lower or "yld" in path_lower or "wilt" in path_lower or "death_leaf" in path_lower:
                        matched_cond = "Yellowing_Wilt"
                    elif "healthy" in path_lower or "fresh" in path_lower:
                        matched_cond = "Healthy"
                    
                    if matched_cond:
                        # Nhan dien loai cay tu duong dan
                        species = "Ornamental"
                        if "rose" in path_lower:
                            species = "Rose"
                        elif "hibiscus" in path_lower:
                            species = "Hibiscus"
                        elif "jasmine" in path_lower:
                            species = "Crape_Jasmine"
                        elif "bauhinia" in path_lower:
                            species = "Bauhinia"
                        elif "banana" in path_lower:
                            species = "Banana_Bush"
                        elif "mango" in path_lower:
                            species = "Mango"
                        elif "snake" in path_lower:
                            species = "Snake_Plant"
                        elif "money" in path_lower:
                            species = "Money_Plant"
                            
                        collected[matched_cond].append({
                            "path": full_path,
                            "source": "kaggle_raw",
                            "species": species
                        })

    return collected

def split_and_export(collected, max_per_class=1200, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    
    print("\n" + "=" * 70)
    print("THONG KE DU LIEU THU THAP DUOC THEO TUNG NHAN BENH:")
    print("=" * 70)
    
    for cond in TARGET_CONDITIONS:
        imgs = collected.get(cond, [])
        custom_cnt = sum(1 for x in imgs if x['source'] == 'phenikaa_custom')
        raw_cnt = sum(1 for x in imgs if x['source'] == 'kaggle_raw')
        print(f" - {cond:18s}: {len(imgs):5d} anh (Phenikaa: {custom_cnt:3d}, Kaggle: {raw_cnt:5d})")
        
    total_imgs = sum(len(v) for v in collected.values())
    print(f"\n[+] Tong cong anh phu hop: {total_imgs} anh tren 7 lop benh & suc khoe.")
    
    if total_imgs == 0:
        print("\n[!] Chua co anh nao trong 'data/raw' hoac 'data/phenikaa_custom'.")
        return

    # Tao cac thu muc train/val/test
    for split in ["train", "val", "test"]:
        for cond in TARGET_CONDITIONS:
            os.makedirs(os.path.join(PROCESSED_DIR, split, cond), exist_ok=True)

    print(f"\n[*] Dang can bang du lieu (gioi han toi da {max_per_class} anh/lop) va chia Stratified Split...")
    for cond, items in collected.items():
        random.shuffle(items)
        # Uu tien anh phenikaa_custom vao truoc
        phenikaa_items = [x for x in items if x['source'] == 'phenikaa_custom']
        kaggle_items = [x for x in items if x['source'] != 'phenikaa_custom']
        
        selected_items = phenikaa_items + kaggle_items
        if len(selected_items) > max_per_class:
            selected_items = selected_items[:max_per_class]
            
        random.shuffle(selected_items)
        n = len(selected_items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits = {
            "train": selected_items[:n_train],
            "val": selected_items[n_train:n_train + n_val],
            "test": selected_items[n_train + n_val:]
        }
        
        for split_name, split_items in splits.items():
            for idx, item in enumerate(split_items):
                ext = os.path.splitext(item['path'])[1]
                dest_fname = f"{item['species']}_{item['source']}_{idx:05d}{ext}"
                dest_path = os.path.join(PROCESSED_DIR, split_name, cond, dest_fname)
                try:
                    shutil.copyfile(item['path'], dest_path)
                except Exception:
                    pass

    print("[+] Hoan tat! Tap du lieu PCOPD da duoc xuat san sang vao 'data/processed/'.")

if __name__ == "__main__":
    collected = scan_and_collect_images()
    split_and_export(collected)
