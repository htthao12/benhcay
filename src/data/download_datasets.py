"""
Script tải và tổ chức các tập dữ liệu thành phần từ Kaggle cho dự án
REAL-TIME CAMPUS & ORNAMENTAL PLANT HEALTH DIAGNOSIS
"""

import os
import sys
import shutil

DATA_RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))

# Cac bo du lieu Kaggle mo (khong yeu cau dang nhap API van tai duoc)
DATASETS_CONFIG = [
    {
        "id": "shuvokumarbasak4004/flower-leaf-diseases-dataset-new-and-update",
        "name": "flower_leaf_dataset",
        "desc": "Hoa dam but, hoa ngoc but, hoa hong, hoa ban trang (6 loai hoa canh - 25,200 anh)"
    },
    {
        "id": "rashikrahmanpritom/plant-disease-recognition-dataset",
        "name": "powdery_rust_dataset",
        "desc": "Tap du lieu mo chua Benh Phan trang (Powdery) va Ri sat (Rust)"
    }
]

def check_or_install_kagglehub():
    try:
        import kagglehub
        return kagglehub
    except ImportError:
        print("[*] Dang cai dat thu vien 'kagglehub'...")
        os.system(f"{sys.executable} -m pip install kagglehub")
        try:
            import kagglehub
            return kagglehub
        except ImportError:
            return None

def download_all():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    print("=" * 70)
    print("DANG KIEM TRA VA TAI CAC BO DU LIEU NGUON CHO DE TAI CAY CANH KHUON VIEN")
    print("=" * 70)

    kagglehub = check_or_install_kagglehub()

    for item in DATASETS_CONFIG:
        print(f"\n[+] Kiem tra bo du lieu: {item['name']}")
        print(f"    Mo ta: {item['desc']}")
        print(f"    Kaggle ID: {item['id']}")

        dest_folder = os.path.join(DATA_RAW_DIR, item['name'])
        if os.path.exists(dest_folder) and len(os.listdir(dest_folder)) > 0:
            print(f"    => [DA CO] Du lieu da co san tai '{dest_folder}', bo qua tai lai.")
            continue

        if kagglehub:
            try:
                print("    Dang tai tu dong qua kagglehub...")
                download_path = kagglehub.dataset_download(item['id'])
                print(f"    Tai thanh cong ve cache: {download_path}")

                print(f"    Dang sao chep vao {dest_folder} ...")
                if os.path.exists(dest_folder):
                    shutil.rmtree(dest_folder)
                shutil.copytree(download_path, dest_folder)
                print(f"    => [HOAN TAT]: {dest_folder}")
            except Exception as e:
                print(f"    [!] Loi tai: {e}")
                print(f"    Ban co the mo trinh duyet tai truc tiep tu link: https://www.kaggle.com/datasets/{item['id']}")
                print(f"    sau do giai nen vao: {dest_folder}")

    print("\n" + "=" * 70)
    print("KIEM TRA TRANG THAI DU LIEU RAW TRONG 'data/raw/':")
    for item in os.listdir(DATA_RAW_DIR):
        p = os.path.join(DATA_RAW_DIR, item)
        if os.path.isdir(p):
            count = sum(len(files) for _, _, files in os.walk(p))
            print(f"  - {item:30s}: {count:6d} files")
    print("=" * 70)

if __name__ == "__main__":
    download_all()
