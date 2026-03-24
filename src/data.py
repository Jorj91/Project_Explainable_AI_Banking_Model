# src/data.py

import os
import zipfile

from pathlib import Path
import shutil
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def download_and_extract_dataset(
    dataset_name="matteocarnebella/cedar-signatures",
    zip_name="cedar-signatures.zip",
    data_dir="data"
):
    # Skip if already exists
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print("Dataset already exists. Skipping download.")
        return

    print("Downloading dataset...")
    os.system(f"kaggle datasets download -d {dataset_name}")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    print("Dataset ready.")


def get_image_files(data_path):
    files = sorted(list(Path(data_path).glob("*.png")))
    genuine = [f for f in files if "original" in f.name.lower()]
    forged = [f for f in files if "forg" in f.name.lower()]
    return files, genuine, forged


def get_writers(root="data/signatures"):
    return list(Path(root).glob("signatures_*"))


def split_writers(writers, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    writers = writers.copy()
    random.shuffle(writers)

    n = len(writers)

    train_writers = writers[:int(train_ratio*n)]
    val_writers = writers[int(train_ratio*n):int((train_ratio+val_ratio)*n)]
    test_writers = writers[int((train_ratio+val_ratio)*n):]

    return train_writers, val_writers, test_writers


def create_split_folders(base="data"):
    base = Path(base)

    for split in ["train", "val", "test"]:
        for cls in ["genuine", "forged"]:
            (base / split / cls).mkdir(parents=True, exist_ok=True)

    print(f"Folder structure created under: {base.resolve()}")


def copy_images(writer_list, split, base="data"):
    base = Path(base)

    for writer in writer_list:
        for img in writer.glob("*.png"):

            name = img.name.lower()

            if "original" in name:
                dst = base / split / "genuine" / img.name
            elif "forgeries" in name:
                dst = base / split / "forged" / img.name
            else:
                continue

            shutil.copy(img, dst)

