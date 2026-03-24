# src/data.py

import os
import zipfile

from pathlib import Path
import shutil
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



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

    # print("Dataset ready.")
    print(f"Dataset ready at: {Path(data_dir).resolve()}")



def quick_stats(data_dir="data/signatures"):
    base = Path(data_dir)

    writers = sorted([d for d in base.iterdir() if d.is_dir()],
                     key=lambda x: int(x.name.split("_")[-1]))

    orig_counts = []
    forg_counts = []

    for w in writers:
        files = list(w.glob("*.png"))

        orig = sum("original" in f.name for f in files)
        forg = sum("forg" in f.name for f in files)

        orig_counts.append(orig)
        forg_counts.append(forg)

    total_original = sum(orig_counts)
    total_forged = sum(forg_counts)
    total_images = total_original + total_forged

    # ---- PRINT ---
    print("\n📊 Dataset Statistics")
    print("-" * 30)
    print(f"Total images: {total_images}")
    print(f"Genuine:     {total_original}")
    print(f"Forged:       {total_forged}\n")

    colors = ["#D6CDEA", "#7E57C2"]

    # ---- PIE CHART ----
    print("\n")
    plt.figure(figsize=(5, 5))
    plt.pie([total_original, total_forged],
            labels=["Genuine", "Forged"],
            autopct="%1.1f%%",
            colors=colors,
            startangle=90)
    plt.title("Signature Type Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print("\n")

    # ---- BARPLOT----
    print("\n")
    x = range(len(writers))
    plt.figure(figsize=(10, 4))
    plt.bar(x, orig_counts, label="Genuine", color=colors[0])
    plt.bar(x, forg_counts, bottom=orig_counts, label="Forged", color=colors[1])
    plt.title("Signatures per Writer", fontsize=14, fontweight="bold")
    plt.legend()
    plt.xticks(x, [w.name.split("_")[-1] for w in writers], rotation=45)
    plt.tight_layout()
    plt.show()


    

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

    print("Folder structure created")


def copy_images(writer_list, split, base):

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

