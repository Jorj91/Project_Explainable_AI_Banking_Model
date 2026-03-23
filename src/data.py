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

