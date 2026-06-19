"""Data handling: directory scanning, train/val split, and the PyTorch dataset.

The training data is expected as one sub-folder per class
(``data/train/<ClassName>/*.jpg``) and the test data as a flat folder of
unlabelled images (``data/test/*.jpg``).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .config import MEAN, STD, SEED, VAL_SPLIT

_IMG_EXTS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
# Training augmentation is light and label-preserving: satellite tiles have no
# canonical orientation, so flips and small rotations are safe.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Validation/test transform: deterministic, no augmentation.
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# --------------------------------------------------------------------------- #
# Directory scanning
# --------------------------------------------------------------------------- #
def preprocess(data_folder: str) -> tuple[pd.DataFrame, dict]:
    """Scan a class-per-sub-folder directory.

    Returns a DataFrame with ``folder``, ``file_name`` and a numeric ``label``
    per image, plus a dict mapping class name -> numeric label. Class names are
    sorted so the label assignment is deterministic and reproducible.
    """
    class_names = sorted(
        d for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d))
    )
    label_dict = {name: idx for idx, name in enumerate(class_names)}

    rows = []
    for name in class_names:
        folder = os.path.join(data_folder, name)
        for file_name in sorted(os.listdir(folder)):
            if file_name.lower().endswith(_IMG_EXTS):
                rows.append(
                    {"folder": folder, "file_name": file_name,
                     "label": label_dict[name]}
                )
    df = pd.DataFrame(rows, columns=["folder", "file_name", "label"])
    return df, label_dict


def preprocess_test(data_folder: str) -> pd.DataFrame:
    """Build a DataFrame of ``folder`` + ``file_name`` for a flat test folder."""
    rows = [
        {"folder": data_folder, "file_name": f}
        for f in sorted(os.listdir(data_folder))
        if f.lower().endswith(_IMG_EXTS)
    ]
    return pd.DataFrame(rows, columns=["folder", "file_name"])


def split_data(df: pd.DataFrame, val_size: float = VAL_SPLIT, seed: int = SEED):
    """Stratified, reproducible train/validation split."""
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class EuroSATDataset(Dataset):
    """Dataset backed by a :func:`preprocess` DataFrame.

    The images are small and few, so they are decoded once and cached in RAM to
    keep CPU epochs fast.
    """

    def __init__(self, df: pd.DataFrame, transform=None, has_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.has_labels = has_labels
        self.images = [
            Image.open(os.path.join(r.folder, r.file_name)).convert("RGB").copy()
            for r in self.df.itertuples(index=False)
        ]
        if has_labels:
            self.labels = self.df["label"].to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.has_labels:
            return img, int(self.labels[idx])
        return img


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def compute_norm_stats(df: pd.DataFrame, sample: int = 2000, seed: int = SEED):
    """Compute per-channel mean/std over a sample of the training images.

    Used once to derive the normalization constants stored in the config.
    """
    sub = df.sample(min(sample, len(df)), random_state=seed)
    arr = np.stack([
        np.asarray(Image.open(os.path.join(r.folder, r.file_name)).convert("RGB"))
        for r in sub.itertuples(index=False)
    ]).astype(np.float32) / 255.0
    return arr.mean(axis=(0, 1, 2)), arr.std(axis=(0, 1, 2))
