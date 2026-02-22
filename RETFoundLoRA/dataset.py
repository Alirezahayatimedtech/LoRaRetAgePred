"""
dataset.py

Custom loaders for RETFound LoRA age tasks with explicit Ground (train) vs HLS/Recovery (test) separation.
Supports:
- Raw image loading for feature extraction.
- Precomputed feature loading from .npy files for training/eval.

Directory expectation: group folders (e.g., Controls, HLS, Recovery) containing week/day subfolders with images or .npy files.
Labeling: you supply a resolver to convert a file path into a chronological age (days). This keeps labeling logic explicit and avoids silent mistakes.
"""

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from config import (
    GROUND_FOLDER,
    HLS_FOLDER,
    RECOVERY_FOLDER,
    BATCH_SIZE,
    NUM_WORKERS,
    IMG_SIZE,
)
from data_prep_age_lora import make_transform


# ------------------- Defaults -------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FEATURE_EXT = ".npy"


def default_age_resolver(path: Path) -> float:
    """Example resolver: extracts week number from folder name like 'week14' -> 98 days."""
    m = re.search(r"week\s*(\d+)", path.as_posix(), flags=re.IGNORECASE)
    if m:
        week = int(m.group(1))
        return float(week * 7)
    raise ValueError(f"Cannot infer age from path {path}")


def _collect_files(root: Path, groups: Sequence[str], allowed_exts: set) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    for grp in groups:
        gdir = root / grp
        if not gdir.exists():
            continue
        for p in gdir.rglob("*"):
            if p.suffix.lower() in allowed_exts and p.is_file():
                items.append((p, grp))
    return items


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: Path,
        groups: Sequence[str],
        age_resolver: Callable[[Path], float],
        img_size: int = IMG_SIZE,
        transform=None,
    ):
        self.root = Path(root)
        self.groups = list(groups)
        self.age_resolver = age_resolver
        self.paths = _collect_files(self.root, self.groups, IMAGE_EXTS)
        self.transform = transform or make_transform(img_size=img_size, train=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, grp = self.paths[idx]
        with Image.open(path).convert("RGB") as im:
            img = self.transform(im)
        age_days = self.age_resolver(path)
        return {"image": img, "age_days": float(age_days), "group": grp, "path": str(path)}


class FeatureNPYDataset(Dataset):
    def __init__(
        self,
        root: Path,
        groups: Sequence[str],
        age_resolver: Callable[[Path], float],
    ):
        self.root = Path(root)
        self.groups = list(groups)
        self.age_resolver = age_resolver
        self.paths = _collect_files(self.root, self.groups, {FEATURE_EXT})

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, grp = self.paths[idx]
        feats = np.load(path)
        age_days = self.age_resolver(path)
        return {
            "features": torch.as_tensor(feats, dtype=torch.float32),
            "age_days": float(age_days),
            "group": grp,
            "path": str(path),
        }


def build_group_loaders(
    *,
    ground_root: Path = GROUND_FOLDER,
    hls_root: Path = HLS_FOLDER,
    recovery_root: Optional[Path] = RECOVERY_FOLDER,
    train_groups: Sequence[str] = ("Controls",),
    test_groups: Sequence[str] = ("HLS", "Recovery"),
    age_resolver: Callable[[Path], float] = default_age_resolver,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    img_size: int = IMG_SIZE,
    use_features: bool = False,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Build train (ground) and test (HLS/Recovery) loaders with strict group separation.
    Set use_features=True to load .npy features instead of images.
    """
    train_ds_cls = FeatureNPYDataset if use_features else ImageFolderDataset
    test_ds_cls = FeatureNPYDataset if use_features else ImageFolderDataset

    train_ds = train_ds_cls(ground_root, train_groups, age_resolver, img_size=img_size) if ground_root else None

    test_roots = [r for r in [hls_root, recovery_root] if r is not None]
    test_paths = []
    for root in test_roots:
        ds = test_ds_cls(root, test_groups, age_resolver, img_size=img_size)
        test_paths.append(ds)
    # Concatenate test datasets (simple) if multiple roots
    if len(test_paths) == 1:
        test_ds = test_paths[0]
    elif len(test_paths) > 1:
        test_ds = torch.utils.data.ConcatDataset(test_paths)
    else:
        test_ds = None

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin) if train_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin) if test_ds else None
    return train_loader, test_loader

