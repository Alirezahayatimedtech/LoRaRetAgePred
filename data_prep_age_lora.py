#!/usr/bin/env python3
"""
data_prep_age_lora.py

Lightweight data loading and preprocessing utilities for RETFound + LoRA age prediction.
Leverages the same CSV filters used in analyze_age_gap.py, and returns a torch Dataset / DataLoader.

Usage (example):
  python data_prep_age_lora.py --csv metadata/image_age_mapping.csv --batch-size 8 --img-size 256
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms as T

# ------------------- Defaults -------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_ROOT / "metadata/image_age_mapping.csv"
DEFAULT_IMAGE_TYPES = ["BScanThumb", "REGAVG"]
CONTROL_TOKENS = {"controls", "control"}
BASELINE_TOKENS = {"baseline", "base"}
HLS_TOKENS = {"hls", "hls (u)", "hls(u)", "hls u"}
COHORT4_ALT_GROUP = "High_CO2"
UNIT_TO_DAYS = {
    "day": 1.0, "days": 1.0,
    "week": 7.0, "weeks": 7.0,
    "month": 30.0, "months": 30.0,
    "year": 365.0, "years": 365.0,
}


# ------------------- Helpers -------------------
def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def parse_duration(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    txt = str(text).strip()
    if not txt or txt.lower() == "nan":
        return None
    m = re.match(r"^\s*([-+]?\d*\.?\d+)\s*\{?\s*([a-zA-Z]+)\s*\}?\s*$", txt)
    if not m:
        return None
    val = float(m.group(1)); unit = m.group(2).lower()
    mult = UNIT_TO_DAYS.get(unit)
    return val * mult if mult is not None else None


def resolve_image_path(path_str: str, root: Path) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = (root / p).resolve()
    return str(p)


def load_metadata(
    csv_path: Path,
    image_types: List[str],
    day_whitelist: Optional[List[int]] = None,
    include_recovery_days: bool = False,
    recovery_day_min: int = 91,
    cohorts_to_keep: Optional[List[str]] = None,
    exclude_recovery_paths: bool = False,
    include_baseline: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load and filter the metadata CSV (mirrors analyze_age_gap.py filters)."""
    df = pd.read_csv(csv_path)

    # Cohort filter
    if cohorts_to_keep is not None and "cohort" in df.columns:
        keep = {str(c) for c in cohorts_to_keep}
        df = df[df["cohort"].astype(str).isin(keep)]

    # Image-type filter
    allow = {normalize(t) for t in image_types}
    df = df[df["image_type"].fillna("").apply(lambda x: normalize(x) in allow)]

    # Day whitelist & recovery days
    if "day" not in df.columns:
        raise SystemExit("CSV missing 'day' column.")
    day_int = np.rint(df["day"].astype(float).to_numpy()).astype(int)
    mask = np.ones(len(df), dtype=bool)
    if day_whitelist is not None:
        mask &= np.isin(day_int, list(day_whitelist))
        if include_recovery_days:
            mask |= day_int >= int(recovery_day_min)
    elif include_recovery_days:
        recovery_mask = day_int >= int(recovery_day_min)
        if recovery_mask.any():
            print(f"[load_metadata] Including {int(recovery_mask.sum())} recovery-day rows (>= {recovery_day_min}).")
    df = df.loc[mask].copy()
    df["day"] = np.rint(df["day"].astype(float)).astype(int)

    # AGE in days (cohort-specific baselines)
    base_age_map = {"1": 90.0, "2": 90.0, "3": 270.0, "4": 270.0}
    cohort_str = df.get("cohort", "").astype(str).str.strip()
    base_age = cohort_str.map(base_age_map).fillna(90.0)
    df["AGE"] = base_age + df["day"].astype(float)
    if "final_age_days" in df.columns:
        df["final_age_days"] = df["final_age_days"].astype(float)
    if "base_age_days" in df.columns:
        df["base_age_days"] = df["base_age_days"].astype(float)

    # Group normalization
    grp_raw = df.get("group_from_path").fillna("").astype(str).apply(normalize)
    is_ctrl = grp_raw.isin({normalize(x) for x in CONTROL_TOKENS})
    is_base = grp_raw.isin({normalize(x) for x in BASELINE_TOKENS})
    is_hls = grp_raw.isin({normalize(x) for x in HLS_TOKENS})
    df["group_norm"] = np.where(
        is_ctrl,
        "Controls",
        np.where(is_base, "Baseline", np.where(is_hls, "HLS (U)", "Unknown")),
    )
    # If group_from_path is missing, fall back to hindlimb_unloading metadata
    if "hindlimb_unloading" in df.columns:
        hlu_raw = df["hindlimb_unloading"].fillna("").astype(str).apply(normalize)
        hlu_ctrl_tokens = {normalize(x) for x in CONTROL_TOKENS} | {"normallyloadedcontrol", "normallyloadedcontrols"}
        hlu_base_tokens = {normalize(x) for x in BASELINE_TOKENS}
        hlu_hls_tokens = {"hindlimbunloaded", "hindlimbunloading"} | {normalize(x) for x in HLS_TOKENS}
        is_ctrl_hlu = hlu_raw.isin(hlu_ctrl_tokens)
        is_base_hlu = hlu_raw.isin(hlu_base_tokens)
        is_hls_hlu = hlu_raw.isin(hlu_hls_tokens)
        unknown_mask = df["group_norm"] == "Unknown"
        df.loc[unknown_mask & is_ctrl_hlu, "group_norm"] = "Controls"
        df.loc[unknown_mask & is_base_hlu, "group_norm"] = "Baseline"
        df.loc[unknown_mask & is_hls_hlu, "group_norm"] = "HLS (U)"

    # Cohort 4: tag as high CO2 based on path (avoid mixing baseline/end/recovery per rat)
    if "cohort" in df.columns:
        cohort_str = df["cohort"].astype(str)
        mask_c4 = cohort_str == "4"
        if mask_c4.any():
            path_lower = df["image_path"].astype(str).str.lower()
            is_hlsu_path = path_lower.str.contains("hls_u")
            is_ctrl_path = path_lower.str.contains("cohort_4_control")
            df.loc[mask_c4 & is_hlsu_path, "group_norm"] = "High_CO2_HLS"
            df.loc[mask_c4 & is_ctrl_path, "group_norm"] = "High_CO2_Controls"
            # Any remaining cohort4 unknowns default to High_CO2_Controls
            df.loc[mask_c4 & (df["group_norm"] == "Unknown"), "group_norm"] = "High_CO2_Controls"

    # Do not remap based on recovery subfolders; keep only primary group tokens
    allowed_groups = ["Controls", "HLS (U)", "High_CO2_Controls", "High_CO2_HLS"]
    if include_baseline:
        allowed_groups.append("Baseline")
    df = df[df["group_norm"].isin(allowed_groups)]

    # Rat ID inference from path/sample
    def infer_rat_from_text(txt: str) -> Optional[str]:
        s = str(txt)
        m = re.search(r"\b(MT|FT)\s*[-_ ]?(\d+)\b", s, flags=re.IGNORECASE)
        if m:
            return (m.group(1).upper() + m.group(2))
        m = re.search(r"\b(MT\d+|FT\d+)\b", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    if "rat_id" not in df.columns:
        df["rat_id"] = ""
    rat_raw = df["rat_id"].astype(str)
    miss_mask = rat_raw.isna() | (rat_raw.str.strip() == "") | (rat_raw.str.lower() == "nan")
    if miss_mask.any():
        inferred = df.loc[miss_mask, "image_path"].astype(str).apply(infer_rat_from_text)
        if "sample_name" in df.columns:
            inferred = inferred.fillna(df.loc[miss_mask, "sample_name"].astype(str).apply(infer_rat_from_text))
        n_filled = int(inferred.notna().sum())
        if n_filled:
            df.loc[miss_mask, "rat_id"] = df.loc[miss_mask, "rat_id"].mask(miss_mask, inferred)
            print(f"[load_metadata] Filled rat_id from path/sample for {n_filled} rows.")

    # Parse durations
    for col in ["hindlimb_unloading_duration", "hindlimb_reloading_duration", "experiment_duration", "age"]:
        if col in df.columns:
            df[col + "_days"] = df[col].apply(parse_duration)

    # Resolve image paths to absolute
    df["image_path"] = df["image_path"].apply(lambda p: resolve_image_path(p, PROJECT_ROOT))

    # Quick summary
    present_days = sorted(df['day'].unique().tolist())
    present_coh = sorted(df['cohort'].astype(str).unique().tolist()) if 'cohort' in df.columns else ['n/a']
    if verbose:
        print(f"[INFO] After filters: N={len(df)} | cohorts={present_coh} | days={present_days}")
    return df


# ------------------- Transforms -------------------

class RobustIntensityNormalize:
    """
    Deterministic percentile-based intensity normalization.

    Designed for OCT-like images with large dark background regions:
    - estimate robust low/high percentiles from non-near-black pixels when possible
    - apply the same linear rescale to all RGB channels (preserve relative channel structure)
    """

    def __init__(
        self,
        enabled: bool = True,
        p_low: float = 1.0,
        p_high: float = 99.0,
        ignore_near_black: bool = True,
        black_thresh: float = 2.0,
        min_valid_pixels: int = 128,
    ):
        self.enabled = enabled
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.ignore_near_black = bool(ignore_near_black)
        self.black_thresh = float(black_thresh)
        self.min_valid_pixels = int(min_valid_pixels)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.enabled:
            return img

        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return img

        gray = arr.mean(axis=2)
        mask = np.ones_like(gray, dtype=bool)
        if self.ignore_near_black:
            mask = gray > self.black_thresh
            if int(mask.sum()) < self.min_valid_pixels:
                mask = np.ones_like(gray, dtype=bool)

        vals = gray[mask]
        if vals.size == 0:
            return img

        lo = float(np.percentile(vals, self.p_low))
        hi = float(np.percentile(vals, self.p_high))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
            return img

        arr = (arr - lo) / (hi - lo)
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")


class RandomPhotometricAugment:
    """
    Tensor-space photometric augmentation (train-only).

    Simulates acquisition/environment variability:
    - global exposure/brightness gain + bias
    - contrast scaling around image mean
    - gamma shift
    - simple illumination gradients and mild vignette

    Expects tensor in [0, 1] before ImageNet normalization.
    """

    def __init__(
        self,
        p: float = 0.7,
        gain_range=(0.85, 1.20),
        bias_range=(-0.06, 0.06),
        contrast_range=(0.85, 1.20),
        gamma_range=(0.80, 1.25),
        grad_p: float = 0.35,
        grad_strength: float = 0.18,
        vignette_p: float = 0.20,
        vignette_strength: float = 0.20,
    ):
        self.p = float(p)
        self.gain_range = tuple(gain_range)
        self.bias_range = tuple(bias_range)
        self.contrast_range = tuple(contrast_range)
        self.gamma_range = tuple(gamma_range)
        self.grad_p = float(grad_p)
        self.grad_strength = float(grad_strength)
        self.vignette_p = float(vignette_p)
        self.vignette_strength = float(vignette_strength)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self.p:
            return x
        if not torch.is_tensor(x):
            return x
        if x.ndim != 3:
            return x

        x = x.clone()
        c, h, w = x.shape

        # Global exposure + black-level shift.
        gain = float(np.random.uniform(*self.gain_range))
        bias = float(np.random.uniform(*self.bias_range))
        x = x * gain + bias

        # Contrast around per-channel mean.
        contrast = float(np.random.uniform(*self.contrast_range))
        mean = x.mean(dim=(1, 2), keepdim=True)
        x = (x - mean) * contrast + mean

        # Gamma adjustment on clipped intensities.
        gamma = float(np.random.uniform(*self.gamma_range))
        x = torch.pow(torch.clamp(x, 0.0, 1.0), gamma)

        # Simulate uneven illumination/exposure across width or height.
        if np.random.rand() < self.grad_p:
            if np.random.rand() < 0.5:
                axis = torch.linspace(-1.0, 1.0, steps=w, dtype=x.dtype, device=x.device).view(1, 1, w)
            else:
                axis = torch.linspace(-1.0, 1.0, steps=h, dtype=x.dtype, device=x.device).view(1, h, 1)
            slope = float(np.random.uniform(-self.grad_strength, self.grad_strength))
            offset = float(np.random.uniform(-0.5 * self.grad_strength, 0.5 * self.grad_strength))
            illum = 1.0 + slope * axis + offset
            x = x * illum

        # Mild vignette / center-brightness variation.
        if np.random.rand() < self.vignette_p:
            yy = torch.linspace(-1.0, 1.0, steps=h, dtype=x.dtype, device=x.device).view(1, h, 1)
            xx = torch.linspace(-1.0, 1.0, steps=w, dtype=x.dtype, device=x.device).view(1, 1, w)
            r2 = xx * xx + yy * yy
            strength = float(np.random.uniform(0.0, self.vignette_strength))
            sign = -1.0 if np.random.rand() < 0.5 else 1.0  # darken or brighten periphery
            vignette = 1.0 + sign * strength * r2
            x = x * vignette

        return torch.clamp(x, 0.0, 1.0)


def make_transform(
    img_size: int = 256,
    train: bool = False,
    aug_level: str = "medium",
    enable_photometric_aug: bool = True,
) -> T.Compose:
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]

    class SquarePad:
        def __call__(self, img):
            w, h = img.size
            max_side = max(w, h)
            pad_w = (max_side - w) // 2
            pad_h = (max_side - h) // 2
            padding = (pad_w, pad_h, max_side - w - pad_w, max_side - h - pad_h)
            return ImageOps.expand(img, padding, fill=0)

    class RandomGamma:
        def __init__(self, gamma_range=(0.7, 1.5), p: float = 0.8):
            self.low, self.high = gamma_range
            self.p = p

        def __call__(self, img):
            if np.random.rand() < self.p:
                gamma = np.random.uniform(self.low, self.high)
                return T.functional.adjust_gamma(img, gamma, gain=1.0)
            return img

    class AddGaussianNoise:
        def __init__(self, std: float = 0.02, p: float = 0.8):
            self.std = std
            self.p = p

        def __call__(self, tensor):
            if np.random.rand() < self.p:
                noise = torch.randn_like(tensor) * self.std
                return tensor + noise
            return tensor

    # Fixed robust intensity normalization is applied in both train and eval
    # to reduce brightness/contrast domain shift across eyes/sessions.
    robust_intensity_norm = RobustIntensityNormalize(enabled=True, p_low=1.0, p_high=99.0)

    level = str(aug_level).strip().lower()
    aug_params = {
        "low": {
            "rot": 5,
            "crop_scale": (0.9, 1.0),
            "jitter": 0.05,
            "gamma_range": (0.8, 1.2),
            "gamma_p": 0.5,
            "blur_sigma": (0.1, 0.5),
            "noise_std": 0.01,
            "noise_p": 0.5,
            "photo_p": 0.35,
            "photo_gain": (0.92, 1.10),
            "photo_bias": (-0.03, 0.03),
            "photo_contrast": (0.92, 1.10),
            "photo_gamma": (0.90, 1.15),
            "photo_grad_p": 0.20,
            "photo_grad_strength": 0.08,
            "photo_vignette_p": 0.10,
            "photo_vignette_strength": 0.08,
        },
        "medium": {
            "rot": 10,
            "crop_scale": (0.8, 1.0),
            "jitter": 0.1,
            "gamma_range": (0.6, 1.6),
            "gamma_p": 0.8,
            "blur_sigma": (0.1, 1.5),
            "noise_std": 0.02,
            "noise_p": 0.8,
            # Milder default profile: focus on small global photometric shifts,
            # avoid strong spatial illumination artifacts that harmed inter-eye consistency.
            "photo_p": 0.30,
            "photo_gain": (0.95, 1.08),
            "photo_bias": (-0.02, 0.02),
            "photo_contrast": (0.95, 1.08),
            "photo_gamma": (0.90, 1.12),
            "photo_grad_p": 0.0,
            "photo_grad_strength": 0.08,
            "photo_vignette_p": 0.0,
            "photo_vignette_strength": 0.08,
        },
        "high": {
            "rot": 15,
            "crop_scale": (0.7, 1.0),
            "jitter": 0.2,
            "gamma_range": (0.5, 1.8),
            "gamma_p": 0.9,
            "blur_sigma": (0.1, 2.0),
            "noise_std": 0.03,
            "noise_p": 0.9,
            "photo_p": 0.85,
            "photo_gain": (0.75, 1.30),
            "photo_bias": (-0.08, 0.08),
            "photo_contrast": (0.75, 1.30),
            "photo_gamma": (0.70, 1.35),
            "photo_grad_p": 0.50,
            "photo_grad_strength": 0.24,
            "photo_vignette_p": 0.25,
            "photo_vignette_strength": 0.24,
        },
    }
    params = aug_params.get(level, aug_params["medium"])

    if train:
        aug = [
            robust_intensity_norm,
            SquarePad(),
            T.RandomRotation(degrees=params["rot"]),
            T.RandomResizedCrop(img_size, scale=params["crop_scale"]),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=params["jitter"], contrast=params["jitter"]),
            RandomGamma(gamma_range=params["gamma_range"], p=params["gamma_p"]),
            T.GaussianBlur(kernel_size=3, sigma=params["blur_sigma"]),
        ]
    else:
        aug = [
            robust_intensity_norm,
            SquarePad(),
            T.Resize((img_size, img_size)),
        ]
    tail = [T.ToTensor()]
    # Keep validation/test deterministic: no stochastic noise at eval time.
    if train:
        if enable_photometric_aug:
            tail.append(RandomPhotometricAugment(
                p=params["photo_p"],
                gain_range=params["photo_gain"],
                bias_range=params["photo_bias"],
                contrast_range=params["photo_contrast"],
                gamma_range=params["photo_gamma"],
                grad_p=params["photo_grad_p"],
                grad_strength=params["photo_grad_strength"],
                vignette_p=params["photo_vignette_p"],
                vignette_strength=params["photo_vignette_strength"],
            ))
        tail.append(AddGaussianNoise(std=params["noise_std"], p=params["noise_p"]))
    tail.append(T.Normalize(mean, std))
    return T.Compose(aug + tail)


# ------------------- Dataset -------------------
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


def collate_bag_batch(batch):
    """Collate variable-length bags (each bag is a tensor [N_i, C, H, W])."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    out = {
        "bags": [b["images"] for b in batch],
        "bag_sizes": torch.as_tensor([int(b["images"].shape[0]) for b in batch], dtype=torch.long),
        "day": torch.as_tensor([float(b["day"]) for b in batch], dtype=torch.float32),
        "age_days": torch.as_tensor([float(b["age_days"]) for b in batch], dtype=torch.float32),
        "group": [b["group"] for b in batch],
        "rat_id": [b["rat_id"] for b in batch],
        "eye": [b["eye"] for b in batch],
        "sex": [b["sex"] for b in batch],
        "cohort": [b["cohort"] for b in batch],
        "paths": [b.get("paths", []) for b in batch],
    }
    return out


class AgeImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: T.Compose, skip_broken: bool = True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.skip_broken = skip_broken
        self.canonicalize_os_to_od = False  # do not mirror OS to OD; keep original geometry

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row["image_path"])
        try:
            with Image.open(path).convert("RGB") as im:
                if self.canonicalize_os_to_od and str(row.get("eye", "")).strip().upper() == "OS":
                    # Canonicalize left eyes to right-eye geometry to reduce OD/OS divergence
                    im = ImageOps.mirror(im)
                img = self.transform(im)
        except Exception as e:
            if self.skip_broken:
                return None
            raise FileNotFoundError(f"Failed to load image: {path}") from e

        sample = {
            "image": img,
            "day": float(row["day"]),
            "age_days": float(row.get("AGE", math.nan)),
            "group": row.get("group_norm", "Unknown"),
            "rat_id": row.get("rat_id", ""),
            "eye": row.get("eye", "Unknown"),
            "sex": row.get("sex", "Unknown"),
            "cohort": row.get("cohort", "Unknown"),
            "path": str(path),
        }
        return sample


class AgeBagDataset(Dataset):
    """
    Bag dataset for MIL where one sample = all images for a (rat_id, eye, day) case.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: T.Compose,
        bag_keys=("rat_id", "eye", "day"),
        skip_broken: bool = True,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.skip_broken = skip_broken
        self.bag_keys = tuple(bag_keys)
        self.canonicalize_os_to_od = False

        if self.df.empty:
            self.groups = []
            return

        grouped = self.df.groupby(list(self.bag_keys), sort=False)
        self.groups = []
        for key, grp in grouped:
            rows = grp.index.to_list()
            if not rows:
                continue
            self.groups.append((key, rows))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.groups):
            raise IndexError(idx)
        _, row_indices = self.groups[idx]
        imgs = []
        paths = []
        meta_row = None
        for ridx in row_indices:
            row = self.df.iloc[ridx]
            meta_row = row
            path = Path(row["image_path"])
            try:
                with Image.open(path).convert("RGB") as im:
                    if self.canonicalize_os_to_od and str(row.get("eye", "")).strip().upper() == "OS":
                        im = ImageOps.mirror(im)
                    img = self.transform(im)
            except Exception as e:
                if self.skip_broken:
                    continue
                raise FileNotFoundError(f"Failed to load image: {path}") from e
            imgs.append(img)
            paths.append(str(path))

        if not imgs or meta_row is None:
            return None

        bag = torch.stack(imgs, dim=0)
        sample = {
            "images": bag,
            "day": float(meta_row["day"]),
            "age_days": float(meta_row.get("AGE", math.nan)),
            "group": meta_row.get("group_norm", "Unknown"),
            "rat_id": meta_row.get("rat_id", ""),
            "eye": meta_row.get("eye", "Unknown"),
            "sex": meta_row.get("sex", "Unknown"),
            "cohort": meta_row.get("cohort", "Unknown"),
            "paths": paths,
        }
        return sample


# ------------------- DataLoader builder -------------------
def make_dataloaders(
    df: pd.DataFrame,
    img_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: int = 42,
    aug_level: str = "medium",
    enable_photometric_aug: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Split dataframe into train/val and create loaders."""
    tf_train = make_transform(img_size=img_size, train=True, aug_level=aug_level, enable_photometric_aug=enable_photometric_aug)
    tf_val = make_transform(img_size=img_size, train=False)

    n_total = len(df)
    n_val = max(1, int(n_total * val_split)) if n_total > 1 else 0
    n_train = n_total - n_val
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_ds = AgeImageDataset(df.iloc[train_idx], tf_train) if n_train > 0 else None
    val_ds   = AgeImageDataset(df.iloc[val_idx], tf_val) if n_val > 0 else None

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin) if train_ds else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin) if val_ds else None
    return train_loader, val_loader


# ------------------- Quality checks -------------------
def run_quality_check(df: pd.DataFrame, img_size: int, limit: int = 500) -> None:
    """Quick pass to catch missing/duplicate/corrupted images and show day/group coverage."""
    total = len(df)
    dup_count = int(df["image_path"].duplicated().sum())
    dup_paths = df["image_path"][df["image_path"].duplicated()].unique().tolist()

    missing = 0
    unreadable = 0
    seen_sizes: Counter = Counter()
    seen_modes: Counter = Counter()

    to_check = total if limit == 0 else min(total, limit)
    for i in range(to_check):
        path = Path(df.iloc[i]["image_path"])
        if not path.exists():
            missing += 1
            continue
        try:
            with Image.open(path) as im:
                seen_modes[im.mode] += 1
                seen_sizes[im.size] += 1
                # ensure convertible to RGB and resizable
                _ = im.convert("RGB").resize((img_size, img_size))
        except Exception:
            unreadable += 1

    day_counts = df["day"].value_counts().sort_index().to_dict()
    group_counts = df["group_norm"].value_counts().to_dict()

    print("[QC] rows=", total)
    print(f"[QC] duplicates={dup_count}{' (showing first 3: ' + ', '.join(dup_paths[:3]) + ')' if dup_paths else ''}")
    print(f"[QC] missing_files={missing} unreadable={unreadable} (checked {to_check} images; limit={limit})")
    if seen_modes:
        common_mode, common_mode_n = seen_modes.most_common(1)[0]
        print(f"[QC] common mode={common_mode} ({common_mode_n}/{max(1,to_check)})")
    if seen_sizes:
        common_size, common_size_n = seen_sizes.most_common(1)[0]
        print(f"[QC] common size={common_size} ({common_size_n}/{max(1,to_check)})")
    print(f"[QC] day distribution: {day_counts}")
    print(f"[QC] group distribution: {group_counts}")


# ------------------- CLI -------------------
def parse_args():
    p = argparse.ArgumentParser(description="RETFound LoRA age data loader")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to metadata CSV")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--day-whitelist", type=int, nargs="*", default=[0, 90, 180], help="Days to keep")
    p.add_argument("--include-recovery-days", action="store_true")
    p.add_argument("--recovery-day-min", type=int, default=91)
    p.add_argument("--cohorts", type=str, nargs="*", default=["2"], help="Cohorts to keep (string match)")
    p.add_argument("--exclude-recovery-paths", action="store_true")
    p.add_argument("--quality-check", action="store_true", help="Run a quick quality check over images and metadata")
    p.add_argument("--quality-limit", type=int, default=500, help="Max images to load for quality check (0 = all)")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_metadata(
        csv_path=args.csv,
        image_types=DEFAULT_IMAGE_TYPES,
        day_whitelist=args.day_whitelist,
        include_recovery_days=args.include_recovery_days,
        recovery_day_min=args.recovery_day_min,
        cohorts_to_keep=args.cohorts,
        exclude_recovery_paths=args.exclude_recovery_paths,
    )

    train_loader, val_loader = make_dataloaders(
        df=df,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    n_train = len(train_loader.dataset) if train_loader and train_loader.dataset else 0
    n_val = len(val_loader.dataset) if val_loader and val_loader.dataset else 0
    print(f"[DATA] train={n_train}, val={n_val}, img_size={args.img_size}, batch_size={args.batch_size}")

    # Quick sanity pass over one batch
    if train_loader:
        batch = next(iter(train_loader))
        imgs = batch["image"]
        days = batch["day"]
        print(f"[SANITY] batch images shape={tuple(imgs.shape)}, day range=({float(days.min()):.1f}, {float(days.max()):.1f})")

    if args.quality_check:
        run_quality_check(df, img_size=args.img_size, limit=args.quality_limit)


if __name__ == "__main__":
    main()
