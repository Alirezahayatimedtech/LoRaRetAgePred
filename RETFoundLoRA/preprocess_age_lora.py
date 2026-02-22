"""
Preprocessing utilities for RETFound LoRA age regression.
Handles metadata loading, group filtering, splitting, and DataLoader creation.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

# Make local modules importable when run as a script
import sys

CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.append(str(CUR_DIR))
if str(CUR_DIR.parent) not in sys.path:
    sys.path.append(str(CUR_DIR.parent))

from data_prep_age_lora import (
    load_metadata,
    make_transform,
    AgeImageDataset,
    AgeBagDataset,
    collate_skip_none,
    collate_bag_batch,
)
from config import (
    CSV_PATH,
    IMAGE_TYPES,
    DAY_WHITELIST,
    COHORTS_TO_KEEP,
)
from utils import normalize_eye_side


def split_dataframe(df, val_split: float, test_split: float, seed: int = 42):
    """
    Rat-level split to prevent leakage across slices/timepoints, stratified by mean AGE bin per rat.
    """
    if val_split < 0 or test_split < 0 or val_split + test_split >= 1:
        raise ValueError("val_split and test_split must be >=0 and sum to < 1")

    rat_stats = df.groupby("rat_id")["AGE"].mean().reset_index()
    ages = rat_stats["AGE"].to_numpy()
    rat_ids = rat_stats["rat_id"].to_numpy()

    # Bin ages for stratification (coarse 2–3 bins) to keep splits stable
    def make_bins(vals: np.ndarray, min_bins: int = 2, max_bins: int = 3):
        if len(vals) <= 1:
            return np.zeros_like(vals, dtype=int)
        for nb in range(max_bins, min_bins - 1, -1):
            edges = np.linspace(vals.min(), vals.max(), num=nb + 1)
            edges = np.unique(edges)
            if len(edges) < 2:
                continue
            bins = np.digitize(vals, edges[1:-1], right=True)
            counts = np.bincount(bins, minlength=nb)
            if counts.min(initial=0) >= 2:  # no singleton bins
                return bins
        # Fallback: coarse binning into 2 bins
        edges = np.linspace(vals.min(), vals.max(), num=3)
        bins = np.digitize(vals, edges[1:-1], right=True)
        return bins

    age_bins = make_bins(ages)

    rng = np.random.default_rng(seed)
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_split + test_split if (val_split + test_split) > 0 else 0.0,
        random_state=seed,
    )
    if (val_split + test_split) > 0 and len(rat_ids) > 1:
        # First split train vs holdout (stratified if possible)
        if len(np.unique(age_bins)) > 1 and min(np.bincount(age_bins)) >= 2:
            train_idx, hold_idx = next(sss.split(rat_ids.reshape(-1, 1), age_bins))
        else:
            rng_split = rng.permutation(len(rat_ids))
            cut = int(len(rat_ids) * (1 - (val_split + test_split)))
            train_idx, hold_idx = rng_split[:cut], rng_split[cut:]
        train_rats = rat_ids[train_idx]
        hold_rats = rat_ids[hold_idx]
        hold_bins = age_bins[hold_idx] if len(age_bins) else np.array([])
        if test_split > 0 and len(hold_rats) > 0:
            if val_split <= 0:
                # all holdout -> test
                val_rats = np.array([])
                test_rats = hold_rats
            else:
                test_frac = test_split / (val_split + test_split)
                if test_frac >= 1.0:
                    # avoid scikit-learn error when val_split is ~0 and test_frac==1
                    val_rats = np.array([])
                    test_rats = hold_rats
                else:
                    # normal val/test split on the holdout pool
                    if len(hold_rats) > 1:
                        if len(np.unique(hold_bins)) > 1 and min(np.bincount(hold_bins)) >= 2:
                            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
                            val_idx_rel, test_idx_rel = next(sss2.split(hold_rats.reshape(-1, 1), hold_bins))
                        else:
                            rng_split2 = rng.permutation(len(hold_rats))
                            cut2 = int(len(hold_rats) * (1 - test_frac))
                            val_idx_rel, test_idx_rel = rng_split2[:cut2], rng_split2[cut2:]
                        val_rats = hold_rats[val_idx_rel] if val_split > 0 else np.array([])
                        test_rats = hold_rats[test_idx_rel] if test_split > 0 else np.array([])
                    else:
                        val_rats = np.array([])
                        test_rats = hold_rats
        else:
            val_rats = hold_rats if val_split > 0 else np.array([])
            test_rats = np.array([])
    else:
        train_rats = rat_ids
        val_rats = np.array([])
        test_rats = np.array([])

    train_df = df[df["rat_id"].isin(train_rats)]
    val_df = df[df["rat_id"].isin(val_rats)] if len(val_rats) else df.iloc[0:0]
    test_df = df[df["rat_id"].isin(test_rats)] if len(test_rats) else df.iloc[0:0]

    print(f"[SPLIT] Rats: Train={len(np.unique(train_rats))}, Val={len(np.unique(val_rats))}, Test={len(np.unique(test_rats))}")
    return train_df, val_df, test_df


def split_dataframe_by_cohort(df, val_split: float, test_split: float, seed: int = 42):
    """
    Rat-level split within each cohort to preserve cohort-specific distributions
    in train/val/test splits.
    """
    if "cohort" not in df.columns:
        # Fallback to global split if cohort column is missing
        return split_dataframe(df, val_split=val_split, test_split=test_split, seed=seed)

    train_parts = []
    val_parts = []
    test_parts = []
    cohorts = sorted(df["cohort"].dropna().astype(str).unique())
    for i, c in enumerate(cohorts):
        sub = df[df["cohort"].astype(str) == str(c)]
        # Offset seed per cohort for determinism without identical splits across cohorts
        seed_c = seed + (i * 101)
        tr, va, te = split_dataframe(sub, val_split=val_split, test_split=test_split, seed=seed_c)
        if not tr.empty:
            train_parts.append(tr)
        if not va.empty:
            val_parts.append(va)
        if not te.empty:
            test_parts.append(te)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0]
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0]
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0]
    return train_df, val_df, test_df


def make_loaders(
    train_df,
    val_df,
    test_df,
    ctrl_test_df,
    img_size: int,
    batch_size: int,
    num_workers: int,
    aug_level: str = "medium",
    enable_photometric_aug: bool = True,
    mil_attention: bool = False,
):
    tf_train = make_transform(
        img_size=img_size,
        train=True,
        aug_level=aug_level,
        enable_photometric_aug=enable_photometric_aug,
    )
    tf_eval = make_transform(img_size=img_size, train=False)
    pin = torch.cuda.is_available()

    if mil_attention:
        train_ds = AgeBagDataset(train_df, tf_train) if len(train_df) else None
        val_ds = AgeBagDataset(val_df, tf_eval) if len(val_df) else None
        test_ds = AgeBagDataset(test_df, tf_eval) if len(test_df) else None
        ctrl_test_ds = AgeBagDataset(ctrl_test_df, tf_eval) if len(ctrl_test_df) else None
    else:
        train_ds = AgeImageDataset(train_df, tf_train) if len(train_df) else None
        val_ds = AgeImageDataset(val_df, tf_eval) if len(val_df) else None
        test_ds = AgeImageDataset(test_df, tf_eval) if len(test_df) else None
        ctrl_test_ds = AgeImageDataset(ctrl_test_df, tf_eval) if len(ctrl_test_df) else None

    if train_ds and len(train_df):
        if mil_attention:
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin,
                collate_fn=collate_bag_batch,
            )
        else:
            ages = train_df["AGE"].to_numpy()
            uniq, counts = np.unique(ages, return_counts=True)
            inv_count = {a: 1.0 / c for a, c in zip(uniq, counts)}
            weights = np.array([inv_count[a] for a in ages], dtype=np.float64)
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                len(weights),
                replacement=True,
            )
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin,
                collate_fn=collate_skip_none,
            )
    else:
        train_loader = None
    bag_collate = collate_bag_batch if mil_attention else collate_skip_none
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, collate_fn=bag_collate) if val_ds else None
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, collate_fn=bag_collate) if test_ds else None
    ctrl_test_loader = torch.utils.data.DataLoader(ctrl_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, collate_fn=bag_collate) if ctrl_test_ds else None
    return train_loader, val_loader, test_loader, ctrl_test_loader


def prepare_data(
    *,
    csv_path: Path = CSV_PATH,
    image_types: List[str] = IMAGE_TYPES,
    day_whitelist: Optional[List[int]] = DAY_WHITELIST,
    test_image_types: Optional[List[str]] = ("REGAVG",),
    test_single_image: bool = False,
    include_recovery_days: bool = False,
    cohorts_to_keep: Optional[List[str]] = COHORTS_TO_KEEP,
    exclude_recovery_paths: bool = False,
    train_groups: List[str] = None,
    test_groups: List[str] = None,
    val_split: float = 0.1,
    test_split: float = 0.1,
    baseline_test_split: float = 0.0,
    holdout_day: Optional[int] = None,
    holdout_test_only: bool = False,
    subset_size: Optional[int] = None,
    subset_fraction: Optional[float] = None,
    img_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    right_eye_only: bool = False,
    aug_level: str = "medium",
    cohort_stratified_split: bool = False,
    enable_photometric_aug: bool = True,
    mil_attention: bool = False,
):
    train_groups_set = set(train_groups or [])
    test_groups_set = set(test_groups or [])
    include_baseline = any(str(g).strip().lower() == "baseline" for g in (train_groups_set | test_groups_set))
    df = load_metadata(
        csv_path=csv_path,
        image_types=image_types,
        day_whitelist=day_whitelist,
        include_recovery_days=include_recovery_days,
        cohorts_to_keep=cohorts_to_keep,
        exclude_recovery_paths=exclude_recovery_paths,
        include_baseline=include_baseline,
    )

    # Clean eye column: ignore metadata eye field; infer from material_type or path
    df["eye"] = df.apply(lambda r: normalize_eye_side(r.get("eye"), r.get("image_path", ""), r.get("material_type", "")), axis=1)

    if right_eye_only:
        before = len(df)
        df = df[df["eye"].str.upper() == "OD"]
        print(f"[INFO] Right-eye only enabled: kept {len(df)} / {before} rows.")

    # Normalize sex column if present
    if "sex" in df.columns:
        df["sex"] = df["sex"].fillna("Unknown").astype(str).str.strip()
    else:
        df["sex"] = "Unknown"

    # Normalize cohort column (prefer cohort_number if present)
    if "cohort_number" in df.columns:
        def _norm_cohort(val):
            if pd.isna(val):
                return None
            s = str(val).strip()
            if not s:
                return None
            try:
                fv = float(s)
                if fv.is_integer():
                    return str(int(fv))
                return str(fv)
            except Exception:
                return s
        cohort_num = df["cohort_number"].apply(_norm_cohort)
        if "cohort" in df.columns:
            df["cohort"] = cohort_num.fillna(df["cohort"])
        else:
            df["cohort"] = cohort_num
    if "cohort" in df.columns:
        df["cohort"] = df["cohort"].fillna("Unknown").astype(str).str.strip()
    else:
        df["cohort"] = "Unknown"

    keep_groups = train_groups_set | test_groups_set if (train_groups_set or test_groups_set) else None
    if keep_groups:
        df = df[df["group_norm"].isin(keep_groups)]
        print(f"[INFO] Kept groups (train+test): {sorted(keep_groups)} -> N={len(df)}")

    train_pool = df[df["group_norm"].isin(train_groups_set)] if train_groups_set else df
    # If no explicit test_groups, leave test_pool empty to avoid reusing train data
    test_pool = df[df["group_norm"].isin(test_groups_set)] if test_groups_set else df.iloc[0:0]

    if holdout_day is not None:
        before = len(train_pool)
        train_pool = train_pool[train_pool["day"] != int(holdout_day)]
        if len(train_pool) != before:
            print(f"[INFO] Holdout day {holdout_day}: removed {before - len(train_pool)} train/val rows")
        if holdout_test_only:
            test_pool = test_pool[test_pool["day"] == int(holdout_day)]
            print(f"[INFO] Holdout test-only day {holdout_day}: test rows={len(test_pool)}")

    # Optional: hold out a fraction of Baseline rats into the test pool (rat-level)
    if baseline_test_split and baseline_test_split > 0:
        base_mask = train_pool["group_norm"] == "Baseline"
        base_df = train_pool[base_mask]
        base_rats = base_df["rat_id"].dropna().unique()
        if len(base_rats) == 0:
            print("[WARN] baseline_test_split set but no Baseline rats found in train_pool.")
        else:
            split = float(baseline_test_split)
            split = max(0.0, min(split, 1.0))
            n_test = int(round(len(base_rats) * split))
            if n_test >= len(base_rats):
                n_test = max(1, len(base_rats) - 1)
            if n_test <= 0:
                print("[WARN] baseline_test_split too small; no Baseline rats held out.")
            else:
                rng = np.random.default_rng(seed)
                test_rats = rng.choice(base_rats, size=n_test, replace=False)
                base_test_df = base_df[base_df["rat_id"].isin(test_rats)]
                base_train_df = base_df[~base_df["rat_id"].isin(test_rats)]
                train_pool = pd.concat([train_pool[~base_mask], base_train_df], ignore_index=True)
                test_pool = pd.concat([test_pool, base_test_df], ignore_index=True).drop_duplicates()
                print(f"[INFO] Baseline holdout: moved {len(base_test_df)} rows from {n_test} rats into test pool.")

    # Optional subsetting of train/val to study data efficiency
    if subset_size is not None or subset_fraction is not None:
        rng = np.random.default_rng(seed)
        n_rows = len(train_pool)
        if subset_size is None and subset_fraction is not None:
            subset_size = max(1, int(round(n_rows * subset_fraction)))
        subset_size = min(subset_size, n_rows)
        idx = rng.choice(train_pool.index.to_numpy(), size=subset_size, replace=False)
        train_pool = train_pool.loc[idx]
        print(f"[INFO] Subset training pool to {subset_size} rows (from {n_rows}) for data-efficiency run")

    if cohort_stratified_split:
        train_df, val_df, ctrl_test_df = split_dataframe_by_cohort(
            train_pool, val_split=val_split, test_split=test_split, seed=seed
        )
    else:
        train_df, val_df, ctrl_test_df = split_dataframe(
            train_pool, val_split=val_split, test_split=test_split, seed=seed
        )

    # If Baseline is in training, keep the Control holdout clean (no Baseline rows)
    if "Baseline" in train_groups_set and baseline_test_split and baseline_test_split > 0:
        base_ctrl = ctrl_test_df[ctrl_test_df["group_norm"] == "Baseline"]
        if len(base_ctrl):
            ctrl_test_df = ctrl_test_df[ctrl_test_df["group_norm"] != "Baseline"]
            train_df = pd.concat([train_df, base_ctrl], ignore_index=True)
            print(f"[INFO] Removed {len(base_ctrl)} Baseline rows from ctrl_test (moved to train).")

    # Prevent leakage: remove any rat that landed in train/val from test_pool
    used_rats = set(train_df["rat_id"]).union(set(val_df["rat_id"]))
    test_df = test_pool[~test_pool["rat_id"].isin(used_rats)].copy()
    overlap = set(test_pool["rat_id"]).intersection(used_rats)
    if overlap:
        print(f"[WARN] Removed {len(overlap)} overlapping rats from test to avoid leakage.")

    # Optional: restrict test/ctrl_test to specific image types (e.g., REGAVG only)
    if test_image_types:
        test_image_types_set = set(test_image_types)
        before_test = len(test_df)
        before_ctrl = len(ctrl_test_df)
        test_df = test_df[test_df["image_type"].isin(test_image_types_set)]
        ctrl_test_df = ctrl_test_df[ctrl_test_df["image_type"].isin(test_image_types_set)]
        print(f"[INFO] Test image types filter {sorted(test_image_types_set)}: test {before_test}->{len(test_df)}, ctrl_test {before_ctrl}->{len(ctrl_test_df)}")

    if test_single_image:
        before_test = len(test_df)
        before_ctrl = len(ctrl_test_df)
        test_df = test_df.drop_duplicates(subset=["rat_id", "eye", "day"])
        ctrl_test_df = ctrl_test_df.drop_duplicates(subset=["rat_id", "eye", "day"])
        print(f"[INFO] Test single-image mode: test {before_test}->{len(test_df)}, ctrl_test {before_ctrl}->{len(ctrl_test_df)} (dedup by rat_id/eye/day)")

    loaders = make_loaders(
        train_df,
        val_df,
        test_df,
        ctrl_test_df,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        aug_level=aug_level,
        enable_photometric_aug=enable_photometric_aug,
        mil_attention=mil_attention,
    )
    return train_df, val_df, ctrl_test_df, test_df, loaders
