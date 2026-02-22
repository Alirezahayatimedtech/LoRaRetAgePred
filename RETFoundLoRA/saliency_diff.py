#!/usr/bin/env python3
"""
Aggregate saliency maps for Controls vs stressed groups and save mean/difference heatmaps.
Assumes saliency PNGs were saved via run.py with --save-saliency-dir.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import OUTPUT_ROOT


def load_mean_maps(csv_path: Path, groups: List[str], sal_dir: Path, drop_day0: bool, cohorts: Optional[List[str]]) -> Optional[np.ndarray]:
    df = pd.read_csv(csv_path)
    if drop_day0:
        df = df[df["day"] > 0]
    df = df[df["group"].isin(groups)]
    if cohorts is not None:
        df = df[df["cohort"].astype(str).isin(cohorts)]
    imgs = []
    match_count = 0
    for _, r in df.iterrows():
        stem = f"{r['rat_id']}_{r['eye']}_{float(r['day']):.1f}"
        for p in sal_dir.glob(f"{stem}*.png"):
            arr = np.array(Image.open(p)).astype(np.float32) / 255.0
            if arr.ndim == 3:  # overlay RGB -> grayscale intensity
                arr = arr.mean(axis=2)
            imgs.append(arr)
            match_count += 1
    if match_count == 0:
        print(f"[WARN] No saliency PNGs found for groups {groups} in {sal_dir}")
        return None
    else:
        print(f"[INFO] Found {match_count} saliency PNGs for groups {groups}")
    if not imgs:
        return None
    # align shapes by cropping to smallest H/W
    h = min(im.shape[0] for im in imgs)
    w = min(im.shape[1] for im in imgs)
    imgs = [im[:h, :w] for im in imgs]
    return np.mean(np.stack(imgs, axis=0), axis=0)


def save_map(path: Path, arr: np.ndarray, cmap: str = "coolwarm"):
    plt.imshow(arr, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Differential saliency: Controls vs stressed")
    ap.add_argument("--saliency-dir", type=Path, default=OUTPUT_ROOT / "saliency_maps",
                    help="Directory containing saliency PNGs from run.py (stressed by default)")
    ap.add_argument("--control-saliency-dir", type=Path, default=None,
                    help="Optional directory for control saliency PNGs (defaults to --saliency-dir)")
    ap.add_argument("--control-csv", type=Path, default=OUTPUT_ROOT / "predictions/control_test_results.csv",
                    help="Predictions CSV for Controls")
    ap.add_argument("--stress-csv", type=Path, default=OUTPUT_ROOT / "predictions/rag_experimental_results.csv",
                    help="Predictions CSV for stressed groups")
    ap.add_argument("--stress-groups", type=str, nargs="+",
                    default=["HLS (U)", "High_CO2_Controls", "High_CO2_HLS"],
                    help="Groups to include as stressed")
    ap.add_argument("--drop-day0", action="store_true", help="Exclude day 0 from aggregation")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "saliency_diff",
                    help="Output directory for aggregated maps")
    ap.add_argument("--cohorts", type=str, nargs="*", default=["1", "2"],
                    help="Optional cohort filter (as strings)")
    args = ap.parse_args()

    sal_ctrl_dir = args.control_saliency_dir if args.control_saliency_dir else args.saliency_dir
    ctrl_mean = load_mean_maps(args.control_csv, ["Controls"], sal_ctrl_dir, args.drop_day0, args.cohorts)
    stress_mean = load_mean_maps(args.stress_csv, args.stress_groups, args.saliency_dir, args.drop_day0, args.cohorts)

    if ctrl_mean is not None:
        save_map(args.out_dir / "control_mean.png", ctrl_mean, cmap="magma")
    if stress_mean is not None:
        save_map(args.out_dir / "stress_mean.png", stress_mean, cmap="magma")

    if ctrl_mean is not None and stress_mean is not None:
        h = min(ctrl_mean.shape[0], stress_mean.shape[0])
        w = min(ctrl_mean.shape[1], stress_mean.shape[1])
        ctrl_crop = ctrl_mean[:h, :w]
        stress_crop = stress_mean[:h, :w]
        diff = stress_crop - ctrl_crop
        save_map(args.out_dir / "diff.png", diff, cmap="coolwarm")
        rng = diff.max() - diff.min() + 1e-6
        norm = (diff - diff.min()) / rng
        save_map(args.out_dir / "diff_norm.png", norm, cmap="coolwarm")

    print(f"[DONE] Wrote saliency aggregates to {args.out_dir}")


if __name__ == "__main__":
    main()
