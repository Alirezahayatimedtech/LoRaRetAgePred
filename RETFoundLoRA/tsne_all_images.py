#!/usr/bin/env python3
"""
Compute a t-SNE projection for all images in the metadata CSV.
Embeddings are extracted from the RETFound backbone (with optional LoRA adapters)
and reduced to 2D with sklearn's TSNE. Saves both a scatter plot and the raw
coordinates for downstream analysis.

Example:
  python RETFoundLoRA/tsne_all_images.py \
    --csv metadata/image_age_mapping.csv \
    --backbone-ckpt RETFound_MAE_Model/RETFound_mae_natureOCT.pth \
    --output-plot outputs/plots/tsne_all.png
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from config import (
    CSV_PATH,
    BACKBONE_CKPT,
    IMG_SIZE,
    IMAGE_TYPES,
    DAY_WHITELIST,
    COHORTS_TO_KEEP,
    BATCH_SIZE,
    NUM_WORKERS,
    OUTPUT_ROOT,
    LORA_RANK,
    LORA_BLOCKS,
    LORA_ALPHA,
    LORA_DROPOUT,
)
from data_prep_age_lora import load_metadata, make_transform, AgeImageDataset, collate_skip_none
from retfound_lora_age_pred import RETFoundLoRAAgePred


def parse_args():
    p = argparse.ArgumentParser(description="t-SNE visualization of RETFound image embeddings")
    p.add_argument("--csv", type=Path, default=CSV_PATH, help="Metadata CSV (image_age_mapping.csv)")
    p.add_argument("--backbone-ckpt", type=Path, default=BACKBONE_CKPT, help="RETFound MAE checkpoint")
    p.add_argument("--img-size", type=int, default=IMG_SIZE, help="Image resize/crop size")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for feature extraction")
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Dataloader workers")
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--n-components", type=int, default=2, help="t-SNE output dimensions")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images (None = all)")
    p.add_argument("--day-whitelist", type=int, nargs="*", default=None, help="Days to keep (default: all days)")
    p.add_argument("--cohorts", type=str, nargs="*", default=COHORTS_TO_KEEP, help="Cohorts to keep")
    p.add_argument("--groups", type=str, nargs="*", default=None, help="Optional group filter (normalized names)")
    p.add_argument("--include-recovery-days", action="store_true", help="Allow recovery days (>90) even if day whitelist is set")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-plot", type=Path, default=OUTPUT_ROOT / "plots/tsne_all.png")
    p.add_argument("--output-csv", type=Path, default=OUTPUT_ROOT / "plots/tsne_all_coords.csv")
    return p.parse_args()


def build_model(args) -> RETFoundLoRAAgePred:
    model = RETFoundLoRAAgePred(
        ckpt_path=args.backbone_ckpt,
        img_size=args.img_size,
        global_pool=False,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_blocks=LORA_BLOCKS,
        lora_dropout=LORA_DROPOUT,
        upsample_factor=None,  # not needed for embeddings
    )
    return model


def extract_embeddings(model: RETFoundLoRAAgePred, loader: DataLoader, device: torch.device, max_images: Optional[int] = None):
    model.eval()
    embeddings: List[np.ndarray] = []
    meta: List[dict] = []
    processed = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs = batch["image"].to(device, non_blocking=True)
            feats = model.extract_spatial_features(imgs)  # (B, C, H, W)
            pooled = feats.mean(dim=(2, 3))  # (B, C)
            pooled_np = pooled.cpu().numpy()

            for i in range(pooled_np.shape[0]):
                embeddings.append(pooled_np[i])
                meta.append({
                    "rat_id": batch.get("rat_id", [""])[i] if isinstance(batch, dict) else "",
                    "group": batch.get("group", ["Unknown"])[i] if isinstance(batch, dict) else "Unknown",
                    "day": float(batch.get("day", torch.tensor(np.nan))[i]),
                    "cohort": batch.get("cohort", ["Unknown"])[i] if isinstance(batch, dict) else "Unknown",
                    "eye": batch.get("eye", ["Unknown"])[i] if isinstance(batch, dict) else "Unknown",
                })
                processed += 1
                if max_images is not None and processed >= max_images:
                    return np.stack(embeddings, axis=0), pd.DataFrame(meta)
    return (np.stack(embeddings, axis=0) if embeddings else np.empty((0, 0))), pd.DataFrame(meta)


def run_tsne(embeddings: np.ndarray, n_components: int, perplexity: float) -> np.ndarray:
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=42,
    )
    return tsne.fit_transform(embeddings)


def plot_tsne(coords: np.ndarray, meta: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    if meta.empty or coords.shape[0] == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        meta = meta.copy()
        meta["group"] = meta["group"].astype(str)
        groups = sorted(meta["group"].unique())
        cmap = plt.get_cmap("tab10")
        for idx, grp in enumerate(groups):
            mask = meta["group"] == grp
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=12,
                alpha=0.8,
                label=f"{grp} (n={mask.sum()})",
                color=cmap(idx % 10),
            )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("RETFound image embeddings (t-SNE)")
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[PLOT] Saved t-SNE scatter to {out_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"[DEVICE] requested={args.device} | cuda_available={torch.cuda.is_available()} | using={device}")

    df = load_metadata(
        csv_path=args.csv,
        image_types=IMAGE_TYPES,
        day_whitelist=args.day_whitelist if args.day_whitelist else None,
        include_recovery_days=args.include_recovery_days,
        cohorts_to_keep=args.cohorts,
        exclude_recovery_paths=False,
    )
    if args.groups:
        df = df[df["group_norm"].isin(args.groups)]
        print(f"[INFO] Group filter {args.groups} -> N={len(df)} rows")
    if args.max_images is not None and len(df) > args.max_images:
        df = df.sample(args.max_images, random_state=42)
        print(f"[INFO] Subsampled to {len(df)} rows for t-SNE")

    tf = make_transform(img_size=args.img_size, train=False)
    ds = AgeImageDataset(df, tf)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_skip_none,
    )

    model = build_model(args).to(device)
    embeddings, meta = extract_embeddings(model, loader, device, max_images=args.max_images)
    if embeddings.size == 0:
        print("[WARN] No embeddings extracted; check dataset/filters.")
        return

    coords = run_tsne(embeddings, n_components=args.n_components, perplexity=args.perplexity)

    out_csv = args.output_csv if args.output_csv.is_absolute() else (OUTPUT_ROOT / args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_out = meta.reset_index(drop=True).copy()
    meta_out["tsne_1"] = coords[:, 0]
    if coords.shape[1] > 1:
        meta_out["tsne_2"] = coords[:, 1]
    meta_out.to_csv(out_csv, index=False)
    print(f"[SAVE] Saved t-SNE coordinates to {out_csv} (N={len(meta_out)})")

    out_plot = args.output_plot if args.output_plot.is_absolute() else (OUTPUT_ROOT / args.output_plot)
    plot_tsne(coords, meta_out, out_plot)


if __name__ == "__main__":
    main()
