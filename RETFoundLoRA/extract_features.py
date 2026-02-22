#!/usr/bin/env python3
"""
extract_features.py

One-time feature extraction using RETFound encoder. Aggregates multiple images per rat/day
into a single feature vector and saves .npy files plus a manifest CSV. This avoids re-running
the heavy encoder during regression training/inference.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_ROOT = REPO_ROOT / "LoraRETfoundageReed"
for p in (REPO_ROOT, NEW_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import (
    CSV_PATH,
    BACKBONE_CKPT,
    IMG_SIZE,
    IMAGE_TYPES,
    DAY_WHITELIST,
    COHORTS_TO_KEEP,
    TRAIN_GROUPS,
    TEST_GROUPS,
    BATCH_SIZE,
    NUM_WORKERS,
    OUTPUT_ROOT,
)
from data_prep_age_lora import load_metadata, make_transform
from retfound_lora_age_pred import load_retfound_backbone_with_lora
from utils import normalize_eye_side


def parse_args():
    p = argparse.ArgumentParser(description="Extract RETFound features and aggregate per rat/day")
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    p.add_argument("--backbone-ckpt", type=Path, default=BACKBONE_CKPT)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "features")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-groups", type=str, nargs="*", default=TRAIN_GROUPS)
    p.add_argument("--test-groups", type=str, nargs="*", default=TEST_GROUPS)
    return p.parse_args()


@torch.no_grad()
def extract_spatial_features(backbone, x: torch.Tensor) -> torch.Tensor:
    # Patch embedding
    x = backbone.patch_embed(x)

    # Add cls token
    cls_token = getattr(backbone, 'cls_token', None)
    if cls_token is not None:
        cls_tok = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tok, x), dim=1)

    # Position embedding
    if hasattr(backbone, 'pos_embed') and backbone.pos_embed is not None:
        x = x + backbone.pos_embed

    # Blocks
    for blk in backbone.blocks:
        x = blk(x)

    # Norm
    if hasattr(backbone, 'norm'):
        x = backbone.norm(x)

    # Remove cls, reshape tokens to spatial map
    tokens = x[:, 1:, :]  # (B, N, C)
    B, N, C = tokens.shape
    H = W = int(N ** 0.5)
    tokens = tokens.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
    return tokens


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"[DEVICE] requested={args.device} | available_cuda={torch.cuda.is_available()} | using={device}")

    df = load_metadata(
        csv_path=args.csv,
        image_types=IMAGE_TYPES,
        day_whitelist=DAY_WHITELIST,
        include_recovery_days=False,
        cohorts_to_keep=COHORTS_TO_KEEP,
        exclude_recovery_paths=False,
    )

    # Ignore metadata eye field; infer from material_type or path
    df["eye"] = df.apply(lambda r: normalize_eye_side(r.get("eye"), r.get("image_path", ""), r.get("material_type", "")), axis=1)

    keep_groups = set(args.train_groups or []) | set(args.test_groups or [])
    if keep_groups:
        df = df[df["group_norm"].isin(keep_groups)]
        print(f"[INFO] Kept groups: {sorted(keep_groups)} -> N={len(df)}")

    # Load backbone (no LoRA)
    backbone = load_retfound_backbone_with_lora(
        ckpt_path=args.backbone_ckpt,
        img_size=args.img_size,
        global_pool=False,
        enable_lora=False,  # allow LoRA adapters (previous setup)
    ).to(device)
    backbone.eval()

    tf = make_transform(img_size=args.img_size, train=False)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    grouped = df.groupby(["rat_id", "eye", "day", "group_norm"], dropna=False)
    for (rat_id, eye, day, group), g in tqdm(grouped, desc="Groups"):
        paths = g["image_path"].tolist()
        feats_list = []
        # process in mini-batches
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start:start + args.batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    with Image.open(p).convert("RGB") as im:
                        # Canonicalize OS to OD orientation to reduce inter-eye variance
                        if str(eye).strip().upper() == "OS":
                            im = ImageOps.mirror(im)
                        imgs.append(tf(im))
                except Exception:
                    continue
            if not imgs:
                continue
            xb = torch.stack(imgs, 0).to(device, non_blocking=True)
            feats = extract_spatial_features(backbone, xb)  # (B, C)
            feats_list.append(feats.detach().cpu().numpy())

        if not feats_list:
            continue
        all_feats = np.concatenate(feats_list, axis=0)
        eye_tag = eye if isinstance(eye, str) and eye else "Unknown"
        fname = f"rat_{rat_id}_eye_{eye_tag}_day_{int(day)}.npy"
        fpath = out_dir / fname
        np.save(fpath, all_feats)  # save full stack of features (N, C, H, W)

        records.append({
            "rat_id": rat_id,
            "eye": eye,
            "day": int(day),
            "group": group,
            "n_images": len(paths),
            "feature_path": str(fpath),
        })

    manifest = pd.DataFrame(records)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[DONE] Saved {len(records)} feature vectors to {out_dir} and manifest to {manifest_path}")


if __name__ == "__main__":
    main()
