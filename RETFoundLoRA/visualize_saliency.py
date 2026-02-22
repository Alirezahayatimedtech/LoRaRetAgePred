#!/usr/bin/env python3
"""
Generate a saliency (age attention) heatmap for a single image using a trained LoRA checkpoint.

Example:
  python visualize_saliency.py --image path/to/image.bmp
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from config import BACKBONE_CKPT, IMG_SIZE, OUTPUT_ROOT
from data_prep_age_lora import make_transform
from retfound_lora_age_pred import RETFoundLoRAAgePred


def load_model(backbone_ckpt: Path, lora_ckpt: Path, device: torch.device):
    model = RETFoundLoRAAgePred(
        ckpt_path=backbone_ckpt,
        img_size=IMG_SIZE,
        global_pool=False,
        lora_rank=16,      # match training config (r=16)
        lora_alpha=16.0,
        lora_blocks=8,
        lora_dropout=0.2,
        upsample_factor=2,
    ).to(device)
    ckpt = torch.load(lora_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "backbone_lora" in ckpt and "head" in ckpt:
        model.backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
        model.head.load_state_dict(ckpt["head"], strict=False)
    else:
        raise SystemExit(f"LoRA checkpoint missing expected keys: {lora_ckpt}")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="Visualize age saliency map for one image")
    ap.add_argument("--image", type=Path, required=True, help="Path to input image")
    ap.add_argument("--lora-ckpt", type=Path, default=OUTPUT_ROOT / "checkpoints/retfound_lora_age_weights.pt",
                    help="LoRA checkpoint with backbone_lora/head")
    ap.add_argument("--backbone-ckpt", type=Path, default=BACKBONE_CKPT,
                    help="RETFound MAE backbone checkpoint")
    ap.add_argument("--out", type=Path, default=OUTPUT_ROOT / "saliency.png", help="Output PNG path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.backbone_ckpt, args.lora_ckpt, device)

    tf = make_transform(img_size=IMG_SIZE, train=False)
    with Image.open(args.image).convert("RGB") as im:
        x = tf(im).unsqueeze(0).to(device)

    sal = model.get_age_saliency_maps(x)[0].detach().cpu().numpy()
    # Squeeze channel dim if present (expected shape HxW)
    if sal.ndim == 3 and sal.shape[0] == 1:
        sal = sal[0]

    # Normalize saliency 0-1 and overlay on the original image with jet colormap
    sal = sal - sal.min()
    if sal.max() > 0:
        sal = sal / sal.max()

    with Image.open(args.image).convert("RGB") as im:
        im = im.resize((IMG_SIZE, IMG_SIZE))

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(im)
    ax.imshow(sal, cmap="jet", alpha=0.45)
    ax.axis("off")
    plt.tight_layout(pad=0.05)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[SAL] Saved saliency map to {args.out}")


if __name__ == "__main__":
    main()
