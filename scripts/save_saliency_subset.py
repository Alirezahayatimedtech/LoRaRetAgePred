#!/usr/bin/env python3
"""
Save saliency/attention overlays for a filtered subset (e.g., day 90, cohort 1, Controls vs HLS).
Includes:
 - Higher-resolution head upsample (factor 4) for finer maps.
 - Full colormap overlay (not just top-5% mask).
 - De-padding to original aspect ratio before overlaying on the raw OCT.

Example:
  python scripts/save_saliency_subset.py \
    --csv metadata/image_age_mapping.csv \
    --cohorts 1 \
    --days 90 \
    --groups Controls "HLS (U)" \
    --lora-ckpt outputs/checkpoints/retfound_lora_age_weights_fold2.pt \
    --out-dir outputs/saliency/day90_cohort1
"""

import argparse
from pathlib import Path
import sys
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib import cm

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "RETFoundLoRA"):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from config import BACKBONE_CKPT, IMAGE_TYPES, IMG_SIZE  # type: ignore
from data_prep_age_lora import load_metadata  # type: ignore
from retfound_lora_age_pred import RETFoundLoRAAgePred  # type: ignore
from simple_baseline import SimpleXceptionAgePred  # type: ignore
from utils import normalize_eye_side  # type: ignore


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SaliencyDataset(Dataset):
    def __init__(self, df, img_size: int = IMG_SIZE, right_eye_only: bool = False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.right_eye_only = right_eye_only
        self.normalize = T.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["image_path"])
        try:
            with Image.open(path).convert("RGB") as im:
                orig_w, orig_h = im.size
                max_side = max(orig_w, orig_h)
                pad_left = (max_side - orig_w) // 2
                pad_top = (max_side - orig_h) // 2
                pad_right = max_side - orig_w - pad_left
                pad_bottom = max_side - orig_h - pad_top
                padded = ImageOps.expand(im, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
                resized = padded.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
                x = T.ToTensor()(resized)
                x = self.normalize(x)
        except Exception as e:
            print(f"[WARN] Skipping unreadable image for saliency: {path} ({e})")
            return None

        sample = {
            "image": x,
            "path": str(path),
            "rat_id": row.get("rat_id", ""),
            "day": float(row.get("day", np.nan)),
            "group": row.get("group_norm", row.get("group", "Unknown")),
            "sex": row.get("sex", "Unknown"),
            "cohort": row.get("cohort", "Unknown"),
            "eye": row.get("eye", "Unknown"),
            "orig_size": (orig_w, orig_h),
            "pad": (pad_left, pad_top, pad_right, pad_bottom),
            "max_side": max_side,
        }
        return sample


def collate_batch(batch: List[dict]):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs = torch.stack([b["image"] for b in batch])
    meta = {}
    for k in batch[0]:
        if k == "image":
            continue
        meta[k] = [b[k] for b in batch]
    meta["image"] = imgs
    return meta


def load_model(
    model_type: str,
    backbone_ckpt: Path,
    lora_ckpt: Path,
    device: torch.device,
    img_size: int,
    lora_rank: int,
    lora_alpha: float,
    lora_blocks: int,
    lora_dropout: float,
):
    model_type = str(model_type).lower()
    if model_type == "retfound":
        model = RETFoundLoRAAgePred(
            ckpt_path=backbone_ckpt,
            img_size=img_size,
            global_pool=False,
            lora_rank=int(lora_rank),
            lora_alpha=float(lora_alpha),
            lora_blocks=int(lora_blocks),
            lora_dropout=float(lora_dropout),
            upsample_factor=4,  # higher native saliency resolution
        ).to(device)
    elif model_type == "xception":
        model = SimpleXceptionAgePred(pretrained=False).to(device)
    else:
        raise SystemExit(f"Unsupported --model-type: {model_type}")
    ckpt = torch.load(lora_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "backbone_lora" in ckpt and "head" in ckpt:
        model.backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
        model.head.load_state_dict(ckpt["head"], strict=False)
    else:
        raise SystemExit(f"LoRA checkpoint missing expected keys: {lora_ckpt}")
    model.eval()
    return model


def _normalize_saliency_batch(sal: torch.Tensor) -> torch.Tensor:
    """Normalize saliency per image to [0, 1] over spatial dims."""
    if sal.ndim != 4:
        raise ValueError(f"Expected saliency tensor [B,C,H,W], got {tuple(sal.shape)}")
    B, C, H, W = sal.shape
    flat = sal.view(B, C, -1)
    mins = flat.min(dim=2, keepdim=True).values.view(B, C, 1, 1)
    maxs = flat.max(dim=2, keepdim=True).values.view(B, C, 1, 1)
    return (sal - mins) / (maxs - mins + 1e-8)


def compute_saliency_maps(model, imgs: torch.Tensor, mode: str = "auto") -> torch.Tensor:
    """
    Return saliency maps [B,1,H,W].

    - spatial: use model.get_age_saliency_maps (requires keep_spatial_tokens=True)
    - grad: input-gradient saliency wrt scalar age prediction
    - auto: spatial when available, else grad
    """
    mode = str(mode).lower()
    keep_spatial = bool(getattr(model, "keep_spatial_tokens", False))
    use_spatial = (mode == "spatial") or (mode == "auto" and keep_spatial)
    if use_spatial:
        try:
            return model.get_age_saliency_maps(imgs)
        except RuntimeError as e:
            if mode == "spatial":
                raise
            print(f"[SAL] Spatial saliency unavailable; falling back to grad mode ({e})")

    # Gradient-based saliency fallback for CLS-only models.
    was_training = model.training
    model.eval()
    x = imgs.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    preds, _ = model(x)
    # Sum scalar predictions to get one backward pass for the batch.
    preds.view(-1).sum().backward()
    grad = x.grad.detach().abs()
    sal = grad.mean(dim=1, keepdim=True)  # [B,1,H,W]
    sal = _normalize_saliency_batch(sal)
    if was_training:
        model.train()
    return sal


def unnormalize(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().permute(1, 2, 0).numpy()
    x = x * np.array(STD).reshape(1, 1, 3) + np.array(MEAN).reshape(1, 1, 3)
    return np.clip(x, 0, 1)


def crop_to_original(arr: np.ndarray, orig_size: Tuple[int, int], pad: Tuple[int, int, int, int], max_side: int, img_size: int) -> np.ndarray:
    orig_w, orig_h = int(orig_size[0]), int(orig_size[1])
    pad_left, pad_top, pad_right, pad_bottom = [int(x) for x in pad[:4]]
    scale = img_size / float(max_side)
    l = int(round(pad_left * scale))
    t = int(round(pad_top * scale))
    r = img_size - int(round(pad_right * scale))
    b = img_size - int(round(pad_bottom * scale))
    cropped = arr[t:b, l:r]
    if cropped.size == 0:
        return arr
    pil = Image.fromarray((cropped * 255).astype("uint8"))
    pil = pil.resize((orig_w, orig_h), resample=Image.BILINEAR)
    return np.asarray(pil).astype(np.float32) / 255.0


def save_overlay(
    sample,
    sal_map: np.ndarray,
    out_dir: Path,
    alpha: float = 0.25,
    clip: Optional[Tuple[float, float]] = (2.0, 98.0),
    smooth_sigma: Optional[float] = 1.0,
    overlay_resized: bool = False,
    img_size: int = IMG_SIZE,
    side_by_side: bool = True,
    cmap_name: str = "jet",
    raw_panel: bool = False,
):
    # sal_map shape (H, W) normalized 0-1 (before clipping)
    if clip is not None:
        lo, hi = np.percentile(sal_map, clip)
        if hi > lo:
            sal_map = (sal_map - lo) / (hi - lo)
        sal_map = np.clip(sal_map, 0, 1)
    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            sal_map = gaussian_filter(sal_map, sigma=smooth_sigma)
        except Exception:
            pass

    cmap = cm.get_cmap(cmap_name)

    if overlay_resized:
        # overlay on the padded/resized square view
        with Image.open(sample["path"]).convert("RGB") as im:
            orig_w, orig_h = im.size
            max_side = max(orig_w, orig_h)
            pad_left = (max_side - orig_w) // 2
            pad_top = (max_side - orig_h) // 2
            pad_right = max_side - orig_w - pad_left
            pad_bottom = max_side - orig_h - pad_top
            padded = ImageOps.expand(im, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
            base = np.asarray(padded.resize((img_size, img_size), resample=Image.BILINEAR)).astype(np.float32) / 255.0
        sal_resized = sal_map
        if sal_resized.shape[:2] != base.shape[:2]:
            sal_resized = np.array(
                Image.fromarray((sal_resized * 255).astype("uint8")).resize(base.shape[1::-1], resample=Image.BILINEAR)
            ).astype(np.float32) / 255.0
        heat = cmap(sal_resized)[..., :3]
        overlay_img = np.clip(base * (1 - alpha) + heat * alpha, 0, 1)
        orig_view = base  # for side-by-side in padded space
    else:
        # overlay on original aspect
        with Image.open(sample["path"]).convert("RGB") as im_orig:
            base_orig = np.asarray(im_orig).astype(np.float32) / 255.0
            orig_view = base_orig.copy()
        sal_cropped = crop_to_original(sal_map, sample["orig_size"], sample["pad"], int(sample["max_side"]), img_size)
        if sal_cropped.shape[:2] != base_orig.shape[:2]:
            sal_cropped = np.array(
                Image.fromarray((sal_cropped * 255).astype("uint8")).resize(base_orig.shape[1::-1], resample=Image.BILINEAR)
            ).astype(np.float32) / 255.0
        heat = cmap(sal_cropped)[..., :3]
        overlay_img = np.clip(base_orig * (1 - alpha) + heat * alpha, 0, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{sample['rat_id']}_{sample['eye']}_{int(sample['day'])}_{sample['group']}_{Path(sample['path']).stem}.png"
    if side_by_side:
        # pad to same height and concat horizontally (orig | overlay | raw heatmap optional)
        h1, _, _ = overlay_img.shape
        h2, _, _ = orig_view.shape
        panels = [orig_view, overlay_img]
        if raw_panel:
            heat_rgb = cmap(np.clip(sal_map, 0, 1))[..., :3]
            panels.append(heat_rgb)
        H = max([p.shape[0] for p in panels])

        def pad_to(img, H):
            h, w, _ = img.shape
            pad_top = (H - h) // 2
            pad_bottom = H - h - pad_top
            return np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="constant", constant_values=0)

        padded = [pad_to(p, H) for p in panels]
        combined = np.concatenate(padded, axis=1)
        Image.fromarray((combined * 255).astype("uint8")).save(out_dir / fname)
    else:
        Image.fromarray((overlay_img * 255).astype("uint8")).save(out_dir / fname)


def parse_args():
    p = argparse.ArgumentParser(description="Save saliency overlays for filtered subset")
    p.add_argument("--model-type", type=str, default="retfound", choices=["retfound", "xception"])
    p.add_argument("--csv", type=Path, default=Path("metadata/image_age_mapping.csv"))
    p.add_argument("--cohorts", type=str, nargs="*", default=["1"])
    p.add_argument("--days", type=int, nargs="*", default=[90])
    p.add_argument("--groups", type=str, nargs="*", default=["Controls", "HLS (U)"])
    p.add_argument("--lora-ckpt", type=Path, required=True)
    p.add_argument("--backbone-ckpt", type=Path, default=BACKBONE_CKPT)
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA rank used in the checkpoint")
    p.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha used in the checkpoint")
    p.add_argument("--lora-blocks", type=int, default=4, help="Number of LoRA-adapted transformer blocks in the checkpoint")
    p.add_argument("--lora-dropout", type=float, default=0.2, help="LoRA dropout used in the checkpoint")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--img-size", type=int, default=IMG_SIZE, help="Resize for saliency (use 384/448 for sharper maps if VRAM allows)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/saliency_subset"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--right-eye-only", action="store_true")
    p.add_argument("--alpha", type=float, default=0.25, help="Overlay alpha")
    p.add_argument("--clip", type=float, nargs=2, default=(2.0, 98.0), help="Percentile clip for saliency")
    p.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian sigma for saliency smoothing (0 to disable)")
    p.add_argument("--overlay-resized", action="store_true", help="Overlay on padded/resized square view instead of original aspect")
    p.add_argument("--side-by-side", action="store_true", help="Save original + overlay concatenated horizontally")
    p.add_argument("--raw-panel", action="store_true", help="When side-by-side, append a raw saliency heatmap panel")
    p.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name for overlay (e.g., viridis, jet, inferno, magma)")
    p.add_argument("--saliency-mode", type=str, choices=["auto", "spatial", "grad"], default="auto",
                   help="Saliency extraction mode. auto=spatial if available, else gradient input saliency.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    df = load_metadata(
        csv_path=args.csv,
        image_types=IMAGE_TYPES,
        day_whitelist=args.days,
        include_recovery_days=False,
        cohorts_to_keep=args.cohorts,
        exclude_recovery_paths=False,
    )
    df["eye"] = df.apply(lambda r: normalize_eye_side(r.get("eye"), r.get("image_path", ""), r.get("material_type", "")), axis=1)
    if args.right_eye_only:
        df = df[df["eye"].str.upper() == "OD"]
    df = df[df["group_norm"].isin(args.groups)]
    if df.empty:
        raise SystemExit("No rows after filtering.")

    ds = SaliencyDataset(df, img_size=args.img_size, right_eye_only=args.right_eye_only)
    pin = torch.cuda.is_available()
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin, collate_fn=collate_batch)

    model = load_model(
        args.model_type,
        args.backbone_ckpt,
        args.lora_ckpt,
        device,
        img_size=args.img_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_blocks=args.lora_blocks,
        lora_dropout=args.lora_dropout,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    warned_grad = False
    for batch in loader:
        if batch is None:
            continue
        imgs = batch["image"].to(device, non_blocking=True)
        if args.saliency_mode in ("grad", "auto") and not bool(getattr(model, "keep_spatial_tokens", False)) and not warned_grad:
            print("[SAL] Using gradient-based saliency fallback (CLS-only checkpoint).")
            warned_grad = True
        sal = compute_saliency_maps(model, imgs, mode=args.saliency_mode)
        sal = sal.detach().cpu().numpy()
        if sal.ndim == 4 and sal.shape[1] > 1:
            sal = sal[:, 0]  # take first channel
        elif sal.ndim == 4:
            sal = sal[:, 0]
        B = sal.shape[0]
        for i in range(B):
            sample = {k: v[i] for k, v in batch.items() if k != "image"}
            # normalize types
            osz = sample.get("orig_size")
            if osz is not None and len(osz) >= 2:
                sample["orig_size"] = (int(osz[0]), int(osz[1]))
            padv = sample.get("pad")
            if padv is not None and len(padv) >= 4:
                sample["pad"] = tuple(int(x) for x in padv[:4])
            sample["max_side"] = int(sample.get("max_side", IMG_SIZE))
            save_overlay(
                sample,
                sal[i],
                out_dir,
                alpha=args.alpha,
                clip=tuple(args.clip) if args.clip else None,
                smooth_sigma=args.smooth_sigma,
                overlay_resized=args.overlay_resized,
                img_size=args.img_size,
                side_by_side=args.side_by_side,
                cmap_name=args.cmap,
                raw_panel=args.raw_panel,
            )

    print(f"[DONE] Saliency overlays saved to {out_dir}")


if __name__ == "__main__":
    main()
