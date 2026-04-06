#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "paper1" / "control_best_worst_magma_xception"
MANIFEST = BASE / "selected_sample_manifest.csv"
IMAGES_META = BASE / "selected_images_metadata.csv"
PANELS_DIR = BASE / "paper_panels"
SALIENCY_DIR = BASE / "saliency_magma_allimages"
REVIEW_DIR = BASE / "review_by_sample"
RESTORED_DIR = BASE / "restored_case_assets"
README = BASE / "RESTORED_README.txt"


def safe_day(day: float | int | str) -> str:
    return str(int(float(day)))


def sample_dirname(rank: int, rat_id: str, eye: str, day: float | int | str) -> str:
    return f"rank{int(rank):02d}_{rat_id}_{eye}_day{safe_day(day)}"


def sample_key(row: pd.Series) -> tuple[str, str, int]:
    return str(row["rat_id"]), str(row["eye"]), int(float(row["day"]))


def render_png(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        rgb.save(dst)


def side_by_side(original: Path, overlay: Path, dst: Path, title_left: str, title_right: str) -> None:
    with Image.open(original) as a, Image.open(overlay) as b:
        a = a.convert("RGB")
        b = b.convert("RGB")
        w = max(a.width, b.width)
        a = ImageOps.pad(a, (w, a.height), color=(0, 0, 0))
        b = ImageOps.pad(b, (w, b.height), color=(0, 0, 0))
        banner_h = 26
        out = Image.new("RGB", (w * 2, max(a.height, b.height) + banner_h), color=(15, 15, 15))
        draw = ImageDraw.Draw(out)
        draw.text((10, 6), title_left, fill=(255, 255, 255))
        draw.text((w + 10, 6), title_right, fill=(255, 255, 255))
        out.paste(a, (0, banner_h))
        out.paste(b, (w, banner_h))
        out.save(dst)


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    images = pd.read_csv(IMAGES_META)
    images["day_int"] = images["day"].astype(float).astype(int)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    RESTORED_DIR.mkdir(parents=True, exist_ok=True)

    for _, sample in manifest.iterrows():
        bucket = str(sample["bucket"])
        rank = int(sample["rank"])
        rat_id = str(sample["rat_id"])
        eye = str(sample["eye"])
        day = int(float(sample["day"]))
        dirname = sample_dirname(rank, rat_id, eye, day)

        bucket_dir = REVIEW_DIR / f"{bucket}5" / dirname
        bucket_dir.mkdir(parents=True, exist_ok=True)
        restored_dir = RESTORED_DIR / f"{bucket}5" / dirname
        restored_dir.mkdir(parents=True, exist_ok=True)

        originals_dir = bucket_dir / "original_images"
        overlays_dir = bucket_dir / "magma_overlays"
        pairs_dir = bucket_dir / "side_by_side"
        for d in [originals_dir, overlays_dir, pairs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Rebuild summary CSV from manifest row.
        summary_path = bucket_dir / "sample_summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample.index))
            writer.writeheader()
            writer.writerow(sample.to_dict())

        # Copy paper panel if available.
        panel_glob = f"{bucket}_rank{rank:02d}_{rat_id}_{eye}_day{day}_mae*.png"
        for panel_path in PANELS_DIR.glob(panel_glob):
            shutil.copy2(panel_path, bucket_dir / "panel.png")
            shutil.copy2(panel_path, restored_dir / panel_path.name)

        sample_images = images[
            (images["rat_id"].astype(str) == rat_id)
            & (images["eye"].astype(str) == eye)
            & (images["day_int"] == day)
        ].copy()
        if sample_images.empty:
            continue

        overlay_prefix = f"{rat_id}_{eye}_{day}_"
        overlay_matches = sorted(SALIENCY_DIR.glob(f"{overlay_prefix}*.png"))
        overlay_by_suffix = {}
        for overlay in overlay_matches:
            suffix = overlay.name.split("_Controls_", 1)[-1]
            overlay_by_suffix[suffix] = overlay

        for idx, (_, img_row) in enumerate(sample_images.iterrows(), start=1):
            src = Path(str(img_row["image_path"]))
            if not src.exists():
                continue
            stem = src.stem
            original_png = originals_dir / f"{idx:02d}_{stem}.png"
            render_png(src, original_png)

            overlay = overlay_by_suffix.get(f"{stem}.png")
            if overlay is not None and overlay.exists():
                overlay_dst = overlays_dir / overlay.name
                shutil.copy2(overlay, overlay_dst)
                pair_dst = pairs_dir / f"{idx:02d}_{stem}_original_vs_magma.png"
                side_by_side(original_png, overlay_dst, pair_dst, "Original", "Magma overlay")

        # Mirror the rebuilt folder into a simpler restored tree.
        for item in bucket_dir.iterdir():
            target = restored_dir / item.name
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)

    README.write_text(
        "Restored best/worst sample assets.\n"
        "\n"
        "Primary folders:\n"
        f"- {REVIEW_DIR}\n"
        f"- {RESTORED_DIR}\n"
        "\n"
        "Each sample folder contains:\n"
        "- sample_summary.csv\n"
        "- panel.png when available\n"
        "- original_images/\n"
        "- magma_overlays/\n"
        "- side_by_side/\n",
        encoding="utf-8",
    )
    print(REVIEW_DIR)
    print(RESTORED_DIR)
    print(README)


if __name__ == "__main__":
    main()
