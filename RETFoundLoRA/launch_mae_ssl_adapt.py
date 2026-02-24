"""Prepare and launch MAE-style self-supervised adaptation from an unlabeled manifest.

This repo currently does not include a MAE pretraining entry point (`main_pretrain.py`).
This launcher therefore focuses on two things:

1) Building an ImageFolder-compatible dataset (symlink/copy) from a manifest CSV.
2) Generating a reproducible `torchrun ... main_pretrain.py` command script and
   optionally executing it when an external pretrain script is provided.

Typical usage (prepare only, transductive SSL):
    python3 RETFoundLoRA/launch_mae_ssl_adapt.py \
      --manifest outputs/ssl_manifests/.../ssl_transductive_all_rats_unlabeled.csv

When you have an MAE pretrain repo:
    python3 RETFoundLoRA/launch_mae_ssl_adapt.py \
      --manifest outputs/ssl_manifests/.../ssl_transductive_all_rats_unlabeled.csv \
      --pretrain-script /path/to/mae/main_pretrain.py \
      --init-ckpt-arg --finetune \
      --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_manifest() -> Path:
    return PROJECT_ROOT / "outputs/ssl_manifests/c123_allages_controls_hls_strict_vs_transductive/ssl_transductive_all_rats_unlabeled.csv"


def _default_retfound_ckpt() -> Path:
    return PROJECT_ROOT / "RETFound_MAE_Model/RETFound_mae_natureOCT.pth"


def _safe_slug(txt: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(txt))
    return out.strip("._") or "run"


def _resolve_pretrain_script(path: Optional[Path]) -> Optional[Path]:
    if path is not None:
        return Path(path)
    local = PROJECT_ROOT / "RETFound_MAE" / "main_pretrain.py"
    return local if local.exists() else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare/launch MAE SSL adaptation from an unlabeled OCT manifest.")
    p.add_argument("--manifest", type=Path, default=_default_manifest())
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/ssl_adapt/mae_transductive_c123_allages",
        help="Working directory for prepared ImageFolder dataset, logs, and launch scripts.",
    )
    p.add_argument("--class-name", type=str, default="unlabeled")
    p.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking (slower, more disk).")
    p.add_argument(
        "--force-rgb-export",
        action="store_true",
        help="Materialize all images as RGB files (PNG by default) instead of symlinking/copying raw source files. Use this if the MAE repo expects 3-channel RGB and your OCT files may be grayscale.",
    )
    p.add_argument(
        "--rgb-export-format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output format used when --force-rgb-export is enabled (default: png).",
    )
    p.add_argument("--max-images", type=int, default=0, help="Optional cap for quick smoke checks (0 = all).")
    p.add_argument("--skip-missing", action="store_true", help="Skip missing image paths instead of failing.")
    p.add_argument("--run-name", type=str, default="rat_oct_ssl_mae_transductive")

    # MAE launch config (command generation)
    p.add_argument("--pretrain-script", type=Path, default=None, help="Path to external MAE `main_pretrain.py`.")
    p.add_argument(
        "--autodetect-script-args",
        action="store_true",
        default=True,
        help="Inspect `pretrain-script -h` and adapt common MAE CLI arg names automatically (default: on).",
    )
    p.add_argument(
        "--no-autodetect-script-args",
        action="store_false",
        dest="autodetect_script_args",
        help="Disable help-based arg autodetection and use the default Facebook-MAE-style arg names.",
    )
    p.add_argument("--python-bin", type=str, default="python3")
    p.add_argument("--torchrun-bin", type=str, default="torchrun")
    p.add_argument(
        "--launch-backend",
        type=str,
        choices=["torchrun", "python-module"],
        default="torchrun",
        help="How to launch distributed pretraining. `python-module` uses `<python-bin> -m torch.distributed.run` (useful for venvs without torchrun script).",
    )
    p.add_argument("--nproc-per-node", type=int, default=1)
    p.add_argument("--master-port", type=int, default=29641)
    p.add_argument("--model", type=str, default="mae_vit_large_patch16")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--accum-iter", type=int, default=1)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--blr", type=float, default=1.5e-4, help="Base learning rate used by many MAE scripts.")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=8)

    # RETFound initialization checkpoint (optional, script-specific arg)
    p.add_argument("--retfound-init-ckpt", type=Path, default=_default_retfound_ckpt())
    p.add_argument(
        "--init-ckpt-arg",
        type=str,
        default="",
        help="Optional arg name for initializing from RETFound weights in your MAE pretrain script (e.g., --finetune or --pretrained).",
    )
    p.add_argument(
        "--auto-init-ckpt-arg",
        action="store_true",
        help="If --init-ckpt-arg is not set, try to infer a checkpoint-init arg from pretrain-script help (e.g., --finetune/--pretrained).",
    )
    p.add_argument("--resume", type=Path, default=None, help="Optional MAE pretrain resume checkpoint.")
    p.add_argument("--extra-arg", action="append", default=[], help="Repeatable raw extra arg string (appended as one token each).")
    p.add_argument("--execute", action="store_true", help="Execute the generated command if pretrain script exists.")
    return p.parse_args()


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    if "image_path" not in df.columns:
        raise ValueError(f"Manifest must contain `image_path` column: {manifest_path}")
    # Keep first occurrence of each image path to avoid duplicate symlinks.
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return df


def _link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except FileExistsError:
        return
    except OSError:
        # Fallback in environments where symlinks are not allowed.
        shutil.copy2(src, dst)


def _export_rgb(src: Path, dst: Path, fmt: str) -> None:
    from PIL import Image  # lazy import
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    fmt_norm = str(fmt).upper()
    if fmt_norm == "JPG":
        fmt_norm = "JPEG"
    with Image.open(src) as im:
        rgb = im.convert("RGB")
        save_kwargs = {}
        if fmt_norm == "JPEG":
            save_kwargs.update({"quality": 95})
        rgb.save(dst, format=fmt_norm, **save_kwargs)


def _prepare_imagefolder(
    df: pd.DataFrame,
    out_dir: Path,
    class_name: str,
    copy_images: bool,
    max_images: int,
    skip_missing: bool,
    force_rgb_export: bool = False,
    rgb_export_format: str = "png",
) -> pd.DataFrame:
    imagefolder_root = out_dir / "imagefolder"
    class_dir = imagefolder_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    # MAE official pretrain script expects ImageNet-style "<data_path>/train/<class>/*".
    # Keep generic ImageFolder compatibility at root, and add a train/ alias for MAE.
    train_root = imagefolder_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    train_class_dir = train_root / class_name
    if not train_class_dir.exists():
        try:
            os.symlink(class_dir, train_class_dir, target_is_directory=True)
        except OSError:
            train_class_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_limit = int(max_images) if int(max_images or 0) > 0 else len(df)
    for i, row in df.head(n_limit).iterrows():
        src = Path(str(row["image_path"]))
        if not src.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"Missing image in manifest: {src}")
        if force_rgb_export:
            ext = "." + str(rgb_export_format).lower().lstrip(".")
        else:
            ext = src.suffix if src.suffix else ".png"
        stem = _safe_slug(src.stem)
        suffix_hash = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
        dst_name = f"{i:06d}_{stem}_{suffix_hash}{ext}"
        dst = class_dir / dst_name
        if force_rgb_export:
            _export_rgb(src, dst, fmt=str(rgb_export_format))
        else:
            _link_or_copy(src, dst, copy_images=copy_images)
        if train_class_dir.is_dir() and not train_class_dir.is_symlink():
            # Fallback path when directory symlink creation is unavailable.
            _link_or_copy(dst, train_class_dir / dst_name, copy_images=False)
        item = row.to_dict()
        item["ssl_imagefolder_path"] = str(dst)
        item["ssl_imagefolder_relpath"] = f"{class_name}/{dst_name}"
        rows.append(item)
    out_df = pd.DataFrame(rows)
    return out_df


def _probe_script_help(pretrain_script: Optional[Path], python_bin: str, enabled: bool) -> Dict[str, object]:
    if (not enabled) or pretrain_script is None or (not Path(pretrain_script).exists()):
        return {"options": set(), "raw": ""}
    try:
        res = subprocess.run(
            [python_bin, str(pretrain_script), "-h"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        text = (res.stdout or "") + "\n" + (res.stderr or "")
        opts = set(re.findall(r"(--[a-zA-Z0-9][a-zA-Z0-9_-]*)", text))
        return {"options": opts, "raw": text}
    except Exception as e:
        return {"options": set(), "raw": f"[help probe failed] {e}"}


def _pick_opt(help_opts: Sequence[str], *candidates: str) -> Optional[str]:
    opt_set = set(help_opts or [])
    for c in candidates:
        if c in opt_set:
            return c
    return candidates[0] if candidates else None


def _append_kv(cmd: List[str], opt: Optional[str], value: object) -> None:
    if not opt:
        return
    cmd.extend([str(opt), str(value)])


def _maybe_append_if_supported(cmd: List[str], help_opts: Sequence[str], candidates: Sequence[str], value: object) -> None:
    opt = _pick_opt(help_opts, *candidates)
    if opt is None:
        return
    if help_opts and opt not in set(help_opts):
        return
    _append_kv(cmd, opt, value)


def _infer_init_ckpt_arg(args: argparse.Namespace, help_opts: Sequence[str]) -> str:
    init_arg = str(args.init_ckpt_arg or "").strip()
    if init_arg:
        return init_arg
    if not bool(getattr(args, "auto_init_ckpt_arg", False)):
        return ""
    for cand in ("--finetune", "--pretrained", "--pretrained_ckpt", "--pretrained-ckpt", "--init_ckpt", "--init-ckpt"):
        if cand in set(help_opts or []):
            return cand
    return ""


def _build_mae_command(args: argparse.Namespace, imagefolder_root: Path, work_dir: Path, help_probe: Optional[Dict[str, object]] = None) -> List[str]:
    pretrain_script = _resolve_pretrain_script(args.pretrain_script)
    help_opts = list((help_probe or {}).get("options", []) or [])
    if str(getattr(args, "launch_backend", "torchrun")) == "python-module":
        cmd = [
            str(args.python_bin),
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={int(args.nproc_per_node)}",
            f"--master_port={int(args.master_port)}",
        ]
    else:
        cmd = [
            args.torchrun_bin,
            f"--nproc_per_node={int(args.nproc_per_node)}",
            f"--master_port={int(args.master_port)}",
        ]
    if pretrain_script is None:
        # Placeholder path for scripting clarity; execution will be blocked.
        cmd.append("main_pretrain.py")
    else:
        cmd.append(str(pretrain_script))

    run_dir = work_dir / "mae_run"
    output_dir = run_dir / "output"
    log_dir = run_dir / "logs"

    _append_kv(cmd, _pick_opt(help_opts, "--model"), str(args.model))
    _append_kv(cmd, _pick_opt(help_opts, "--input_size", "--input-size"), int(args.input_size))
    _append_kv(cmd, _pick_opt(help_opts, "--batch_size", "--batch-size"), int(args.batch_size))
    _append_kv(cmd, _pick_opt(help_opts, "--epochs"), int(args.epochs))
    _append_kv(cmd, _pick_opt(help_opts, "--accum_iter", "--accum-iter"), int(args.accum_iter))
    _append_kv(cmd, _pick_opt(help_opts, "--mask_ratio", "--mask-ratio"), float(args.mask_ratio))

    # Learning rate flag names vary by MAE repo. Prefer --blr when available, else --lr.
    lr_opt = _pick_opt(help_opts, "--blr", "--base-lr", "--lr")
    _append_kv(cmd, lr_opt, float(args.blr))
    _append_kv(cmd, _pick_opt(help_opts, "--weight_decay", "--weight-decay"), float(args.weight_decay))
    _append_kv(cmd, _pick_opt(help_opts, "--num_workers", "--num-workers", "--workers"), int(args.num_workers))
    _append_kv(cmd, _pick_opt(help_opts, "--data_path", "--data-path", "--dataset_path", "--dataset-path"), str(imagefolder_root))
    _append_kv(cmd, _pick_opt(help_opts, "--output_dir", "--output-dir"), str(output_dir))
    # log_dir is optional in some repos; only pass it if supported (or if help was not available).
    _maybe_append_if_supported(cmd, help_opts, ["--log_dir", "--log-dir", "--tensorboard_dir", "--tensorboard-dir"], str(log_dir))

    if args.resume:
        _append_kv(cmd, _pick_opt(help_opts, "--resume"), str(args.resume))

    init_arg = _infer_init_ckpt_arg(args, help_opts)
    if init_arg:
        cmd.extend([init_arg, str(args.retfound_init_ckpt)])

    for tok in args.extra_arg:
        if tok is None:
            continue
        t = str(tok).strip()
        if t:
            cmd.append(t)
    return cmd


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else (PROJECT_ROOT / args.manifest)
    out_dir = args.out_dir if args.out_dir.is_absolute() else (PROJECT_ROOT / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_manifest(manifest_path)
    prepared_df = _prepare_imagefolder(
        df=df,
        out_dir=out_dir,
        class_name=str(args.class_name),
        copy_images=bool(args.copy_images),
        max_images=int(args.max_images or 0),
        skip_missing=bool(args.skip_missing),
        force_rgb_export=bool(args.force_rgb_export),
        rgb_export_format=str(args.rgb_export_format),
    )
    if prepared_df.empty:
        raise RuntimeError("Prepared SSL ImageFolder is empty.")

    # Persist a clean prepared manifest (contains original metadata + linked path)
    prepared_manifest = out_dir / "prepared_ssl_manifest.csv"
    prepared_df.to_csv(prepared_manifest, index=False)

    imagefolder_root = out_dir / "imagefolder"
    pretrain_script = _resolve_pretrain_script(args.pretrain_script)
    help_probe = _probe_script_help(pretrain_script, python_bin=str(args.python_bin), enabled=bool(args.autodetect_script_args))
    cmd = _build_mae_command(args, imagefolder_root=imagefolder_root, work_dir=out_dir, help_probe=help_probe)

    command_sh = out_dir / "launch_mae_ssl_adapt.sh"
    command_txt = " ".join(subprocess.list2cmdline([c]) if " " in str(c) else str(c) for c in cmd)
    command_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + command_txt + "\n")
    try:
        command_sh.chmod(0o755)
    except Exception:
        pass

    summary = {
        "manifest": str(manifest_path),
        "prepared_manifest": str(prepared_manifest),
        "imagefolder_root": str(imagefolder_root),
        "n_manifest_rows": int(len(df)),
        "n_prepared_rows": int(len(prepared_df)),
        "copy_images": bool(args.copy_images),
        "force_rgb_export": bool(args.force_rgb_export),
        "rgb_export_format": str(args.rgb_export_format),
        "max_images": int(args.max_images or 0),
        "run_name": str(args.run_name),
        "launch_backend": str(args.launch_backend),
        "pretrain_script": str(pretrain_script) if pretrain_script is not None else None,
        "autodetect_script_args": bool(args.autodetect_script_args),
        "script_help_options_detected": sorted(list(help_probe.get("options", set())))[:200],
        "retfound_init_ckpt": str(args.retfound_init_ckpt),
        "init_ckpt_arg": str(args.init_ckpt_arg or ""),
        "resolved_init_ckpt_arg": _infer_init_ckpt_arg(args, list(help_probe.get("options", []))),
        "command_script": str(command_sh),
        "command_tokens": [str(c) for c in cmd],
    }
    (out_dir / "launcher_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[SSL-MAE] Prepared ImageFolder root: {imagefolder_root}")
    print(f"[SSL-MAE] Prepared manifest rows: {len(prepared_df)} (from input {len(df)})")
    print(f"[SSL-MAE] Prepared manifest: {prepared_manifest}")
    print(f"[SSL-MAE] Launch script: {command_sh}")

    if pretrain_script is None:
        print("[SSL-MAE][WARN] No main_pretrain.py found in this repo. Provide --pretrain-script to execute MAE adaptation.")
        return
    if help_probe.get("raw"):
        (out_dir / "pretrain_script_help.txt").write_text(str(help_probe["raw"]))

    if not args.execute:
        print("[SSL-MAE] Pretrain script detected, but --execute was not set. Command prepared only.")
        return

    # Execute the generated command.
    print("[SSL-MAE] Executing MAE adaptation command...")
    subprocess.run(cmd, check=True, cwd=str(pretrain_script.parent))


if __name__ == "__main__":
    main()
