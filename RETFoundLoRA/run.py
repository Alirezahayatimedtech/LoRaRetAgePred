#!/usr/bin/env python3
"""
Quick training/eval runner for RETFound + LoRA age regression.
Uses preprocess_age_lora.py for metadata filtering and dataloaders.
"""

import argparse
import sys
from pathlib import Path
import copy
import json
import shutil

import pandas as pd
import torch
import numpy as np
import loralib as lora
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from scipy.stats import pearsonr, spearmanr

MIN_CALIB_SAMPLES = 20  # guardrail to avoid fitting corrections on tiny val splits

# Make repo root and module dir importable for data prep helpers
LORA_DIR = Path(__file__).resolve().parent
REPO_ROOT = LORA_DIR.parents[0]
for path in (REPO_ROOT, LORA_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

def apply_suffix(path_obj: Path, suffix: str) -> Path:
    """Return a new Path with suffix inserted before extension."""
    return path_obj.with_name(path_obj.stem + suffix + path_obj.suffix)


def apply_dir_suffix(path_obj: Path, suffix: str) -> Path:
    """Append a suffix to a directory name."""
    return path_obj.with_name(path_obj.name + suffix)


def cleanup_outputs(pred_suffix: str, args):
    """Remove stale outputs (CSVs/dirs) that would collide with this run."""
    pred_dir = args.pred_csv.parent if args.pred_csv else (OUTPUT_ROOT / "predictions")
    targets = [
        pred_dir / f"control_test_results{pred_suffix}.csv",
        pred_dir / f"rag_experimental_results{pred_suffix}.csv",
        args.pred_csv if args.pred_csv else None,
        args.save_val_preds if args.save_val_preds else None,
    ]
    for p in targets:
        if p and p.exists() and p.is_file():
            try:
                p.unlink()
                print(f"[CLEANUP] Removed old file: {p}")
            except Exception as e:
                print(f"[CLEANUP] Failed to remove {p}: {e}")

    if args.save_saliency_dir:
        try:
            if args.save_saliency_dir.exists():
                shutil.rmtree(args.save_saliency_dir)
                print(f"[CLEANUP] Removed old saliency dir: {args.save_saliency_dir}")
            args.save_saliency_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[CLEANUP] Failed to clean saliency dir {args.save_saliency_dir}: {e}")

def compute_metrics_csv(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "age_true" not in df or "age_pred" not in df:
        return None
    rag = df["age_pred"] - df["age_true"]
    try:
        r, rp = pearsonr(df["age_true"], df["age_pred"])
        sr, srp = spearmanr(df["age_true"], df["age_pred"])
        adc, adcp = pearsonr(df["age_true"], rag)
    except Exception:
        r = sr = float("nan")
        rp = srp = float("nan")
        adc = adcp = float("nan")
    mae = float(np.mean(np.abs(df["age_true"] - df["age_pred"])))
    rmse = float(np.sqrt(np.mean((df["age_true"] - df["age_pred"]) ** 2)))
    ss_res = float(np.sum((df["age_true"] - df["age_pred"]) ** 2))
    ss_tot = float(np.sum((df["age_true"] - df["age_true"].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "file": str(path),
        "n_rows": int(len(df)),
        "mae": mae,
        "rmse": rmse,
        "pearson_r": float(r),
        "pearson_p": float(rp),
        "spearman_r": float(sr),
        "spearman_p": float(srp),
        "adc": float(adc),
        "adc_p": float(adcp),
        "r2": float(r2),
    }


def average_corrections(corrections):
    """Average a list of bias-correction tuples returned by run_fold."""
    corr_list = [c for c in corrections if c]
    if not corr_list:
        return None
    modes = {c[0] for c in corr_list}
    if len(modes) != 1:
        print(f"[CV] Mixed correction modes across folds: {modes}; skipping averaging.")
        return None
    mode = corr_list[0][0]
    accum = {}
    for _, cdict in corr_list:
        for key, coeffs in cdict.items():
            accum.setdefault(key, []).append(np.asarray(coeffs, dtype=float))
    averaged = {}
    for key, vals in accum.items():
        mean_vals = np.mean(vals, axis=0)
        if mode.startswith("linear"):
            averaged[key] = (float(mean_vals[0]), float(mean_vals[1]))
        else:
            averaged[key] = mean_vals.tolist()
    return (mode, averaged)


def save_correction_json(path: Path, correction):
    """Persist averaged correction to JSON for later reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"type": correction[0], "coeffs": {}}
    for k, v in correction[1].items():
        if isinstance(v, (list, tuple, np.ndarray)):
            payload["coeffs"][k] = np.asarray(v, dtype=float).tolist()
        else:
            payload["coeffs"][k] = float(v)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_correction_json(path: Path):
    """Load a bias correction JSON saved by save_correction_json."""
    if not path.exists():
        raise FileNotFoundError(f"Correction JSON not found: {path}")
    with path.open() as f:
        payload = json.load(f)
    ctype = payload.get("type")
    coeffs = payload.get("coeffs", {})
    corr = {}
    for k, v in coeffs.items():
        if ctype and ctype.startswith("linear"):
            corr[k] = (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else tuple(v)
        else:
            corr[k] = v
    return (ctype, corr)


def _get_age_column(df):
    for col in ("AGE", "age_days", "final_age_days"):
        if col in df.columns:
            return col
    return None


def filter_df_by_days(df: pd.DataFrame, days, label: str) -> pd.DataFrame:
    """Filter dataframe by integer day values for reporting-only loaders."""
    if df is None or df.empty:
        print(f"[DATA] {label}: empty")
        return df
    if days is None:
        return df
    day_arr = np.rint(df["day"].astype(float).to_numpy()).astype(int)
    mask = np.isin(day_arr, list(days))
    out = df.loc[mask].copy()
    kept_days = sorted(np.unique(day_arr[mask]).tolist()) if mask.any() else []
    print(f"[DATA] {label}: day filter {list(days)} -> {len(df)} to {len(out)} rows (days kept={kept_days})")
    return out


def check_split_health(train_df, val_df, test_df, ctrl_df):
    """Basic sanity checks for split leakage and cohort/age coverage."""
    train_rats = set(train_df["rat_id"].unique())
    val_rats = set(val_df["rat_id"].unique())
    test_rats = set(test_df["rat_id"].unique())
    ctrl_rats = set(ctrl_df["rat_id"].unique())

    overlap_tv = train_rats & val_rats
    overlap_tt = train_rats & test_rats
    overlap_tc = train_rats & ctrl_rats
    overlap_vt = val_rats & test_rats
    if overlap_tv or overlap_tt or overlap_tc or overlap_vt:
        print(f"[WARN] Rat overlap detected: train∩val={len(overlap_tv)}, train∩test={len(overlap_tt)}, train∩ctrl={len(overlap_tc)}, val∩test={len(overlap_vt)}")
    else:
        print("[CHECK] No rat_id overlap across splits.")

    age_col = _get_age_column(train_df) or _get_age_column(val_df) or _get_age_column(test_df) or "AGE"

    def stats(df, label):
        if age_col not in df.columns or df.empty:
            print(f"[AGE] {label}: no data")
            return
        s = df.groupby("cohort")[age_col].agg(["count", "min", "median", "max"])
        print(f"[AGE] {label} age stats by cohort:\n{s}")

    stats(train_df, "train")
    stats(val_df, "val")
    stats(test_df, "test")
    stats(ctrl_df, "ctrl_test")


from preprocess_age_lora import prepare_data, make_loaders  # noqa: E402
from data_prep_age_lora import load_metadata  # noqa: E402
from config import (
    CSV_PATH,
    BACKBONE_CKPT,
    IMG_SIZE,
    IMAGE_TYPES,
    DAY_WHITELIST,
    COHORTS_TO_KEEP,
    LORA_RANK,
    LORA_BLOCKS,
    LORA_ALPHA,
    LORA_DROPOUT,
    UPSAMPLE_FACTOR,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    VAL_SPLIT,
    TEST_SPLIT,
    TRAIN_GROUPS,
    TEST_GROUPS,
    MIXUP_ALPHA,
    MIXUP_PROB,
    CUTMIX_ALPHA,
    CUTMIX_PROB,
    LABEL_NOISE_STD,
    HOLDOUT_DAY,
    HOLDOUT_TEST_ONLY,
    SUBSET_SIZE,
    SUBSET_FRACTION,
    AUG_LEVEL,
    OUTPUT_ROOT,
)
from retfound_lora_age_pred import RETFoundLoRAAgePred  # noqa: E402
from simple_baseline import SimpleXceptionAgePred  # noqa: E402
from bias_correction import fit_linear_correction, apply_correction, fit_poly_correction, apply_poly_correction  # noqa: E402
from trainer import Trainer  # noqa: E402
import eval_suite_retfound as eval_suite  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train/Eval RETFound LoRA age model")
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    p.add_argument("--backbone-ckpt", type=Path, default=BACKBONE_CKPT)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--global-pool", action="store_true",
                   help="Use global pooling (CLS token) in RETFound backbone")
    p.add_argument("--test-image-types", type=str, nargs="*", default=None, help="Override image types for test/ctrl_test loaders (e.g., REGAVG)")
    p.add_argument("--test-single-image", action="store_true", help="Deduplicate test/ctrl_test to one image per rat/eye/day")
    p.add_argument(
        "--day-whitelist",
        type=int,
        nargs="*",
        default=None,
        help="Override allowed study days (e.g., --day-whitelist 0 30 90). Default uses config DAY_WHITELIST.",
    )
    p.add_argument(
        "--all-ages",
        action="store_true",
        help="Disable day whitelist and train/eval on all available ages/days in the metadata.",
    )
    p.add_argument(
        "--control-eval-days",
        type=int,
        nargs="*",
        default=None,
        help="Restrict only control evaluation outputs/metrics to these days (e.g., --control-eval-days 0 90).",
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR,
                   help="Learning rate (suggested: 1e-5 to 5e-4, log scale)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--test-split", type=float, default=TEST_SPLIT)
    p.add_argument("--baseline-test-split", type=float, default=0.0,
                   help="Hold out a fraction of Baseline rats into test set (rat-level) while keeping the rest in training.")
    p.add_argument(
        "--cohort-stratified-split",
        action="store_true",
        help="Split train/val/test within each cohort (rat-level) to preserve cohort balance.",
    )
    p.add_argument("--save-lora", type=Path, default=OUTPUT_ROOT / "checkpoints/retfound_lora_age_weights.pt")
    p.add_argument("--no-save-lora", action="store_true", dest="skip_save_lora",
                   help="Skip saving LoRA weights (useful for rapid sweeps/tuning)")
    p.add_argument("--lora-rank", type=int, default=LORA_RANK,
                   help="LoRA rank (suggested: 4, 8, 16, 32)")
    p.add_argument("--lora-blocks", type=int, default=LORA_BLOCKS)
    p.add_argument("--lora-alpha", type=float, default=LORA_ALPHA,
                   help="LoRA alpha (suggested: 16, 32, 64; often ~2x rank)")
    p.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT,
                   help="LoRA dropout (suggested: 0.05–0.30)")
    p.add_argument("--upsample-factor", type=int, default=UPSAMPLE_FACTOR)
    p.add_argument(
        "--keep-spatial-tokens",
        action="store_true",
        help="Use patch-token spatial feature maps before the regression head (default: CLS-only features for age regression).",
    )
    p.add_argument(
        "--mil-attention",
        action="store_true",
        help="Use attention-MIL over all images in each (rat_id, eye, day) case (disables fusion modes).",
    )
    p.add_argument("--mil-attn-dim", type=int, default=128, help="Hidden dim for MIL attention scorer.")
    p.add_argument("--mil-hidden-dim", type=int, default=256, help="Hidden dim for MIL regression MLP.")
    p.add_argument(
        "--input-pre-adapter",
        action="store_true",
        help="Enable a small residual adapter before RETFound patch embedding (helps device/style shift adaptation).",
    )
    p.add_argument(
        "--input-pre-adapter-hidden",
        type=int,
        default=16,
        help="Hidden channels for the input pre-adapter (default: 16).",
    )
    p.add_argument("--pred-csv", type=Path, default=OUTPUT_ROOT / "predictions/predictions.csv")
    p.add_argument("--metrics-csv", type=Path, default=OUTPUT_ROOT / "predictions/metrics_summary.csv",
                   help="Where to save summary metrics for control/stress predictions")
    p.add_argument("--name-suffix", type=str, default="", help="Optional suffix to append to saved artifacts (e.g., _fold2_eval)")
    p.add_argument("--train-groups", type=str, nargs="*", default=TRAIN_GROUPS,
                   help="Groups to use for training/validation (normalized names)")
    p.add_argument("--test-groups", type=str, nargs="*", default=TEST_GROUPS,
                   help="Groups to use for held-out testing (normalized names)")
    p.add_argument("--bias-correction", action="store_true", default=True,
                   help="Fit linear bias correction on val set and apply to test preds (default: on)")
    p.add_argument("--no-bias-correction", action="store_false", dest="bias_correction",
                   help="Disable bias correction")
    p.add_argument("--bias-correction-cohort-specific", action="store_true",
                   help="Fit/apply bias correction separately for each cohort (overrides young/old buckets)")
    p.add_argument("--bias-correction-mode", type=str, default="linear", choices=["linear", "poly2"], help="Bias correction mode")
    p.add_argument(
        "--save-correction-json",
        type=Path,
        default=None,
        help=(
            "(CV only) Where to save the averaged bias-correction JSON across folds. "
            "Default: outputs/predictions/bias_correction_cv_k{K}.json"
        ),
    )
    p.add_argument(
        "--no-save-correction-json",
        action="store_true",
        help="(CV only) Skip saving the averaged bias-correction JSON (useful for sweeps/Optuna to avoid overwriting).",
    )
    p.add_argument("--baseline-day", type=float, default= None, help="Optional day to anchor RAG to ~0 (subtract mean gap at this day)")
    p.add_argument("--baseline-group", type=str, default="Controls", help="Group to use for baseline anchoring")
    p.add_argument("--mixup-alpha", type=float, default=MIXUP_ALPHA, help="Beta alpha for mixup (0 disables)")
    p.add_argument("--mixup-prob", type=float, default=MIXUP_PROB, help="Probability to apply mixup to a batch")
    p.add_argument("--cutmix-alpha", type=float, default=CUTMIX_ALPHA, help="Beta alpha for CutMix (0 disables)")
    p.add_argument("--cutmix-prob", type=float, default=CUTMIX_PROB, help="Probability to apply CutMix to a batch")
    p.add_argument("--label-noise-std", type=float, default=LABEL_NOISE_STD,
                   help="Label noise std (suggested: 0.5–3.0 days)")
    # Skew loss disabled (kept for backward compatibility; no-op in Trainer)
    p.add_argument("--skew-loss-factor", type=float, default=1.0,
                   help="(Deprecated) Skew disabled; Smooth L1 only")
    p.add_argument("--skew-loss-exp", action="store_true",
                   help="(Deprecated) Skew disabled; Smooth L1 only")
    p.add_argument("--skew-lambda-max", type=float, default=0.0,
                   help="(Deprecated) Skew disabled; Smooth L1 only")
    p.add_argument("--skew-age-min", type=float, default=None,
                   help="(Deprecated) Skew disabled; Smooth L1 only")
    p.add_argument("--skew-age-max", type=float, default=None,
                   help="(Deprecated) Skew disabled; Smooth L1 only")
    p.add_argument("--skew-age-median", type=float, default=None,
                   help="(Deprecated) Skew disabled; Smooth L1 only")
    p.add_argument("--aug-level", type=str, default=AUG_LEVEL, choices=["low", "medium", "high"],
                   help="Augmentation strength for training transforms")
    p.add_argument(
        "--no-photometric-aug",
        action="store_true",
        help="Disable train-time photometric augmentation (keep robust intensity normalization and other transforms).",
    )
    p.add_argument("--early-fusion", action="store_true", help="Average images per rat/eye/day before backbone (early fusion)")
    p.add_argument("--late-fusion", action="store_true", default=True,
                   help="Average predictions per rat/eye/day after head (late fusion, default: on)")
    p.add_argument("--no-late-fusion", action="store_false", dest="late_fusion",
                   help="Disable late fusion")
    p.add_argument("--tta", action="store_true", help="Enable simple TTA (orig + horizontal flip) during predict")
    p.add_argument("--holdout-day", type=float, default=HOLDOUT_DAY, help="Remove this day from train/val; optional day-only test if holdout-test-only")
    p.add_argument("--holdout-test-only", action="store_true", default=HOLDOUT_TEST_ONLY, help="If set, restrict test loader to holdout day")
    p.add_argument("--subset-size", type=int, default=SUBSET_SIZE, help="Optional number of training rows to sample (data efficiency)")
    p.add_argument("--subset-fraction", type=float, default=SUBSET_FRACTION, help="Optional fraction of training rows to sample (data efficiency)")
    p.add_argument("--aggregate-features", action="store_true", help="Average spatial features per rat/day before head (feature-level aggregation)")
    p.add_argument("--no-aggregate", action="store_true", help="Keep per-image rows in predictions (disable rat/eye/day averaging)")
    p.add_argument("--aggregate-by-rat", action="store_true", help="Aggregate across eyes per rat/day (ignore eye in fusion/aggregation)")
    p.add_argument("--save-val-preds", type=Path, default=None, help="Optional path to save validation predictions CSV (useful for baseline stats)")
    p.add_argument("--right-eye-only", action="store_true", help="Use only right-eye (OD) images for training/val/test")
    p.add_argument("--load-lora", type=Path, default=None, help="Optional path to load LoRA weights (for eval-only)")
    p.add_argument("--eval-only", action="store_true", help="Skip training; load weights and run eval/prediction only")
    p.add_argument("--use-saved-correction", action="store_true", help="Apply correction stored in checkpoint even if --bias-correction is off")
    p.add_argument("--lr-patience", type=int, default=3, help="LR scheduler patience (epochs) for Plateau scheduler")
    p.add_argument("--lr-factor", type=float, default=0.5, help="LR scheduler decay factor when plateauing")
    p.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience (epochs)")
    p.add_argument("--model-type", type=str, default="retfound", choices=["retfound", "xception"], help="Model architecture to use")
    p.add_argument("--baseline-pretrained", action="store_true", help="Use ImageNet-pretrained weights for the Xception baseline (requires cached weights)")
    p.add_argument("--save-saliency-dir", type=Path, default=None,
                   help="Optional dir to save saliency heatmaps (one PNG per image)")
    p.add_argument("--save-report-dir", type=Path, default=None,
                   help="Optional dir to save data stats, train curves, and val prediction plots/tables")
    p.add_argument("--run-auroc-report", action="store_true",
                   help="After prediction, run eval_suite_retfound AUROC with --control-day-anchor --show-delta")
    # Cross-validation
    p.add_argument("--kfolds", type=int, default=0, help="If >1, enable K-fold CV on training groups (rat-level)")
    p.add_argument("--fold-index", type=int, default=0, help="Fold index to run when kfolds>1 (0-based)")
    p.add_argument("--fold-seed", type=int, default=42, help="Seed for rat shuffling in K-fold CV")
    p.add_argument("--run-all-folds", action="store_true", help="If set with kfolds>1, iterate over all folds sequentially")
    p.add_argument("--load-correction-json", type=Path, default=None, help="Load a saved bias correction JSON (overrides fitting if provided)")
    args = p.parse_args()
    if args.all_ages:
        args.day_whitelist = None
    elif args.day_whitelist is None:
        args.day_whitelist = list(DAY_WHITELIST) if DAY_WHITELIST is not None else None
    if args.control_eval_days is not None:
        args.control_eval_days = sorted({int(d) for d in args.control_eval_days})
    return args


def build_model(args):
    if args.model_type == "xception":
        print("[MODEL] Using Xception baseline")
        model = SimpleXceptionAgePred(
            pretrained=args.baseline_pretrained,
            head_hidden_dim=256,
            head_dropout=args.lora_dropout,
        )
    else:
        print("[MODEL] Using RETFound + LoRA")
        if args.global_pool:
            print("[WARN] --global-pool is incompatible with the spatial regression head; forcing global_pool=False.")
        if not args.keep_spatial_tokens:
            print("[MODEL] Using CLS-only RETFound features for age regression (spatial tokens disabled).")
        if args.mil_attention:
            print(f"[MODEL] Attention-MIL enabled (attn_dim={args.mil_attn_dim}, hidden_dim={args.mil_hidden_dim})")
        if args.input_pre_adapter:
            print(f"[MODEL] Input pre-adapter enabled (hidden={args.input_pre_adapter_hidden})")
        model = RETFoundLoRAAgePred(
            ckpt_path=args.backbone_ckpt,
            img_size=args.img_size,
            global_pool=False,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_blocks=args.lora_blocks,
            lora_dropout=args.lora_dropout,
            upsample_factor=args.upsample_factor,
            keep_spatial_tokens=args.keep_spatial_tokens,
            use_pre_adapter=args.input_pre_adapter,
            pre_adapter_hidden_dim=args.input_pre_adapter_hidden,
            use_mil_attention=args.mil_attention,
            mil_attn_dim=args.mil_attn_dim,
            mil_hidden_dim=args.mil_hidden_dim,
    )
    return model


def run_fold(args):
    # Fusion mode sanity: only one of early_fusion / aggregate_features / late_fusion
    fusion_flags = int(bool(getattr(args, "early_fusion", False))) + int(bool(getattr(args, "aggregate_features", False))) + int(bool(getattr(args, "late_fusion", False)))
    if fusion_flags > 1:
        raise SystemExit("Choose only one fusion mode: early-fusion OR aggregate-features OR late-fusion.")
    if getattr(args, "mil_attention", False):
        if args.model_type != "retfound":
            raise SystemExit("--mil-attention is currently supported only with --model-type retfound.")
        if any([getattr(args, "early_fusion", False), getattr(args, "aggregate_features", False), getattr(args, "late_fusion", False), getattr(args, "aggregate_by_rat", False)]):
            print("[MIL] Disabling fusion/aggregation flags (MIL mode defines bag-level aggregation).")
        args.early_fusion = False
        args.aggregate_features = False
        args.late_fusion = False
        args.aggregate_by_rat = False
        if args.model_type == "retfound" and args.lora_blocks != 0:
            print(f"[MIL] Freezing RETFound backbone in MIL baseline: forcing --lora-blocks 0 (was {args.lora_blocks}).")
            args.lora_blocks = 0

    device = torch.device(args.device)
    print(f"[DEVICE] requested={args.device} | torch.cuda.is_available()={torch.cuda.is_available()} | using={device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA device requested but torch.cuda.is_available() is False. Check driver/NVML access (see torch.version.cuda).")
        print("[WARN] Falling back to CPU; training will be slow.")
    print(f"[DATA] day_whitelist={'ALL' if args.day_whitelist is None else args.day_whitelist}")
    print(f"[AUG] photometric_aug={'OFF' if args.no_photometric_aug else 'ON'} | aug_level={args.aug_level}")

    use_folds = args.kfolds and args.kfolds > 1
    fold_suffix = f"_fold{args.fold_index}" if use_folds else ""
    extra_suffix = args.name_suffix if args.name_suffix else ""
    full_suffix = f"{fold_suffix}{extra_suffix}"
    # Auto-append fold suffix to saved artifacts to avoid overwriting between folds
    if use_folds:
        if args.save_lora:
            args.save_lora = apply_suffix(args.save_lora, fold_suffix)
        if args.save_val_preds:
            args.save_val_preds = apply_suffix(args.save_val_preds, fold_suffix)
        if args.pred_csv:
            args.pred_csv = apply_suffix(args.pred_csv, fold_suffix)
        if args.metrics_csv:
            args.metrics_csv = apply_suffix(args.metrics_csv, fold_suffix)
    if extra_suffix:
        if args.save_lora:
            args.save_lora = apply_suffix(args.save_lora, extra_suffix)
        if args.save_val_preds:
            args.save_val_preds = apply_suffix(args.save_val_preds, extra_suffix)
        if args.pred_csv:
            args.pred_csv = apply_suffix(args.pred_csv, extra_suffix)
        if args.metrics_csv:
            args.metrics_csv = apply_suffix(args.metrics_csv, extra_suffix)
        if args.save_saliency_dir:
            args.save_saliency_dir = apply_dir_suffix(args.save_saliency_dir, extra_suffix)
    report_dir = args.save_report_dir
    if report_dir:
        report_dir = Path(report_dir)
        if use_folds:
            report_dir = report_dir / f"fold{args.fold_index}"
        report_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.pred_csv.parent if args.pred_csv else (OUTPUT_ROOT / "predictions")

    # Remove stale outputs with matching names to avoid mixing runs
    cleanup_outputs(full_suffix, args)
    if use_folds:
        if args.fold_index < 0 or args.fold_index >= args.kfolds:
            raise SystemExit(f"fold_index must be in [0, {args.kfolds-1}]")
        base_train_df, _, base_ctrl_test_df, base_test_df, _ = prepare_data(
            csv_path=args.csv,
            image_types=IMAGE_TYPES,
            day_whitelist=args.day_whitelist,
            test_image_types=args.test_image_types,
            test_single_image=args.test_single_image,
            include_recovery_days=False,
            cohorts_to_keep=COHORTS_TO_KEEP,
            exclude_recovery_paths=False,
            train_groups=args.train_groups,
            test_groups=args.test_groups,
            val_split=0.0,
            test_split=args.test_split,
            baseline_test_split=args.baseline_test_split,
            holdout_day=args.holdout_day,
            holdout_test_only=args.holdout_test_only,
            subset_size=args.subset_size,
            subset_fraction=args.subset_fraction,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.fold_seed,
            right_eye_only=args.right_eye_only,
            aug_level=args.aug_level,
            cohort_stratified_split=args.cohort_stratified_split,
            enable_photometric_aug=not args.no_photometric_aug,
            mil_attention=args.mil_attention,
        )
        rat_ids = base_train_df["rat_id"].unique()
        rng = np.random.default_rng(args.fold_seed)
        # Coarse age bins per rat for stratified grouping
        rat_age = base_train_df.groupby("rat_id")["AGE"].mean().reindex(rat_ids).to_numpy()
        def make_bins(vals, min_bins=2, max_bins=3):
            for nb in range(max_bins, min_bins - 1, -1):
                edges = np.linspace(vals.min(), vals.max(), num=nb + 1)
                edges = np.unique(edges)
                if len(edges) < 2:
                    continue
                bins = np.digitize(vals, edges[1:-1], right=True)
                if np.bincount(bins, minlength=nb).min(initial=0) >= 2:
                    return bins
            edges = np.linspace(vals.min(), vals.max(), num=3)
            return np.digitize(vals, edges[1:-1], right=True)
        bins = make_bins(rat_age) if len(rat_ids) > 1 else np.zeros_like(rat_ids, dtype=int)
        sgkf = StratifiedGroupKFold(n_splits=args.kfolds, shuffle=True, random_state=args.fold_seed) if len(np.unique(bins)) > 1 else None
        splits = []
        if sgkf:
            for tr_idx, va_idx in sgkf.split(np.zeros_like(rat_ids), bins, groups=rat_ids):
                splits.append((tr_idx, va_idx))
        else:
            gkf = GroupKFold(n_splits=args.kfolds)
            for tr_idx, va_idx in gkf.split(np.zeros_like(rat_ids), groups=rat_ids):
                splits.append((tr_idx, va_idx))
        tr_idx, va_idx = splits[args.fold_index]
        train_rats = rat_ids[tr_idx]
        val_rats = rat_ids[va_idx]
        train_df = base_train_df[base_train_df["rat_id"].isin(train_rats)]
        val_df = base_train_df[base_train_df["rat_id"].isin(val_rats)]
        ctrl_test_df = base_ctrl_test_df
        test_df = base_test_df
        print(f"[FOLD] k={args.kfolds} idx={args.fold_index} | train rats={len(train_rats)} val rats={len(val_rats)}")
        train_loader, val_loader, test_loader, ctrl_test_loader = make_loaders(
            train_df, val_df, test_df, ctrl_test_df,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            aug_level=args.aug_level,
            enable_photometric_aug=not args.no_photometric_aug,
            mil_attention=args.mil_attention,
        )
    else:
        train_df, val_df, ctrl_test_df, test_df, (train_loader, val_loader, test_loader, ctrl_test_loader) = prepare_data(
            csv_path=args.csv,
            image_types=IMAGE_TYPES,
            day_whitelist=args.day_whitelist,
            test_image_types=args.test_image_types,
            test_single_image=args.test_single_image,
            include_recovery_days=False,
            cohorts_to_keep=COHORTS_TO_KEEP,
            exclude_recovery_paths=False,
            train_groups=args.train_groups,
            test_groups=args.test_groups,
            val_split=args.val_split,
            test_split=args.test_split,
            baseline_test_split=args.baseline_test_split,
            holdout_day=args.holdout_day,
            holdout_test_only=args.holdout_test_only,
            subset_size=args.subset_size,
            subset_fraction=args.subset_fraction,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=42,
            right_eye_only=args.right_eye_only,
            aug_level=args.aug_level,
            cohort_stratified_split=args.cohort_stratified_split,
            enable_photometric_aug=not args.no_photometric_aug,
            mil_attention=args.mil_attention,
        )

    full_df = pd.concat([train_df, val_df, ctrl_test_df, test_df], ignore_index=True)
    total_rats = full_df["rat_id"].nunique()
    train_rats = train_df["rat_id"].nunique()
    val_rats = val_df["rat_id"].nunique()
    ctrl_test_rats = ctrl_test_df["rat_id"].nunique()
    test_rats = test_df["rat_id"].nunique()
    missing_ids = (full_df["rat_id"].astype(str).str.strip() == "").sum()
    print(f"[DATA] rats total={total_rats} | train={train_rats} val={val_rats} ctrl_test={ctrl_test_rats} test={test_rats} | rows={len(full_df)} | missing rat_id rows={missing_ids}")
    group_rat_counts = full_df.groupby("group_norm")["rat_id"].nunique()
    cohort_rat_counts = full_df.groupby("cohort")["rat_id"].nunique()
    cohort_rat_counts = cohort_rat_counts.reindex(COHORTS_TO_KEEP, fill_value=0)
    print(f"[DATA] rats per group: {group_rat_counts.to_dict()}")
    print(f"[DATA] rats per cohort (including zeros): {cohort_rat_counts.to_dict()}")
    check_split_health(train_df, val_df, test_df, ctrl_test_df)

    if report_dir:
        data_stats = {
            "total_rats": int(total_rats),
            "train_rats": int(train_rats),
            "val_rats": int(val_rats),
            "ctrl_test_rats": int(ctrl_test_rats),
            "test_rats": int(test_rats),
            "rows": int(len(full_df)),
            "missing_rat_id_rows": int(missing_ids),
            "rats_per_group": group_rat_counts.to_dict(),
            "rats_per_cohort": cohort_rat_counts.to_dict(),
            "days_present": full_df["day"].value_counts().sort_index().to_dict(),
        }
        (report_dir / "data_stats.json").write_text(json.dumps(data_stats, indent=2))

    # Optional reporting-only control day filter (does not affect training/val loss).
    control_holdout_loader = ctrl_test_loader
    control_val_fallback_loader = val_loader
    if args.control_eval_days is not None:
        empty_like = train_df.iloc[0:0]

        def _build_eval_loader_from_df(df_subset: pd.DataFrame):
            if df_subset is None or df_subset.empty:
                return None
            _, val_like_loader, _, _ = make_loaders(
                empty_like,
                df_subset,
                empty_like,
                empty_like,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                aug_level=args.aug_level,
                enable_photometric_aug=not args.no_photometric_aug,
                mil_attention=args.mil_attention,
            )
            return val_like_loader

        ctrl_eval_df = filter_df_by_days(ctrl_test_df, args.control_eval_days, "ctrl_eval_holdout")
        val_ctrl_eval_df = filter_df_by_days(val_df, args.control_eval_days, "ctrl_eval_val_fallback")
        control_holdout_loader = _build_eval_loader_from_df(ctrl_eval_df)
        control_val_fallback_loader = _build_eval_loader_from_df(val_ctrl_eval_df)

    model = build_model(args).to(device)
    trainer = Trainer(model, device)

    correction = None
    if args.load_correction_json:
        try:
            correction = load_correction_json(args.load_correction_json)
            print(f"[LOAD] Loaded bias correction from JSON: {args.load_correction_json}")
        except Exception as e:
            raise SystemExit(f"Failed to load correction JSON: {e}")

    load_path = args.load_lora if args.load_lora else None
    if load_path and load_path.exists():
        ckpt = torch.load(load_path, map_location="cpu")
        if isinstance(ckpt, dict) and "backbone_lora" in ckpt:
            # Put LoRA layers in unmerged mode before loading A/B deltas.
            if args.model_type == "retfound":
                model.backbone.train()
            model.backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
            # Merge loaded deltas for inference / evaluation.
            if args.model_type == "retfound":
                model.backbone.eval()
            if args.model_type == "retfound" and hasattr(model, "pre_adapter") and model.pre_adapter is not None:
                if "pre_adapter" in ckpt:
                    model.pre_adapter.load_state_dict(ckpt["pre_adapter"], strict=False)
                else:
                    print("[LOAD] Checkpoint missing pre_adapter weights (adapter enabled in current model).")
            if args.model_type == "retfound" and hasattr(model, "mil_head") and model.mil_head is not None:
                if "mil_head" in ckpt:
                    model.mil_head.load_state_dict(ckpt["mil_head"], strict=False)
                else:
                    print("[LOAD] Checkpoint missing mil_head weights (MIL enabled in current model).")
            if "head" in ckpt:
                model.head.load_state_dict(ckpt["head"], strict=False)
            if "correction" in ckpt and (args.bias_correction or args.use_saved_correction):
                correction = ckpt["correction"]
                print(f"[LOAD] Loaded bias correction from checkpoint: {correction}")
            elif "correction" in ckpt:
                print("[LOAD] Ignored bias correction in checkpoint (enable --use-saved-correction to apply).")
            print(f"[LOAD] Loaded LoRA weights from {load_path}")
        else:
            model.load_lora_checkpoint(str(load_path))
            print(f"[LOAD] Loaded legacy LoRA checkpoint from {load_path}")
    elif args.eval_only and args.save_lora.exists():
        model.load_lora_checkpoint(str(args.save_lora))
        print(f"[LOAD] Loaded LoRA weights from {args.save_lora}")

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    metrics_log = []
    val_preds_cache = None

    if not args.eval_only:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )
        patience_counter = 0
        early_stop_patience = args.early_stop_patience

        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_one_epoch(train_loader, optimizer, args) if train_loader else float("nan")
            val_loss = trainer.evaluate(val_loader, args) if val_loader else float("nan")
            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else float("nan")
            metrics_log.append({
                "epoch": epoch,
                "train_L1": float(train_loss),
                "val_L1": float(val_loss),
                "lr": float(current_lr),
            })
            print(f"[EPOCH {epoch}] train_L1={train_loss:.4f} val_L1={val_loss:.4f}")
            if val_loader and not np.isnan(val_loss):
                scheduler.step(val_loss)
            if val_loader and not np.isnan(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                print(f"[EARLY STOP] No val improvement for {early_stop_patience} epochs.")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"[INFO] Loaded best checkpoint from epoch {best_epoch} (val_L1={best_val:.4f})")
    else:
        print("[INFO] Eval-only mode: skipping training")

    if args.bias_correction and val_loader:
        y_true, y_pred, y_coh = Trainer.collect_preds(model, val_loader, device)
        if y_true is not None and y_pred is not None and y_coh is not None:
            val_preds_cache = (y_true, y_pred, y_coh)
            n_calib = len(y_true)
            age_span = float(np.max(y_true) - np.min(y_true)) if n_calib else 0.0
            if n_calib < MIN_CALIB_SAMPLES or age_span <= 0.0:
                print(f"[CALIB] Skipped bias correction (n={n_calib}, age_span={age_span:.1f}); keeping existing correction {correction}")
            else:
                coh_str = np.asarray(y_coh).astype(str)
                corr_dict = {}
                if args.bias_correction_cohort_specific:
                    unique_coh = np.unique(coh_str)
                    for c in unique_coh:
                        mask = coh_str == c
                        if not mask.any():
                            continue
                        if args.bias_correction_mode == "poly2":
                            coeffs = fit_poly_correction(y_true[mask], y_pred[mask], degree=2)
                            corr_dict[str(c)] = coeffs
                        else:
                            alpha, beta = fit_linear_correction(y_true[mask], y_pred[mask])
                            corr_dict[str(c)] = (alpha, beta)
                    if args.bias_correction_mode == "poly2":
                        correction = ("poly_cohort_exact", corr_dict)
                    else:
                        correction = ("linear_cohort_exact", corr_dict)
                    print(f"[CALIB] Fitted bias correction per cohort: keys={list(corr_dict.keys())}")
                else:
                    # Fit separate corrections for young (coh 1/2) and old (coh 3)
                    young_mask = np.isin(coh_str, ["1", "2"])
                    old_mask = coh_str == "3"
                    if args.bias_correction_mode == "poly2":
                        if young_mask.any():
                            coeffs = fit_poly_correction(y_true[young_mask], y_pred[young_mask], degree=2)
                            corr_dict["young"] = coeffs
                        if old_mask.any():
                            coeffs = fit_poly_correction(y_true[old_mask], y_pred[old_mask], degree=2)
                            corr_dict["old"] = coeffs
                        correction = ("poly_cohort", corr_dict)
                        print(f"[CALIB] Fitted polynomial bias correction per cohort-group (young/old): keys={list(corr_dict.keys())}")
                    else:
                        if young_mask.any():
                            alpha, beta = fit_linear_correction(y_true[young_mask], y_pred[young_mask])
                            corr_dict["young"] = (alpha, beta)
                        if old_mask.any():
                            alpha, beta = fit_linear_correction(y_true[old_mask], y_pred[old_mask])
                            corr_dict["old"] = (alpha, beta)
                        correction = ("linear_cohort", corr_dict)
                        print(f"[CALIB] Fitted linear bias correction per cohort-group (young/old): keys={list(corr_dict.keys())}")

    if (not args.eval_only) and (not getattr(args, "skip_save_lora", False)):
        args.save_lora.parent.mkdir(parents=True, exist_ok=True)
        if args.model_type == "retfound":
            save_dict = {
                "backbone_lora": lora.lora_state_dict(model.backbone, bias="none"),
                "head": model.head.state_dict(),
            }
            if hasattr(model, "pre_adapter") and model.pre_adapter is not None:
                save_dict["pre_adapter"] = model.pre_adapter.state_dict()
            if hasattr(model, "mil_head") and model.mil_head is not None:
                save_dict["mil_head"] = model.mil_head.state_dict()
        else:
            # Baseline path keeps full backbone weights for compatibility.
            save_dict = {
                "backbone_lora": model.backbone.state_dict(),
                "head": model.head.state_dict() if hasattr(model, "head") else None,
            }
        if correction is not None:
            save_dict["correction"] = correction
        torch.save(save_dict, args.save_lora)
        print(f"[DONE] Saved LoRA weights to {args.save_lora}")

    if val_loader and args.save_val_preds:
        out_path = args.save_val_preds if args.save_val_preds.is_absolute() else (args.pred_csv.parent / args.save_val_preds)
        print("[PRED] Running validation set…")
        trainer.predict_to_csv(val_loader, out_path.name, args, device, correction=correction)

    fold_suffix = f"_fold{args.fold_index}" if use_folds else ""
    control_csv_path = None
    control_loader_to_use = control_holdout_loader
    if use_folds and not control_loader_to_use:
        control_loader_to_use = control_val_fallback_loader
    if control_loader_to_use:
        print("[PRED] Running held-out Controls test set…")
        control_csv_name = f"control_test_results{full_suffix}.csv"
        control_csv_path = pred_dir / control_csv_name
        trainer.predict_to_csv(
            control_loader_to_use,
            control_csv_name,
            args,
            device,
            correction=correction,
            save_saliency_dir=args.save_saliency_dir if args.save_saliency_dir else None,
        )
    elif control_val_fallback_loader:
        print("[PRED] No control holdout; running Controls validation set for metrics…")
        control_csv_name = f"control_val_results{full_suffix}.csv"
        control_csv_path = pred_dir / control_csv_name
        trainer.predict_to_csv(
            control_val_fallback_loader,
            control_csv_name,
            args,
            device,
            correction=correction,
            save_saliency_dir=args.save_saliency_dir if args.save_saliency_dir else None,
        )
    else:
        print("[PRED] Skipping Controls predictions (no control holdout or val set).")
    stress_csv_name = f"rag_experimental_results{full_suffix}.csv"
    print("[PRED] Running HLS/Recovery/High_CO2 test set…")
    trainer.predict_to_csv(
        test_loader,
        stress_csv_name,
        args,
        device,
        correction=correction,
        save_saliency_dir=args.save_saliency_dir if args.save_saliency_dir else None,
    )

    if report_dir:
        metrics = []
        ctrl_path = control_csv_path or (pred_dir / f"control_test_results{full_suffix}.csv")
        stress_path = pred_dir / stress_csv_name
        for label, p in (("control", ctrl_path), ("stress", stress_path)):
            m = compute_metrics_csv(p)
            if m:
                m["split"] = label
                metrics.append(m)
        if metrics:
            pd.DataFrame(metrics).to_csv(report_dir / "test_metrics.csv", index=False)

    if args.metrics_csv:
        metrics = []
        ctrl_path = control_csv_path
        stress_path = pred_dir / stress_csv_name
        for label, p in (("control", ctrl_path), ("stress", stress_path)):
            if p is None:
                continue
            m = compute_metrics_csv(p)
            if m:
                m["split"] = label
                metrics.append(m)
        if metrics:
            args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(metrics).to_csv(args.metrics_csv, index=False)
            print(f"[METRICS] Saved summary metrics to {args.metrics_csv}")
        else:
            print("[METRICS] No metrics written (missing prediction CSVs).")

    if args.run_auroc_report:
        ctrl_path = pred_dir / f"control_test_results{full_suffix}.csv"
        stress_path = pred_dir / f"rag_experimental_results{full_suffix}.csv"
        if ctrl_path.exists() and stress_path.exists():
            print("[REPORT] Running AUROC delta report...")
            report_args = argparse.Namespace(
                pred_csv=[[ctrl_path, stress_path]],
                min_day=0.0,
                exclude_recovery=False,
                control_day_anchor=True,
                filter_cohorts=None,
                filter_sex=None,
                controls_label="Controls",
                hls_label="HLS (U)",
                extra_control_groups=[],
                extra_disease_groups=["High_CO2_Controls", "High_CO2_HLS"],
                control_sources=[],
                strict_hls_only=False,
                show_delta=True,
            )
            try:
                eval_suite.run_auroc(report_args)
            except SystemExit as e:
                print(f"[REPORT] AUROC report skipped: {e}")
        else:
            print("[REPORT] AUROC report skipped (prediction CSVs not found).")

    if report_dir and metrics_log:
        df_metrics = pd.DataFrame(metrics_log)
        df_metrics.to_csv(report_dir / "train_metrics.csv", index=False)
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.figure()
            plt.plot(df_metrics["epoch"], df_metrics["train_L1"], label="train_L1")
            if not df_metrics["val_L1"].isna().all():
                plt.plot(df_metrics["epoch"], df_metrics["val_L1"], label="val_L1")
            plt.xlabel("Epoch")
            plt.ylabel("L1 loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(report_dir / "loss_curve.png", dpi=200)
            plt.close()
        except Exception as e:
            print(f"[REPORT] Could not save loss curve plot: {e}")

    if report_dir and val_loader:
        if val_preds_cache is None:
            val_preds_cache = Trainer.collect_preds(model, val_loader, device)
        if val_preds_cache and val_preds_cache[0] is not None and val_preds_cache[1] is not None:
            v_true = np.asarray(val_preds_cache[0])
            v_pred = np.asarray(val_preds_cache[1])
            df_val = pd.DataFrame({"age_true": v_true, "age_pred": v_pred})
            df_val.to_csv(report_dir / "val_predictions.csv", index=False)
            try:
                import matplotlib.pyplot as plt  # type: ignore
                plt.figure()
                plt.scatter(v_true, v_pred, alpha=0.4, s=10)
                lims = [min(v_true.min(), v_pred.min()), max(v_true.max(), v_pred.max())]
                plt.plot(lims, lims, "r--", linewidth=1)
                plt.xlabel("True age")
                plt.ylabel("Predicted age")
                plt.tight_layout()
                plt.savefig(report_dir / "val_true_vs_pred.png", dpi=200)
                plt.close()
            except Exception as e:
                print(f"[REPORT] Could not save val scatter plot: {e}")

    return correction


def main():
    args = parse_args()
    use_folds = args.kfolds and args.kfolds > 1
    if use_folds and args.run_all_folds:
        corrections = []
        for fi in range(args.kfolds):
            print(f"[CV] Running fold {fi+1}/{args.kfolds}")
            fold_args = copy.deepcopy(args)
            fold_args.fold_index = fi
            suffix = f"_fold{fi}"
            if fold_args.save_lora:
                fold_args.save_lora = apply_suffix(fold_args.save_lora, suffix)
            if fold_args.save_val_preds:
                fold_args.save_val_preds = apply_suffix(fold_args.save_val_preds, suffix)
            corr = run_fold(fold_args)
            corrections.append(corr)
        avg_corr = average_corrections(corrections)
        if avg_corr:
            print(f"[CV] Averaged correction: {avg_corr}")
            if args.no_save_correction_json:
                print("[CV] Skipped saving averaged bias correction (--no-save-correction-json).")
            else:
                out_path = args.save_correction_json or (OUTPUT_ROOT / "predictions" / f"bias_correction_cv_k{args.kfolds}.json")
                save_correction_json(out_path, avg_corr)
                print(f"[CV] Saved averaged bias correction from {len([c for c in corrections if c])} folds to {out_path}")
    else:
        run_fold(args)


if __name__ == "__main__":
    main()
