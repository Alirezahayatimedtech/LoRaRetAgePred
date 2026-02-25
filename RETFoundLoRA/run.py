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
import re
import subprocess
from typing import Optional

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


def parse_progressive_lora_schedule(spec: str):
    """
    Parse progressive LoRA schedule text into 1-based epoch starts.

    Format (blocks:start_epoch):
      "2:0,4:6,6:12"  # 0-based starts accepted and converted to 1-based
      "2:1,4:7,6:13"  # equivalent 1-based form
    Returns: [(start_epoch_1based, active_blocks), ...]
    """
    if spec is None:
        return None
    txt = str(spec).strip()
    if not txt:
        return None

    parsed = []
    for chunk in txt.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^(\d+)\s*[:@]\s*(\d+)$", chunk)
        if not m:
            raise ValueError(
                f"Invalid progressive LoRA token '{chunk}'. "
                "Use format like '2:0,4:6,6:12' (blocks:start_epoch)."
            )
        blocks = int(m.group(1))
        start = int(m.group(2))
        parsed.append((start, blocks))

    if not parsed:
        return None
    parsed.sort(key=lambda x: x[0])
    starts = [s for s, _ in parsed]
    if len(starts) != len(set(starts)):
        raise ValueError("Duplicate start epochs in progressive LoRA schedule.")

    # Support 0-based starts for convenience.
    if starts[0] == 0:
        parsed = [(s + 1, b) for s, b in parsed]

    for start_epoch, blocks in parsed:
        if start_epoch < 1:
            raise ValueError("Progressive LoRA schedule start epochs must be >= 1.")
        if blocks < 0:
            raise ValueError("Progressive LoRA schedule block counts must be >= 0.")
    return parsed


def active_lora_blocks_for_epoch(epoch: int, schedule, default_blocks: int) -> int:
    """Resolve active LoRA block count for a given 1-based epoch."""
    if not schedule:
        return int(default_blocks)
    active = int(default_blocks)
    for start_epoch, blocks in schedule:
        if int(epoch) >= int(start_epoch):
            active = int(blocks)
        else:
            break
    return int(active)


def _get_age_column(df):
    for col in ("AGE", "age_days", "final_age_days"):
        if col in df.columns:
            return col
    return None


def resolve_ordinal_age_bins(train_df: pd.DataFrame, args):
    """
    Resolve ordered age bins for ordinal auxiliary loss from CLI or training data.

    Returns a sorted list of unique age values (floats), or None when ordinal loss is disabled.
    """
    if not bool(getattr(args, "ordinal_aux", False)):
        return None
    if float(getattr(args, "ordinal_aux_weight", 0.0) or 0.0) <= 0:
        print("[ORD] --ordinal-aux enabled but --ordinal-aux-weight <= 0; disabling ordinal auxiliary loss.")
        args.ordinal_aux = False
        return None
    if not bool(getattr(args, "mil_attention", False)):
        print("[ORD] Ordinal auxiliary loss is currently implemented for --mil-attention only; disabling.")
        args.ordinal_aux = False
        return None

    manual_bins = getattr(args, "ordinal_bin_values", None)
    if manual_bins:
        vals = sorted({float(v) for v in manual_bins})
    else:
        age_col = _get_age_column(train_df)
        if age_col is None or train_df is None or train_df.empty:
            print("[ORD] Could not infer age bins from training data; disabling ordinal auxiliary loss.")
            args.ordinal_aux = False
            return None
        age_series = pd.to_numeric(train_df[age_col], errors="coerce").dropna()
        vals = sorted({float(v) for v in age_series.tolist()})

    if len(vals) < 2:
        print(f"[ORD] Need at least 2 distinct age bins, got {len(vals)}; disabling ordinal auxiliary loss.")
        args.ordinal_aux = False
        return None

    return vals


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


def _derive_inter_eye_csv_path(pred_csv_path: Path) -> Path:
    name = pred_csv_path.name
    if "_results" in name:
        return pred_csv_path.with_name(name.replace("_results", "_inter_eye_differences", 1))
    return pred_csv_path.with_name(pred_csv_path.stem + "_inter_eye_differences.csv")


def build_inter_eye_pairs_csv(pred_csv_path: Path, out_csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Build paired OD/OS inter-eye CSV from per-eye predictions (rat_id, eye, day rows).
    """
    if pred_csv_path is None or not pred_csv_path.exists():
        print(f"[INTER-EYE] Skipping pair build (missing file): {pred_csv_path}")
        return None
    df = pd.read_csv(pred_csv_path)
    required = {"rat_id", "eye", "day", "age_true", "age_pred"}
    missing = required - set(df.columns)
    if missing:
        print(f"[INTER-EYE] Skipping pair build for {pred_csv_path.name}; missing columns: {sorted(missing)}")
        return None
    if df.empty:
        print(f"[INTER-EYE] Skipping pair build for {pred_csv_path.name}; CSV empty.")
        return None

    df = df.copy()
    df["eye"] = df["eye"].astype(str).str.upper().str.strip()
    df = df[df["eye"].isin(["OD", "OS"])].copy()
    if df.empty:
        print(f"[INTER-EYE] Skipping pair build for {pred_csv_path.name}; no OD/OS rows.")
        return None

    agg_cols = {"age_true": "mean", "age_pred": "mean"}
    for c in ("group", "sex", "cohort", "RAG"):
        if c in df.columns:
            agg_cols[c] = "first"
    for c in ("mil_bag_n_kept", "mil_bag_n_raw", "mil_bag_n_qc_dropped"):
        if c in df.columns:
            agg_cols[c] = "mean"
    if "mil_low_conf_bag" in df.columns:
        agg_cols["mil_low_conf_bag"] = "max"

    df_eye = df.groupby(["rat_id", "eye", "day"], as_index=False).agg(agg_cols)
    # If RAG is not present, derive it.
    if "RAG" not in df_eye.columns:
        df_eye["RAG"] = pd.to_numeric(df_eye["age_pred"], errors="coerce") - pd.to_numeric(df_eye["age_true"], errors="coerce")

    # Pivot each field separately to avoid fragile mixed-column pivots.
    pair = None
    pivot_fields = [c for c in df_eye.columns if c not in ["rat_id", "eye", "day"]]
    for c in pivot_fields:
        p = df_eye.pivot_table(index=["rat_id", "day"], columns="eye", values=c, aggfunc="first")
        if not {"OD", "OS"} <= set(p.columns):
            # keep partial; final dropna will remove incomplete pairs
            pass
        p = p.rename(columns={"OD": f"{c}_OD", "OS": f"{c}_OS"}).reset_index()
        pair = p if pair is None else pair.merge(p, on=["rat_id", "day"], how="outer")

    if pair is None or pair.empty:
        print(f"[INTER-EYE] No pair rows created for {pred_csv_path.name}")
        return None

    # Keep complete OD/OS prediction pairs only.
    if {"age_pred_OD", "age_pred_OS"} <= set(pair.columns):
        pair = pair.dropna(subset=["age_pred_OD", "age_pred_OS"]).copy()
    else:
        print(f"[INTER-EYE] Missing paired age_pred columns for {pred_csv_path.name}")
        return None

    # Harmonize shared fields.
    for base in ("cohort", "group", "sex"):
        od = f"{base}_OD"
        os_ = f"{base}_OS"
        if od in pair.columns and os_ in pair.columns:
            pair[base] = pair[od].where(pair[od].notna(), pair[os_])

    pair["age_pred_inter_eye_signed_OD_minus_OS"] = pd.to_numeric(pair["age_pred_OD"], errors="coerce") - pd.to_numeric(pair["age_pred_OS"], errors="coerce")
    pair["age_pred_inter_eye_abs"] = pair["age_pred_inter_eye_signed_OD_minus_OS"].abs()
    if {"RAG_OD", "RAG_OS"} <= set(pair.columns):
        pair["RAG_inter_eye_signed_OD_minus_OS"] = pd.to_numeric(pair["RAG_OD"], errors="coerce") - pd.to_numeric(pair["RAG_OS"], errors="coerce")
        pair["RAG_inter_eye_abs"] = pair["RAG_inter_eye_signed_OD_minus_OS"].abs()

    if {"mil_bag_n_kept_OD", "mil_bag_n_kept_OS"} <= set(pair.columns):
        pair["mil_min_bag_n_kept"] = pd.concat(
            [pd.to_numeric(pair["mil_bag_n_kept_OD"], errors="coerce"), pd.to_numeric(pair["mil_bag_n_kept_OS"], errors="coerce")],
            axis=1,
        ).min(axis=1)
    if {"mil_low_conf_bag_OD", "mil_low_conf_bag_OS"} <= set(pair.columns):
        pair["mil_pair_low_conf_any"] = (
            pair["mil_low_conf_bag_OD"].fillna(False).astype(bool) |
            pair["mil_low_conf_bag_OS"].fillna(False).astype(bool)
        )

    # Stable sort.
    try:
        pair["day"] = pd.to_numeric(pair["day"], errors="coerce")
    except Exception:
        pass
    pair = pair.sort_values(["rat_id", "day"], kind="stable").reset_index(drop=True)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pair.to_csv(out_csv_path, index=False)
    print(f"[INTER-EYE] Saved paired OD/OS CSV: {out_csv_path} (N={len(pair)})")
    return pair


def compute_inter_eye_thresholds_from_control(control_pair_df: pd.DataFrame, q95: float = 0.95, q99: float = 0.99) -> Optional[dict]:
    if control_pair_df is None or control_pair_df.empty or "age_pred_inter_eye_abs" not in control_pair_df.columns:
        return None
    vals = pd.to_numeric(control_pair_df["age_pred_inter_eye_abs"], errors="coerce").dropna().astype(float)
    if vals.empty:
        return None
    return {
        "metric_col": "age_pred_inter_eye_abs",
        "q95": float(q95),
        "q99": float(q99),
        "thresh_q95": float(vals.quantile(q95)),
        "thresh_q99": float(vals.quantile(q99)),
        "n_control_pairs": int(len(vals)),
        "control_mean": float(vals.mean()),
        "control_median": float(vals.median()),
        "control_max": float(vals.max()),
    }


def annotate_inter_eye_reliability_flags(pair_df: pd.DataFrame, thresholds: Optional[dict]) -> pd.DataFrame:
    if pair_df is None or pair_df.empty or not thresholds:
        return pair_df
    out = pair_df.copy()
    metric_col = str(thresholds.get("metric_col", "age_pred_inter_eye_abs"))
    if metric_col not in out.columns:
        return out
    vals = pd.to_numeric(out[metric_col], errors="coerce")
    q95_th = float(thresholds["thresh_q95"])
    q99_th = float(thresholds["thresh_q99"])
    out["inter_eye_thresh_q95"] = q95_th
    out["inter_eye_thresh_q99"] = q99_th
    out["inter_eye_flag_unreliable_q95"] = vals > q95_th
    out["inter_eye_flag_extreme_q99"] = vals > q99_th
    tier = np.where(vals > q99_th, "extreme", np.where(vals > q95_th, "unreliable", "ok")).astype(object)
    tier[pd.isna(vals)] = "unknown"
    out["inter_eye_reliability_tier"] = tier
    return out


def run_post_control_inter_eye_analysis(args, control_pred_csv_path: Optional[Path], stress_pred_csv_path: Optional[Path]):
    """
    Optional post-step:
    - Build paired inter-eye CSVs from per-eye prediction CSVs
    - Add control-derived q95/q99 flags to paired CSVs
    - For MIL runs, optionally run matched-view control re-inference utility
    """
    if not bool(getattr(args, "post_control_inter_eye_analysis", False)):
        return
    if control_pred_csv_path is None or not Path(control_pred_csv_path).exists():
        print("[INTER-EYE] Post analysis skipped: control prediction CSV not available.")
        return

    control_pred_csv_path = Path(control_pred_csv_path)
    control_pair_path = _derive_inter_eye_csv_path(control_pred_csv_path)
    control_pair_df = build_inter_eye_pairs_csv(control_pred_csv_path, control_pair_path)
    thresholds = compute_inter_eye_thresholds_from_control(
        control_pair_df,
        q95=float(getattr(args, "post_inter_eye_q95", 0.95)),
        q99=float(getattr(args, "post_inter_eye_q99", 0.99)),
    )
    if thresholds and control_pair_df is not None:
        control_pair_df = annotate_inter_eye_reliability_flags(control_pair_df, thresholds)
        control_pair_df.to_csv(control_pair_path, index=False)
        print(
            "[INTER-EYE] Control thresholds "
            f"q95={thresholds['thresh_q95']:.2f}, q99={thresholds['thresh_q99']:.2f} "
            f"(N={thresholds['n_control_pairs']})"
        )
        thresh_json = control_pair_path.with_name(control_pair_path.stem + "_thresholds.json")
        with thresh_json.open("w") as f:
            json.dump(thresholds, f, indent=2)
        print(f"[INTER-EYE] Saved control thresholds: {thresh_json}")

    if stress_pred_csv_path is not None and Path(stress_pred_csv_path).exists():
        stress_pred_csv_path = Path(stress_pred_csv_path)
        stress_pair_path = _derive_inter_eye_csv_path(stress_pred_csv_path)
        stress_pair_df = build_inter_eye_pairs_csv(stress_pred_csv_path, stress_pair_path)
        if thresholds and stress_pair_df is not None:
            stress_pair_df = annotate_inter_eye_reliability_flags(stress_pair_df, thresholds)
            stress_pair_df.to_csv(stress_pair_path, index=False)
            n_unrel = int(stress_pair_df["inter_eye_flag_unreliable_q95"].fillna(False).astype(bool).sum()) if "inter_eye_flag_unreliable_q95" in stress_pair_df.columns else 0
            print(f"[INTER-EYE] Annotated stress paired CSV with control thresholds ({n_unrel}/{len(stress_pair_df)} q95-unreliable).")

    # Control matched-view MIL re-inference (analysis only)
    if not bool(getattr(args, "mil_attention", False)):
        return
    if not bool(getattr(args, "post_control_matched_view", False)):
        return

    ckpt_for_post = None
    if getattr(args, "load_lora", None) and Path(args.load_lora).exists():
        ckpt_for_post = Path(args.load_lora)
    elif getattr(args, "save_lora", None) and Path(args.save_lora).exists():
        ckpt_for_post = Path(args.save_lora)
    if ckpt_for_post is None:
        print("[INTER-EYE] Matched-view control analysis skipped: no checkpoint available (load/save_lora missing).")
        return
    if control_pair_df is None or control_pair_df.empty:
        print("[INTER-EYE] Matched-view control analysis skipped: no paired control CSV rows.")
        return

    cmd = [
        sys.executable,
        str(LORA_DIR / "control_matched_view_infer.py"),
        "--pair-csv", str(control_pair_path),
        "--load-lora", str(ckpt_for_post),
        "--csv", str(args.csv),
        "--backbone-ckpt", str(args.backbone_ckpt),
        "--device", str(args.device),
        "--img-size", str(int(args.img_size)),
        "--upsample-factor", str(int(args.upsample_factor)),
        "--lora-rank", str(int(args.lora_rank)),
        "--lora-alpha", str(float(args.lora_alpha)),
        "--lora-blocks", str(int(args.lora_blocks)),
        "--lora-dropout", str(float(args.lora_dropout)),
        "--mil-attn-dim", str(int(args.mil_attn_dim)),
        "--mil-hidden-dim", str(int(args.mil_hidden_dim)),
        "--min-common-images", str(int(getattr(args, "post_control_matched_view_min_common_images", 1))),
        "--q95", str(float(getattr(args, "post_inter_eye_q95", 0.95))),
        "--q99", str(float(getattr(args, "post_inter_eye_q99", 0.99))),
    ]
    if bool(getattr(args, "keep_spatial_tokens", False)):
        cmd.append("--keep-spatial-tokens")
    if bool(getattr(args, "input_pre_adapter", False)):
        cmd.extend(["--input-pre-adapter", "--input-pre-adapter-hidden", str(int(args.input_pre_adapter_hidden))])
    if bool(getattr(args, "post_control_matched_view_canonicalize_os", False)):
        cmd.append("--canonicalize-os-to-od")
    try:
        print("[INTER-EYE] Running control matched-view MIL analysis...")
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"[INTER-EYE] Matched-view control analysis failed: {e}")


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
from simple_baseline import SimpleXceptionAgePred, SimpleViTRandomAgePred  # noqa: E402
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
        "--cohorts",
        type=str,
        nargs="*",
        default=None,
        help="Override cohorts to keep (e.g., --cohorts 1 2). Default uses config COHORTS_TO_KEEP.",
    )
    p.add_argument(
        "--train-cohorts",
        type=str,
        nargs="*",
        default=None,
        help="Optional train/val cohort filter applied only to training pools (e.g., --train-cohorts 1 2).",
    )
    p.add_argument(
        "--test-cohorts",
        type=str,
        nargs="*",
        default=None,
        help="Optional held-out test cohort filter applied only to test pools (e.g., --test-cohorts 3).",
    )
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
    p.add_argument(
        "--progressive-lora-schedule",
        type=str,
        default=None,
        help=(
            "Progressively open LoRA blocks during training, format 'blocks:start_epoch,...' "
            "(e.g. '2:0,4:6,6:12'; 0-based or 1-based starts accepted). "
            "Model is built with max(args.lora_blocks, max schedule blocks)."
        ),
    )
    p.add_argument("--lora-alpha", type=float, default=LORA_ALPHA,
                   help="LoRA alpha (suggested: 16, 32, 64; often ~2x rank)")
    p.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT,
                   help="LoRA dropout (suggested: 0.05–0.30)")
    p.add_argument(
        "--retfound-full-finetune",
        action="store_true",
        help="RETFound only: unfreeze the entire RETFound backbone (use with --lora-blocks 0 for true full fine-tuning).",
    )
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
    p.set_defaults(mil_freeze_backbone=True)
    p.add_argument(
        "--mil-freeze-backbone",
        action="store_true",
        dest="mil_freeze_backbone",
        help="Freeze RETFound backbone in MIL mode (default: on; forces --lora-blocks 0).",
    )
    p.add_argument(
        "--no-mil-freeze-backbone",
        action="store_false",
        dest="mil_freeze_backbone",
        help="Allow LoRA adaptation in MIL mode (use with a small --lora-blocks, e.g. 2 or 4).",
    )
    p.add_argument("--mil-attn-dim", type=int, default=128, help="Hidden dim for MIL attention scorer.")
    p.add_argument("--mil-hidden-dim", type=int, default=256, help="Hidden dim for MIL regression MLP.")
    p.add_argument(
        "--heteroscedastic-regression",
        action="store_true",
        help="MIL-only: predict (mu, log_var) and train with Gaussian NLL (heteroscedastic regression).",
    )
    p.add_argument(
        "--hetero-logvar-reg-weight",
        type=float,
        default=1e-2,
        help="L2 regularization weight on heteroscedastic log_var (z-space) to reduce variance inflation/collapse (default: 1e-2).",
    )
    p.add_argument("--mil-view-balance", action="store_true",
                   help="Train-time MIL view-balanced bag sampling (cap instances per coarse view family).")
    p.add_argument("--mil-max-per-view", type=int, default=0,
                   help="Per-view cap for MIL train bags when --mil-view-balance is enabled (0 disables cap).")
    p.add_argument("--mil-bag-quality-filter", action="store_true",
                   help="Apply simple image-quality heuristics before MIL bag assembly (train+eval).")
    p.add_argument("--mil-min-bag-size", type=int, default=0,
                   help="Drop MIL bags with fewer than this many usable images after filtering (0 disables).")
    p.add_argument("--mil-infer-lowconf-bag-size", type=int, default=0,
                   help="Flag MIL predictions with kept bag size below this threshold in prediction CSVs (0 disables).")
    p.add_argument(
        "--ordinal-aux",
        action="store_true",
        help="Add ordinal auxiliary loss (CORAL-style) on top of MIL pooled features using age bins.",
    )
    p.add_argument(
        "--ordinal-aux-weight",
        type=float,
        default=0.1,
        help="Weight for ordinal auxiliary loss when --ordinal-aux is enabled (default: 0.1).",
    )
    p.add_argument(
        "--ordinal-bin-values",
        type=float,
        nargs="*",
        default=None,
        help="Optional explicit age bins for ordinal auxiliary loss (sorted unique values are used). Default: infer from train split ages.",
    )
    p.add_argument(
        "--ordinal-aux-hidden-dim",
        type=int,
        default=0,
        help="Hidden dim for ordinal auxiliary head MLP (0 => reuse regression head hidden dim).",
    )
    p.add_argument(
        "--regime-aux",
        action="store_true",
        help="MIL-only: add binary auxiliary head to classify coarse age regime (young vs old) from pooled bag features.",
    )
    p.add_argument(
        "--regime-aux-weight",
        type=float,
        default=0.0,
        help="Weight for binary regime auxiliary BCE loss when --regime-aux is enabled (default: 0 disables).",
    )
    p.add_argument(
        "--regime-aux-age-threshold",
        type=float,
        default=180.0,
        help="Age threshold for regime aux target: age > threshold => old regime (default: 180).",
    )
    p.add_argument(
        "--regime-aux-hidden-dim",
        type=int,
        default=0,
        help="Hidden dim for regime auxiliary head MLP (0 => reuse regression head hidden dim).",
    )
    p.add_argument(
        "--mil-control-inter-eye-lambda",
        type=float,
        default=0.0,
        help="Weight for MIL control-only inter-eye consistency regularizer |pred_OD-pred_OS| within batch (default: 0 disables).",
    )
    p.add_argument(
        "--mil-control-inter-eye-loss",
        type=str,
        default="l1",
        choices=["l1", "smoothl1"],
        help="Penalty type for MIL control inter-eye consistency regularizer.",
    )
    p.add_argument(
        "--mil-control-day90-weight",
        type=float,
        default=1.0,
        help="MIL-only train-time loss upweight for control day-90 bags (default: 1.0 = disabled).",
    )
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
    p.add_argument("--aug-level", type=str, default=AUG_LEVEL, choices=["mild", "low", "medium", "high"],
                   help="Augmentation strength for training transforms (mild aliases low)")
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
    p.add_argument("--model-type", type=str, default="retfound", choices=["retfound", "xception", "vit_random"], help="Model architecture to use")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Baseline models (xception/vit_random): freeze backbone and train head only.")
    p.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone",
                   help="Baseline models (xception/vit_random): allow backbone training.")
    p.set_defaults(freeze_backbone=False)
    p.add_argument("--baseline-pretrained", action="store_true", help="Use ImageNet-pretrained weights for the Xception baseline (requires cached weights)")
    p.add_argument(
        "--distill-teacher-ckpt",
        type=Path,
        default=None,
        help="Feature-distillation teacher checkpoint path (frozen RETFound teacher; Xception student only).",
    )
    p.add_argument(
        "--distill-alpha",
        type=float,
        default=0.0,
        help="Weight for feature distillation loss added to student regression loss (default: 0 disables).",
    )
    p.add_argument(
        "--distill-feature-only",
        action="store_true",
        help="Feature-only distillation mode (recommended). Output/prediction distillation is not implemented.",
    )
    p.add_argument(
        "--distill-proj-hidden-dim",
        type=int,
        default=512,
        help="Hidden dim for Xception->RETFound feature projection head used in distillation.",
    )
    p.add_argument(
        "--skip-stress-eval",
        action="store_true",
        help="Skip HLS/stress prediction export and metrics (useful for control-only experiments).",
    )
    p.add_argument("--save-saliency-dir", type=Path, default=None,
                   help="Optional dir to save saliency heatmaps (one PNG per image)")
    p.add_argument("--save-report-dir", type=Path, default=None,
                   help="Optional dir to save data stats, train curves, and val prediction plots/tables")
    p.add_argument(
        "--post-control-inter-eye-analysis",
        action="store_true",
        help="After prediction export, build paired inter-eye CSVs, add control-derived q95/q99 flags, and (MIL) run matched-view control analysis.",
    )
    p.add_argument(
        "--post-control-matched-view",
        action="store_true",
        help="With --post-control-inter-eye-analysis and --mil-attention, run matched-view control re-inference analysis.",
    )
    p.add_argument(
        "--post-control-matched-view-min-common-images",
        type=int,
        default=1,
        help="Minimum matched shared images per eye required for control matched-view re-inference (analysis only).",
    )
    p.add_argument(
        "--post-control-matched-view-canonicalize-os",
        action="store_true",
        help="Mirror OS images during control matched-view analysis (analysis only).",
    )
    p.add_argument(
        "--post-inter-eye-q95",
        type=float,
        default=0.95,
        help="Quantile for control-derived inter-eye 'unreliable' flag in paired CSVs (default: 0.95).",
    )
    p.add_argument(
        "--post-inter-eye-q99",
        type=float,
        default=0.99,
        help="Quantile for control-derived inter-eye 'extreme' flag in paired CSVs (default: 0.99).",
    )
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
    if args.cohorts is not None:
        args.cohorts = [str(c) for c in args.cohorts]
    if args.train_cohorts is not None:
        args.train_cohorts = [str(c) for c in args.train_cohorts]
    if args.test_cohorts is not None:
        args.test_cohorts = [str(c) for c in args.test_cohorts]
    return args


def build_model(args):
    if args.model_type == "xception":
        print("[MODEL] Using Xception baseline")
        model = SimpleXceptionAgePred(
            pretrained=args.baseline_pretrained,
            head_hidden_dim=256,
            head_dropout=args.lora_dropout,
        )
        if bool(getattr(args, "freeze_backbone", False)):
            for p in model.backbone.parameters():
                p.requires_grad = False
            print("[MODEL] Baseline backbone frozen (head-only training).")
    elif args.model_type == "vit_random":
        print("[MODEL] Using random-init ViT baseline")
        model = SimpleViTRandomAgePred(
            model_name="vit_base_patch16_224",
            head_hidden_dim=256,
            head_dropout=args.lora_dropout,
        )
        if bool(getattr(args, "freeze_backbone", False)):
            for p in model.backbone.parameters():
                p.requires_grad = False
            print("[MODEL] Baseline backbone frozen (head-only training).")
    else:
        print("[MODEL] Using RETFound + LoRA")
        if args.global_pool:
            print("[WARN] --global-pool is incompatible with the spatial regression head; forcing global_pool=False.")
        if not args.keep_spatial_tokens:
            print("[MODEL] Using CLS-only RETFound features for age regression (spatial tokens disabled).")
        if args.mil_attention:
            print(f"[MODEL] Attention-MIL enabled (attn_dim={args.mil_attn_dim}, hidden_dim={args.mil_hidden_dim})")
        if args.mil_attention and getattr(args, "heteroscedastic_regression", False):
            print("[MODEL] Heteroscedastic regression enabled (mu + log_var, Gaussian NLL).")
        if args.mil_attention and bool(getattr(args, "regime_aux", False)) and float(getattr(args, "regime_aux_weight", 0.0) or 0.0) > 0:
            print(
                "[MODEL] Regime auxiliary head enabled "
                f"(weight={float(getattr(args, 'regime_aux_weight', 0.0)):.4g}, "
                f"threshold>{float(getattr(args, 'regime_aux_age_threshold', 180.0)):.1f})"
            )
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
            heteroscedastic_regression=bool(getattr(args, "heteroscedastic_regression", False)),
            ordinal_num_bins=int(getattr(args, "ordinal_num_bins", 0) or 0),
            ordinal_aux_hidden_dim=int(getattr(args, "ordinal_aux_hidden_dim", 0) or 0),
            use_regime_aux=bool(getattr(args, "regime_aux", False) and float(getattr(args, "regime_aux_weight", 0.0) or 0.0) > 0),
            regime_aux_hidden_dim=int(getattr(args, "regime_aux_hidden_dim", 0) or 0),
        )
        if bool(getattr(args, "retfound_full_finetune", False)):
            if args.lora_blocks > 0:
                print("[WARN] --retfound-full-finetune with --lora-blocks>0 keeps LoRA-injected attention layers. For true full FT use --lora-blocks 0.")
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("[MODEL] RETFound full backbone fine-tuning enabled (all backbone params trainable).")
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
        if getattr(args, "mil_freeze_backbone", True):
            if args.model_type == "retfound" and args.lora_blocks != 0:
                print(f"[MIL] Freezing RETFound backbone in MIL baseline: forcing --lora-blocks 0 (was {args.lora_blocks}).")
                args.lora_blocks = 0
            else:
                print("[MIL] Freezing RETFound backbone in MIL baseline (--mil-freeze-backbone).")
        else:
            if args.lora_blocks > 0:
                print(f"[MIL] Allowing LoRA adaptation in MIL mode with --lora-blocks {args.lora_blocks}.")
            else:
                print("[MIL] --no-mil-freeze-backbone set, but --lora-blocks=0 so RETFound remains effectively frozen.")

    if args.distill_teacher_ckpt is not None or float(getattr(args, "distill_alpha", 0.0) or 0.0) > 0:
        if args.model_type != "xception":
            raise SystemExit("Feature distillation is currently supported only with --model-type xception.")
        if getattr(args, "mil_attention", False):
            raise SystemExit("Feature distillation is not implemented for --mil-attention; use the Xception non-MIL baseline.")
        if args.distill_teacher_ckpt is None:
            raise SystemExit("--distill-alpha > 0 requires --distill-teacher-ckpt.")
        if not bool(getattr(args, "distill_feature_only", False)):
            print("[DISTILL] Output/prediction distillation is not implemented; proceeding with feature-only distillation.")

    progressive_lora_schedule = None
    if getattr(args, "progressive_lora_schedule", None):
        if args.model_type != "retfound":
            raise SystemExit("--progressive-lora-schedule is supported only with --model-type retfound.")
        try:
            progressive_lora_schedule = parse_progressive_lora_schedule(args.progressive_lora_schedule)
        except Exception as e:
            raise SystemExit(f"Invalid --progressive-lora-schedule: {e}")
        if progressive_lora_schedule:
            max_sched_blocks = max(int(b) for _, b in progressive_lora_schedule)
            if max_sched_blocks > int(args.lora_blocks):
                print(
                    f"[LoRA] Increasing --lora-blocks from {args.lora_blocks} to {max_sched_blocks} "
                    "to satisfy --progressive-lora-schedule."
                )
                args.lora_blocks = int(max_sched_blocks)
            print(f"[LoRA] Progressive schedule (epoch->active_blocks): {progressive_lora_schedule}")

    def _uniq_cohorts(vals):
        if vals is None:
            return None
        seen = set()
        out = []
        for v in vals:
            s = str(v).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    base_cohorts = _uniq_cohorts(args.cohorts if args.cohorts is not None else COHORTS_TO_KEEP)
    train_cohorts_only = _uniq_cohorts(args.train_cohorts)
    test_cohorts_only = _uniq_cohorts(args.test_cohorts)
    if train_cohorts_only is not None or test_cohorts_only is not None:
        if train_cohorts_only is None:
            train_cohorts_only = list(base_cohorts) if base_cohorts is not None else None
        if test_cohorts_only is None:
            test_cohorts_only = list(base_cohorts) if base_cohorts is not None else None
        loader_cohorts_keep = _uniq_cohorts((train_cohorts_only or []) + (test_cohorts_only or []))
    else:
        loader_cohorts_keep = base_cohorts

    device = torch.device(args.device)
    print(f"[DEVICE] requested={args.device} | torch.cuda.is_available()={torch.cuda.is_available()} | using={device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA device requested but torch.cuda.is_available() is False. Check driver/NVML access (see torch.version.cuda).")
        print("[WARN] Falling back to CPU; training will be slow.")
    print(f"[DATA] day_whitelist={'ALL' if args.day_whitelist is None else args.day_whitelist}")
    print(f"[DATA] cohorts_to_keep={loader_cohorts_keep}")
    if train_cohorts_only is not None or test_cohorts_only is not None:
        print(f"[DATA] train_cohorts={train_cohorts_only} | test_cohorts={test_cohorts_only}")
    print(f"[AUG] photometric_aug={'OFF' if args.no_photometric_aug else 'ON'} | aug_level={args.aug_level}")
    if getattr(args, "mil_attention", False):
        print(
            "[MIL] bag_qc: "
            f"quality_filter={'ON' if bool(getattr(args, 'mil_bag_quality_filter', False)) else 'OFF'} | "
            f"view_balance={'ON' if bool(getattr(args, 'mil_view_balance', False)) else 'OFF'} | "
            f"max_per_view={int(getattr(args, 'mil_max_per_view', 0) or 0)} | "
            f"min_bag_size={int(getattr(args, 'mil_min_bag_size', 0) or 0)} | "
            f"infer_lowconf_bag_size={int(getattr(args, 'mil_infer_lowconf_bag_size', 0) or 0)}"
        )
        if bool(getattr(args, "heteroscedastic_regression", False)):
            print("[MIL] heteroscedastic regression: ON (train loss = Gaussian NLL, eval metric loss remains SmoothL1)")
    if getattr(args, "mil_attention", False) and float(getattr(args, "mil_control_inter_eye_lambda", 0.0)) > 0:
        print(
            "[MIL] control inter-eye consistency: "
            f"lambda={float(args.mil_control_inter_eye_lambda):.4g}, "
            f"loss={getattr(args, 'mil_control_inter_eye_loss', 'l1')}"
        )
    if getattr(args, "mil_attention", False) and float(getattr(args, "mil_control_day90_weight", 1.0) or 1.0) > 1.0:
        print(
            "[MIL] control day-90 loss upweight: "
            f"x{float(getattr(args, 'mil_control_day90_weight', 1.0)):.4g}"
        )
    if bool(getattr(args, "ordinal_aux", False)):
        print(
            "[ORD] requested ordinal auxiliary loss: "
            f"weight={float(getattr(args, 'ordinal_aux_weight', 0.0)):.4g}"
        )
    if bool(getattr(args, "regime_aux", False)) and float(getattr(args, "regime_aux_weight", 0.0) or 0.0) > 0:
        print(
            "[REGIME] requested binary regime auxiliary loss: "
            f"weight={float(getattr(args, 'regime_aux_weight', 0.0)):.4g}, "
            f"threshold>{float(getattr(args, 'regime_aux_age_threshold', 180.0)):.1f}"
        )
    if args.distill_teacher_ckpt is not None and float(getattr(args, "distill_alpha", 0.0) or 0.0) > 0:
        print(
            "[DISTILL] feature-only distillation: "
            f"teacher={args.distill_teacher_ckpt} | alpha={float(args.distill_alpha):.4g} | "
            f"proj_hidden={int(getattr(args, 'distill_proj_hidden_dim', 512) or 512)}"
        )

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
            cohorts_to_keep=loader_cohorts_keep,
            train_cohorts_to_keep=train_cohorts_only,
            test_cohorts_to_keep=test_cohorts_only,
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
            mil_view_balance=args.mil_view_balance,
            mil_max_per_view=args.mil_max_per_view,
            mil_min_bag_size=args.mil_min_bag_size,
            mil_quality_filter=args.mil_bag_quality_filter,
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
            mil_view_balance=args.mil_view_balance,
            mil_max_per_view=args.mil_max_per_view,
            mil_min_bag_size=args.mil_min_bag_size,
            mil_quality_filter=args.mil_bag_quality_filter,
        )
    else:
        train_df, val_df, ctrl_test_df, test_df, (train_loader, val_loader, test_loader, ctrl_test_loader) = prepare_data(
            csv_path=args.csv,
            image_types=IMAGE_TYPES,
            day_whitelist=args.day_whitelist,
            test_image_types=args.test_image_types,
            test_single_image=args.test_single_image,
            include_recovery_days=False,
            cohorts_to_keep=loader_cohorts_keep,
            train_cohorts_to_keep=train_cohorts_only,
            test_cohorts_to_keep=test_cohorts_only,
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
            mil_view_balance=args.mil_view_balance,
            mil_max_per_view=args.mil_max_per_view,
            mil_min_bag_size=args.mil_min_bag_size,
            mil_quality_filter=args.mil_bag_quality_filter,
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

    args.ordinal_bin_values_resolved = resolve_ordinal_age_bins(train_df, args)
    args.ordinal_num_bins = len(args.ordinal_bin_values_resolved) if args.ordinal_bin_values_resolved else 0
    if getattr(args, "ordinal_aux", False) and args.ordinal_num_bins >= 2:
        bins_preview = args.ordinal_bin_values_resolved
        if len(bins_preview) > 12:
            preview_txt = f"{bins_preview[:6]} ... {bins_preview[-3:]}"
        else:
            preview_txt = str(bins_preview)
        print(
            "[ORD] active: "
            f"num_bins={args.ordinal_num_bins}, weight={float(args.ordinal_aux_weight):.4g}, "
            f"bins={preview_txt}"
        )

    if bool(getattr(args, "heteroscedastic_regression", False)):
        # Normalize heteroscedastic MIL targets using the train-split bag-level age distribution.
        # This keeps NLL numerically stable and makes the learned log_var operate in z-space.
        if getattr(args, "mil_attention", False) and all(c in train_df.columns for c in ("rat_id", "eye", "day", "AGE")):
            age_series = (
                train_df.groupby(["rat_id", "eye", "day"], dropna=False)["AGE"]
                .mean()
                .astype(float)
            )
        elif "AGE" in train_df.columns:
            age_series = train_df["AGE"].astype(float)
        else:
            age_series = pd.Series(dtype=float)
        mean_age = float(age_series.mean()) if len(age_series) else 0.0
        std_age = float(age_series.std(ddof=0)) if len(age_series) else 1.0
        if (not np.isfinite(std_age)) or std_age <= 1e-6:
            std_age = 1.0
        args.hetero_target_mean = mean_age
        args.hetero_target_std = std_age
        print(
            "[MIL][hetero] target z-score stats (train split): "
            f"mean={mean_age:.3f}, std={std_age:.3f}, "
            f"logvar_reg_weight={float(getattr(args, 'hetero_logvar_reg_weight', 0.0)):.4g}"
        )

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
        eval_val_df = val_df

        # `--test-image-types` is already applied inside `prepare_data` for test/ctrl_test.
        # Apply the same filter to the control eval fallback path (val controls) so REGAVG-only
        # ablations are truly matched on both control and HLS reporting.
        if args.test_image_types:
            test_image_types_set = set(args.test_image_types)
            before_val = len(eval_val_df)
            eval_val_df = eval_val_df[eval_val_df["image_type"].isin(test_image_types_set)].copy()
            print(
                f"[DATA] control eval fallback image types filter {sorted(test_image_types_set)}: "
                f"val {before_val}->{len(eval_val_df)}"
            )

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
                mil_view_balance=args.mil_view_balance,
                mil_max_per_view=args.mil_max_per_view,
                mil_min_bag_size=args.mil_min_bag_size,
                mil_quality_filter=args.mil_bag_quality_filter,
            )
            return val_like_loader

        ctrl_eval_df = filter_df_by_days(ctrl_test_df, args.control_eval_days, "ctrl_eval_holdout")
        val_ctrl_eval_df = filter_df_by_days(eval_val_df, args.control_eval_days, "ctrl_eval_val_fallback")
        control_holdout_loader = _build_eval_loader_from_df(ctrl_eval_df)
        control_val_fallback_loader = _build_eval_loader_from_df(val_ctrl_eval_df)

    model = build_model(args).to(device)
    if bool(getattr(args, "heteroscedastic_regression", False)):
        setattr(model, "hetero_target_mean", float(getattr(args, "hetero_target_mean", 0.0) or 0.0))
        setattr(model, "hetero_target_std", float(getattr(args, "hetero_target_std", 1.0) or 1.0))
    trainer = Trainer(model, device)
    if (
        args.model_type == "xception"
        and args.distill_teacher_ckpt is not None
        and float(getattr(args, "distill_alpha", 0.0) or 0.0) > 0
    ):
        teacher = RETFoundLoRAAgePred(
            ckpt_path=args.distill_teacher_ckpt,
            img_size=args.img_size,
            global_pool=False,
            lora_rank=max(1, int(getattr(args, "lora_rank", 8) or 8)),
            lora_alpha=float(getattr(args, "lora_alpha", 16.0) or 16.0),
            lora_blocks=0,
            lora_dropout=0.0,
            head_hidden_dim=256,
            head_dropout=0.0,
            upsample_factor=None,
            keep_spatial_tokens=False,
            use_pre_adapter=False,
            pre_adapter_hidden_dim=16,
            use_mil_attention=False,
            mil_attn_dim=128,
            mil_hidden_dim=256,
            heteroscedastic_regression=False,
            ordinal_num_bins=0,
            ordinal_aux_hidden_dim=0,
            use_regime_aux=False,
            regime_aux_hidden_dim=0,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        teacher_feat_dim = int(getattr(getattr(teacher, "backbone", None), "embed_dim", 0) or 0)
        if teacher_feat_dim <= 0:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, int(args.img_size), int(args.img_size), device=device)
                teacher_feat_dim = int(teacher.extract_image_features(dummy).shape[-1])
        student_feat_dim = int(getattr(model, "backbone_channels", 0) or 0)
        if student_feat_dim <= 0 and hasattr(getattr(model, "backbone", None), "num_features"):
            student_feat_dim = int(model.backbone.num_features)
        if student_feat_dim <= 0:
            raise SystemExit("Could not infer Xception student feature dimension for distillation.")

        proj_hidden = int(max(16, getattr(args, "distill_proj_hidden_dim", 512) or 512))
        model.distill_proj = torch.nn.Sequential(
            torch.nn.Linear(student_feat_dim, proj_hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(proj_hidden, teacher_feat_dim),
        ).to(device)
        trainer.distill_teacher = teacher
        trainer.distill_alpha = float(args.distill_alpha)
        print(
            "[DISTILL] Teacher/student feature dims: "
            f"student={student_feat_dim} -> teacher={teacher_feat_dim} (proj hidden={proj_hidden})"
        )

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
            if bool(getattr(args, "heteroscedastic_regression", False)) and isinstance(ckpt.get("hetero_target_norm"), dict):
                norm = ckpt.get("hetero_target_norm") or {}
                try:
                    args.hetero_target_mean = float(norm.get("mean", getattr(args, "hetero_target_mean", 0.0)))
                    args.hetero_target_std = float(norm.get("std", getattr(args, "hetero_target_std", 1.0)))
                    setattr(model, "hetero_target_mean", float(args.hetero_target_mean))
                    setattr(model, "hetero_target_std", float(args.hetero_target_std))
                    print(
                        "[LOAD] Loaded heteroscedastic target norm from checkpoint: "
                        f"mean={args.hetero_target_mean:.3f}, std={args.hetero_target_std:.3f}"
                    )
                except Exception:
                    print("[LOAD] Warning: invalid hetero_target_norm in checkpoint; using current run split stats.")
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
            if args.model_type == "retfound" and hasattr(model, "ordinal_head") and model.ordinal_head is not None:
                if "ordinal_head" in ckpt:
                    model.ordinal_head.load_state_dict(ckpt["ordinal_head"], strict=False)
                else:
                    print("[LOAD] Checkpoint missing ordinal_head weights (ordinal aux enabled in current model).")
            if args.model_type == "retfound" and hasattr(model, "regime_head") and model.regime_head is not None:
                if "regime_head" in ckpt:
                    model.regime_head.load_state_dict(ckpt["regime_head"], strict=False)
                else:
                    print("[LOAD] Checkpoint missing regime_head weights (regime aux enabled in current model).")
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
        current_active_lora_blocks = None

        for epoch in range(1, args.epochs + 1):
            if progressive_lora_schedule and hasattr(model, "set_active_lora_blocks"):
                target_active = active_lora_blocks_for_epoch(epoch, progressive_lora_schedule, args.lora_blocks)
                if current_active_lora_blocks != int(target_active):
                    applied = int(model.set_active_lora_blocks(int(target_active)))
                    current_active_lora_blocks = applied
                    print(f"[LoRA] Epoch {epoch}: active_lora_blocks={applied}")
            train_loss = trainer.train_one_epoch(train_loader, optimizer, args) if train_loader else float("nan")
            val_loss = trainer.evaluate(val_loader, args) if val_loader else float("nan")
            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else float("nan")
            metrics_log.append({
                "epoch": epoch,
                "train_L1": float(train_loss),
                "val_L1": float(val_loss),
                "lr": float(current_lr),
                "active_lora_blocks": float(current_active_lora_blocks) if current_active_lora_blocks is not None else float(args.lora_blocks),
            })
            train_metric_name = "train_NLL" if bool(getattr(args, "heteroscedastic_regression", False)) else "train_L1"
            print(f"[EPOCH {epoch}] {train_metric_name}={train_loss:.4f} val_L1={val_loss:.4f}")
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
            if hasattr(model, "ordinal_head") and model.ordinal_head is not None:
                save_dict["ordinal_head"] = model.ordinal_head.state_dict()
                if getattr(args, "ordinal_bin_values_resolved", None):
                    save_dict["ordinal_bin_values"] = [float(v) for v in args.ordinal_bin_values_resolved]
            if hasattr(model, "regime_head") and model.regime_head is not None:
                save_dict["regime_head"] = model.regime_head.state_dict()
            if bool(getattr(args, "heteroscedastic_regression", False)):
                save_dict["hetero_target_norm"] = {
                    "mean": float(getattr(args, "hetero_target_mean", 0.0) or 0.0),
                    "std": float(getattr(args, "hetero_target_std", 1.0) or 1.0),
                }
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
    stress_csv_path = None
    if bool(getattr(args, "skip_stress_eval", False)):
        print("[PRED] Skipping HLS/Recovery/High_CO2 test set (--skip-stress-eval).")
    else:
        print("[PRED] Running HLS/Recovery/High_CO2 test set…")
        trainer.predict_to_csv(
            test_loader,
            stress_csv_name,
            args,
            device,
            correction=correction,
            save_saliency_dir=args.save_saliency_dir if args.save_saliency_dir else None,
        )
        stress_csv_path = pred_dir / stress_csv_name

    # Optional control-first inter-eye post analysis (paired CSVs + reliability flags + matched-view MIL analysis).
    try:
        run_post_control_inter_eye_analysis(args, control_csv_path, stress_csv_path)
    except Exception as e:
        print(f"[INTER-EYE] Post analysis failed: {e}")

    if report_dir:
        metrics = []
        ctrl_path = control_csv_path or (pred_dir / f"control_test_results{full_suffix}.csv")
        stress_path = stress_csv_path
        for label, p in (("control", ctrl_path), ("stress", stress_path)):
            if p is None:
                continue
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
