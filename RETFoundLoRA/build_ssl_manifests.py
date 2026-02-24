"""Build strict/transductive unlabeled OCT manifests for self-supervised adaptation.

This script reuses the current supervised split logic (`prepare_data`) so the resulting
"strict" SSL manifest is guaranteed to avoid rat-level leakage with respect to the
supervised train/val/test partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

from config import CSV_PATH, IMAGE_TYPES, COHORTS_TO_KEEP
from preprocess_age_lora import prepare_data


def _normalize_group_set(vals: Optional[Iterable[str]]) -> Set[str]:
    if not vals:
        return set()
    return {str(v).strip() for v in vals if str(v).strip()}


def _df_with_split_label(df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["split_role"])
    out = df.copy()
    out["split_role"] = split_label
    return out


def _safe_group_col(df: pd.DataFrame) -> str:
    return "group_norm" if "group_norm" in df.columns else ("group" if "group" in df.columns else "")


def _safe_day_col(df: pd.DataFrame) -> str:
    return "day" if "day" in df.columns else ("DAY" if "DAY" in df.columns else "")


def _safe_age_col(df: pd.DataFrame) -> str:
    return "AGE" if "AGE" in df.columns else ("age_true" if "age_true" in df.columns else "")


def _manifest_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "image_path",
        "rat_id",
        "eye",
        "sex",
        "cohort",
        "group_norm",
        "group",
        "day",
        "AGE",
        "image_type",
        "material_type",
        "sample_id",
        "split_role",
    ]
    return [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]


def _summarize_manifest(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty:
        return {"rows": 0, "rats": 0}
    group_col = _safe_group_col(df)
    day_col = _safe_day_col(df)
    age_col = _safe_age_col(df)
    summary: Dict[str, object] = {
        "rows": int(len(df)),
        "rats": int(df["rat_id"].nunique()) if "rat_id" in df.columns else 0,
        "eyes": int(df[["rat_id", "eye"]].drop_duplicates().shape[0]) if {"rat_id", "eye"}.issubset(df.columns) else 0,
        "bags_rat_eye_day": int(df[["rat_id", "eye", "day"]].drop_duplicates().shape[0]) if {"rat_id", "eye", "day"}.issubset(df.columns) else 0,
    }
    if group_col:
        summary["rows_by_group"] = {str(k): int(v) for k, v in df.groupby(group_col).size().to_dict().items()}
        summary["rats_by_group"] = {str(k): int(v) for k, v in df.groupby(group_col)["rat_id"].nunique().to_dict().items()}
    if "cohort" in df.columns:
        summary["rows_by_cohort"] = {str(k): int(v) for k, v in df.groupby("cohort").size().to_dict().items()}
        summary["rats_by_cohort"] = {str(k): int(v) for k, v in df.groupby("cohort")["rat_id"].nunique().to_dict().items()}
    if day_col:
        vc = df[day_col].value_counts().sort_index()
        summary["rows_by_day"] = {str(k): int(v) for k, v in vc.to_dict().items()}
    if age_col:
        ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
        if len(ages):
            summary["age_min"] = float(ages.min())
            summary["age_median"] = float(ages.median())
            summary["age_max"] = float(ages.max())
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build strict/transductive SSL manifests from RETFoundLoRA supervised splits.")
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    p.add_argument("--image-types", nargs="*", default=list(IMAGE_TYPES))
    p.add_argument("--all-ages", action="store_true", help="Use all days (overrides --day-whitelist).")
    p.add_argument("--day-whitelist", type=int, nargs="*", default=None)
    p.add_argument("--cohorts", nargs="*", default=list(COHORTS_TO_KEEP))
    p.add_argument("--train-groups", nargs="*", default=["Controls"])
    p.add_argument("--test-groups", nargs="*", default=["HLS (U)"])
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cohort-stratified-split", action="store_true")
    p.add_argument("--ssl-groups", nargs="*", default=["Controls", "HLS (U)"], help="Groups to include in SSL manifests.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/ssl_manifests"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    day_whitelist = None if args.all_ages else (list(args.day_whitelist) if args.day_whitelist is not None else None)
    ssl_groups = _normalize_group_set(args.ssl_groups)

    # Build the exact supervised split using the same pipeline logic, but keep all image types
    # in returned split dataframes so SSL manifests can include every OCT image from eligible rats.
    train_df, val_df, ctrl_test_df, test_df, _ = prepare_data(
        csv_path=args.csv,
        image_types=list(args.image_types),
        day_whitelist=day_whitelist,
        test_image_types=None,
        test_single_image=False,
        include_recovery_days=False,
        cohorts_to_keep=(list(args.cohorts) if args.cohorts else None),
        exclude_recovery_paths=False,
        train_groups=list(args.train_groups),
        test_groups=list(args.test_groups),
        val_split=float(args.val_split),
        test_split=float(args.test_split),
        baseline_test_split=0.0,
        holdout_day=None,
        holdout_test_only=False,
        subset_size=None,
        subset_fraction=None,
        img_size=224,
        batch_size=1,
        num_workers=0,
        seed=int(args.seed),
        right_eye_only=False,
        aug_level="mild",
        cohort_stratified_split=bool(args.cohort_stratified_split),
        enable_photometric_aug=False,
        mil_attention=False,
        mil_view_balance=False,
        mil_max_per_view=0,
        mil_min_bag_size=0,
        mil_quality_filter=False,
    )

    # Union of rows considered by the supervised protocol (after split and leakage cleanup).
    split_parts = [
        _df_with_split_label(train_df, "supervised_train"),
        _df_with_split_label(val_df, "supervised_val"),
        _df_with_split_label(ctrl_test_df, "supervised_ctrl_eval"),
        _df_with_split_label(test_df, "supervised_test"),
    ]
    full = pd.concat([d for d in split_parts if d is not None and not d.empty], ignore_index=True)
    if "image_path" in full.columns:
        full = full.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    train_rats = set(train_df["rat_id"].dropna().astype(str).tolist())
    val_rats = set(val_df["rat_id"].dropna().astype(str).tolist())
    ctrl_eval_rats = set(ctrl_test_df["rat_id"].dropna().astype(str).tolist())
    test_rats = set(test_df["rat_id"].dropna().astype(str).tolist())
    eval_rats = val_rats | ctrl_eval_rats | test_rats
    overlap = train_rats & eval_rats
    if overlap:
        raise RuntimeError(f"Strict SSL train-rat leakage detected ({len(overlap)} overlapping rats).")

    group_col = _safe_group_col(full)
    strict_df = full[full["rat_id"].astype(str).isin(train_rats)].copy()
    if ssl_groups and group_col:
        strict_df = strict_df[strict_df[group_col].astype(str).isin(ssl_groups)].copy()

    transductive_df = full.copy()
    if ssl_groups and group_col:
        transductive_df = transductive_df[transductive_df[group_col].astype(str).isin(ssl_groups)].copy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    strict_path = args.out_dir / "ssl_strict_train_rats_unlabeled.csv"
    trans_path = args.out_dir / "ssl_transductive_all_rats_unlabeled.csv"
    split_rats_path = args.out_dir / "supervised_split_rats.csv"
    summary_path = args.out_dir / "ssl_manifest_summary.json"

    strict_df = strict_df[_manifest_columns(strict_df)]
    transductive_df = transductive_df[_manifest_columns(transductive_df)]

    strict_df.to_csv(strict_path, index=False)
    transductive_df.to_csv(trans_path, index=False)

    rat_rows = []
    for split_label, rats in [
        ("supervised_train", train_rats),
        ("supervised_val", val_rats),
        ("supervised_ctrl_eval", ctrl_eval_rats),
        ("supervised_test", test_rats),
    ]:
        for r in sorted(rats):
            rat_rows.append({"rat_id": str(r), "split_role": split_label})
    pd.DataFrame(rat_rows).to_csv(split_rats_path, index=False)

    strict_group_counts = {}
    trans_group_counts = {}
    if group_col and not strict_df.empty:
        strict_group_counts = {str(k): int(v) for k, v in strict_df.groupby(group_col).size().to_dict().items()}
    if group_col and not transductive_df.empty:
        trans_group_counts = {str(k): int(v) for k, v in transductive_df.groupby(group_col).size().to_dict().items()}

    summary = {
        "protocol": {
            "csv": str(args.csv),
            "image_types": list(args.image_types),
            "day_whitelist": "ALL" if day_whitelist is None else list(day_whitelist),
            "cohorts": list(args.cohorts) if args.cohorts else None,
            "train_groups": list(args.train_groups),
            "test_groups": list(args.test_groups),
            "ssl_groups": sorted(ssl_groups),
            "val_split": float(args.val_split),
            "test_split": float(args.test_split),
            "seed": int(args.seed),
            "cohort_stratified_split": bool(args.cohort_stratified_split),
            "strict_rule": "Only rows whose rat_id is in supervised TRAIN rats (no VAL/CTRL_EVAL/TEST rats).",
            "transductive_rule": "All rows in supervised universe (train/val/ctrl_eval/test rats) after supervised filtering and leakage cleanup.",
        },
        "supervised_split": {
            "train_rats": len(train_rats),
            "val_rats": len(val_rats),
            "ctrl_eval_rats": len(ctrl_eval_rats),
            "test_rats": len(test_rats),
            "strict_train_vs_eval_overlap_rats": len(overlap),
        },
        "strict_manifest": _summarize_manifest(strict_df),
        "transductive_manifest": _summarize_manifest(transductive_df),
        "strict_rows_by_group": strict_group_counts,
        "transductive_rows_by_group": trans_group_counts,
    }

    # Helpful note for the common current protocol: HLS are test-group rats, so strict SSL will
    # typically exclude HLS entirely.
    if "HLS (U)" in ssl_groups:
        strict_hls_rows = int(strict_group_counts.get("HLS (U)", 0))
        trans_hls_rows = int(trans_group_counts.get("HLS (U)", 0))
        if strict_hls_rows == 0 and trans_hls_rows > 0:
            summary["note"] = (
                "Strict SSL manifest contains no HLS rows under the current supervised protocol because "
                "HLS rats are held in supervised_test; including them in SSL would be transductive."
            )

    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[SSL] Saved strict manifest: {strict_path} (rows={len(strict_df)})")
    print(f"[SSL] Saved transductive manifest: {trans_path} (rows={len(transductive_df)})")
    print(f"[SSL] Saved split-rat audit: {split_rats_path}")
    print(f"[SSL] Saved summary: {summary_path}")
    print("[SSL] Strict rows by group:", strict_group_counts)
    print("[SSL] Transductive rows by group:", trans_group_counts)
    if summary.get("note"):
        print("[SSL][NOTE]", summary["note"])


if __name__ == "__main__":
    main()

