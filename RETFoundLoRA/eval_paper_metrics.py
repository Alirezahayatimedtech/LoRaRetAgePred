#!/usr/bin/env python3
"""
Paper-style evaluation utilities focused on Controls only:
1) Inter-eye consistency (MAD between OD/OS for Controls).

Usage:
  python eval_paper_metrics.py --pred-csv lora_age_outputs/control_test_results.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from config import OUTPUT_ROOT


def load_predictions(paths) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Predictions CSV not found: {p}")
        df = pd.read_csv(p)
        required = {"rat_id", "eye", "group", "day", "age_true", "age_pred"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV {p} missing columns: {missing}")
        df["group"] = df["group"].astype(str).str.strip()
        df["__source"] = str(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def inter_eye_mad(df: pd.DataFrame, groups=None) -> float:
    if groups:
        df = df[df["group"].isin(groups)].copy()
    if df.empty:
        return float("nan")
    pivot = df.pivot_table(
        index=["rat_id", "day"],
        columns="eye",
        values="age_pred",
        aggfunc="mean",
    )
    if not {"OD", "OS"} <= set(pivot.columns):
        return float("nan")
    diffs = (pivot["OD"] - pivot["OS"]).abs().dropna()
    return float(diffs.mean()) if len(diffs) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", type=Path, nargs="+", required=True,
                    help="One or more Predictions CSVs from run.py (they will be concatenated)")
    ap.add_argument("--mad-groups", type=str, nargs="*", default=["Controls"],
                    help="Groups to include for inter-eye MAD (default: Controls)")
    ap.add_argument("--mad-out", type=Path, default=OUTPUT_ROOT / "mad_summary.csv",
                    help="CSV to save MAD for Controls")
    ap.add_argument("--mad-by-day-out", type=Path, default=OUTPUT_ROOT / "mad_by_day.csv",
                    help="CSV to save per-day MAD and pair counts for Controls")
    args = ap.parse_args()

    df = load_predictions(args.pred_csv)

    mad = inter_eye_mad(df, groups=args.mad_groups)
    grp_label = ",".join(args.mad_groups) if args.mad_groups else "ALL"
    print(f"[METRIC] Inter-eye MAD ({grp_label}): {mad:.3f} days" if mad == mad else "[METRIC] Inter-eye MAD unavailable")

    # Optional CSV summary for controls
    if args.mad_out:
        controls_mad = inter_eye_mad(df, groups=["Controls"])
        out_df = pd.DataFrame([{"group_set": "Controls", "mad_days": controls_mad}])
        args.mad_out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.mad_out, index=False)
        print(f"[MAD] Saved MAD summary to {args.mad_out}")

    # Optional per-day MAD breakdown for Controls
    if args.mad_by_day_out:
        controls = df[df["group"] == "Controls"].copy()
        pivot = controls.pivot_table(
            index=["rat_id", "day"],
            columns="eye",
            values="age_pred",
            aggfunc="mean",
        )
        if {"OD", "OS"} <= set(pivot.columns):
            pivot = pivot.dropna(subset=["OD", "OS"])
            pivot["abs_diff"] = (pivot["OD"] - pivot["OS"]).abs()
            by_day = pivot.reset_index().groupby("day").agg(
                mad_days=("abs_diff", "mean"),
                n_pairs=("abs_diff", "count"),
            ).reset_index()
            args.mad_by_day_out.parent.mkdir(parents=True, exist_ok=True)
            by_day.to_csv(args.mad_by_day_out, index=False)
            print(f"[MAD] Saved per-day MAD to {args.mad_by_day_out}")


if __name__ == "__main__":
    main()
