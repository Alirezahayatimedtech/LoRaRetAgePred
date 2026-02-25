#!/usr/bin/env python3
"""Aggregate K-fold CV outputs from RETFoundLoRA run directories.

Expected input layout: any directory tree containing `metrics_summary.csv`
files (e.g., `outputs/core/exp01_ctrl_cv/fold0/...`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from paper_common import (  # noqa: E402
    collect_run_dirs,
    discover_run_files,
    flatten_metric_summary_rows,
    groupwise_prediction_metrics,
    load_prediction_csv,
    read_metrics_summary,
    safe_mkdir_for_file,
)


def summarize_metrics(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for split, sub in df.groupby("split", dropna=False):
        row = {"split": split, "n_folds": int(sub["fold"].notna().sum()) if "fold" in sub.columns else int(len(sub))}
        for m in metrics:
            if m not in sub.columns:
                continue
            vals = pd.to_numeric(sub[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
            row[f"{m}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{m}_max"] = float(vals.max()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def collect_fold_breakdowns(run_dirs: List[Path], split_name: str) -> pd.DataFrame:
    csv_attr = "control_pred_csv" if split_name == "control" else "stress_pred_csv"
    frames = []
    for run_dir in run_dirs:
        rf = discover_run_files(run_dir)
        pred_csv = getattr(rf, csv_attr)
        if pred_csv is None or not pred_csv.exists():
            continue
        df = load_prediction_csv(pred_csv)
        breakdown = groupwise_prediction_metrics(df, ["cohort", "day"], include_inter_eye=(split_name == "control"))
        breakdown["run_dir"] = str(run_dir)
        breakdown["run_name"] = run_dir.name
        frames.append(breakdown)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_breakdown(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    numeric_cols = [c for c in df.columns if c not in {"cohort", "day", "run_dir", "run_name"} and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return
    grouped = df.groupby(["cohort", "day"], dropna=False)[numeric_cols].agg(["mean", "std", "count"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    grouped = grouped.reset_index()
    safe_mkdir_for_file(out_path)
    grouped.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate CV metrics across fold outputs")
    ap.add_argument("--input-dir", type=Path, nargs="+", required=True, help="Root dir(s) containing fold run outputs")
    ap.add_argument("--output", type=Path, required=True, help="Output CSV summary path")
    ap.add_argument("--metrics", nargs="+", default=["mae", "rmse", "r2", "pearson_r"], help="Metric columns from metrics_summary.csv to aggregate")
    ap.add_argument("--emit-breakdowns", action="store_true", help="Also aggregate per-cohort/per-day prediction metrics and inter-eye summaries")
    args = ap.parse_args()

    run_dirs = collect_run_dirs(args.input_dir)
    if not run_dirs:
        raise SystemExit("No run directories with metrics_summary.csv found.")

    metric_rows = []
    for rd in run_dirs:
        rf = discover_run_files(rd)
        if rf.metrics_csv is None:
            continue
        mdf = read_metrics_summary(rf.metrics_csv)
        metric_rows.append(flatten_metric_summary_rows(rd, mdf))
    if not metric_rows:
        raise SystemExit("No readable metrics_summary.csv files found.")

    all_metrics = pd.concat(metric_rows, ignore_index=True)
    summary = summarize_metrics(all_metrics, args.metrics)
    safe_mkdir_for_file(args.output)
    summary.to_csv(args.output, index=False)
    print(f"[CV] Saved fold summary to {args.output}")

    details_path = args.output.with_name(args.output.stem + "_fold_rows.csv")
    all_metrics.to_csv(details_path, index=False)
    print(f"[CV] Saved per-fold metric rows to {details_path}")

    if args.emit_breakdowns:
        ctrl = collect_fold_breakdowns(run_dirs, "control")
        stress = collect_fold_breakdowns(run_dirs, "stress")
        if not ctrl.empty:
            ctrl_raw = args.output.with_name(args.output.stem + "_control_cohort_day_fold_rows.csv")
            ctrl.to_csv(ctrl_raw, index=False)
            aggregate_breakdown(ctrl, args.output.with_name(args.output.stem + "_control_cohort_day_summary.csv"))
            print(f"[CV] Saved control cohort/day breakdown rows to {ctrl_raw}")
        if not stress.empty:
            stress_raw = args.output.with_name(args.output.stem + "_stress_cohort_day_fold_rows.csv")
            stress.to_csv(stress_raw, index=False)
            aggregate_breakdown(stress, args.output.with_name(args.output.stem + "_stress_cohort_day_summary.csv"))
            print(f"[CV] Saved stress cohort/day breakdown rows to {stress_raw}")


if __name__ == "__main__":
    main()

