#!/usr/bin/env python3
"""Aggregate K-fold CV outputs from RETFoundLoRA run directories.

Expected input layout: any directory tree containing `metrics_summary.csv`
files (e.g., `outputs/core/exp01_ctrl_cv/fold0/...`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from paper_common import (  # noqa: E402
    collect_run_dirs,
    flatten_metric_summary_rows,
    groupwise_prediction_metrics,
    infer_fold_index,
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


def _metrics_files_from_inputs(input_roots: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for root in input_roots:
        root = Path(root)
        if root.is_file() and root.name.startswith("metrics_summary") and root.suffix == ".csv":
            files.append(root)
            continue
        if root.is_dir():
            files.extend(root.rglob("metrics_summary*.csv"))
    # stable unique
    uniq: List[Path] = []
    seen = set()
    for p in sorted(files):
        sp = str(p.resolve())
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p)
    return uniq


def _suffix_from_metrics_file(metrics_csv: Path) -> str:
    stem = metrics_csv.stem
    if stem == "metrics_summary":
        return ""
    if stem.startswith("metrics_summary_"):
        return "_" + stem[len("metrics_summary_") :]
    return ""


def _find_pred_csv_for_metrics(metrics_csv: Path, split_name: str) -> Optional[Path]:
    parent = metrics_csv.parent
    suf = _suffix_from_metrics_file(metrics_csv)
    if split_name == "control":
        candidates = [
            f"control_val_results{suf}.csv",
            f"control_test_results{suf}.csv",
            f"control_results{suf}.csv",
        ]
    else:
        candidates = [
            f"rag_experimental_results{suf}.csv",
            f"stress_results{suf}.csv",
            f"test_results{suf}.csv",
        ]
    for name in candidates:
        p = parent / name
        if p.exists():
            return p
    return None


def collect_fold_breakdowns(metric_files: List[Path], split_name: str) -> pd.DataFrame:
    frames = []
    for mfile in metric_files:
        run_dir = mfile.parent
        pred_csv = _find_pred_csv_for_metrics(mfile, split_name)
        if pred_csv is None or not pred_csv.exists():
            continue
        df = load_prediction_csv(pred_csv)
        breakdown = groupwise_prediction_metrics(df, ["cohort", "day"], include_inter_eye=(split_name == "control"))
        breakdown["run_dir"] = str(run_dir)
        breakdown["run_name"] = run_dir.name
        breakdown["fold"] = infer_fold_index(mfile) if infer_fold_index(mfile) is not None else infer_fold_index(run_dir)
        breakdown["metrics_file"] = str(mfile)
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

    metric_files = _metrics_files_from_inputs(args.input_dir)
    if not metric_files:
        # backwards compatible exact-mode fallback
        run_dirs = collect_run_dirs(args.input_dir)
        if not run_dirs:
            raise SystemExit("No metrics_summary*.csv files found.")
        metric_files = []
        for rd in run_dirs:
            p = rd / "metrics_summary.csv"
            if p.exists():
                metric_files.append(p)
    run_dirs = sorted({p.parent for p in metric_files})

    metric_rows = []
    for mfile in metric_files:
        rd = mfile.parent
        mdf = read_metrics_summary(mfile)
        flat = flatten_metric_summary_rows(rd, mdf)
        # Override fold inference from filename when fold suffix is only on metrics filename.
        fold_idx = infer_fold_index(mfile)
        if fold_idx is not None:
            flat["fold"] = fold_idx
        flat["metrics_file"] = str(mfile)
        metric_rows.append(flat)
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
        ctrl = collect_fold_breakdowns(metric_files, "control")
        stress = collect_fold_breakdowns(metric_files, "stress")
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
