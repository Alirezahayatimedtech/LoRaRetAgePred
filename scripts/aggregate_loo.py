#!/usr/bin/env python3
"""Aggregate leave-one-cohort-out experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from paper_common import (  # noqa: E402
    collect_run_dirs,
    discover_run_files,
    flatten_metric_summary_rows,
    infer_heldout_cohort,
    load_prediction_csv,
    groupwise_prediction_metrics,
    read_metrics_summary,
    safe_mkdir_for_file,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate LOO cohort runs")
    ap.add_argument("--input-dir", type=Path, nargs="+", required=True, help="Dirs containing loo_cohort*/.../metrics_summary.csv")
    ap.add_argument("--output", type=Path, required=True, help="Summary CSV output")
    ap.add_argument("--emit-breakdowns", action="store_true", help="Also emit cohort/day breakdown tables")
    args = ap.parse_args()

    run_dirs = collect_run_dirs(args.input_dir)
    if not run_dirs:
        raise SystemExit("No run directories found.")

    metric_rows = []
    for rd in run_dirs:
        rf = discover_run_files(rd)
        if rf.metrics_csv is None:
            continue
        m = flatten_metric_summary_rows(rd, read_metrics_summary(rf.metrics_csv))
        if "heldout_cohort" not in m.columns or m["heldout_cohort"].isna().all():
            m["heldout_cohort"] = infer_heldout_cohort(rd)
        metric_rows.append(m)
    allm = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    if allm.empty:
        raise SystemExit("No metrics found for LOO aggregation.")

    numeric_cols = [c for c in ("mae", "rmse", "r2", "pearson_r", "spearman_r") if c in allm.columns]
    summary = (
        allm.groupby(["heldout_cohort", "split"], dropna=False)[numeric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["heldout_cohort", "split"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[2:]]
    safe_mkdir_for_file(args.output)
    summary.to_csv(args.output, index=False)
    print(f"[LOO] Saved summary to {args.output}")

    rows_path = args.output.with_name(args.output.stem + "_rows.csv")
    allm.to_csv(rows_path, index=False)
    print(f"[LOO] Saved per-run metric rows to {rows_path}")

    if args.emit_breakdowns:
        breakdown_rows = []
        for rd in run_dirs:
            heldout = infer_heldout_cohort(rd)
            rf = discover_run_files(rd)
            for split, pred_csv in (("control", rf.control_pred_csv), ("stress", rf.stress_pred_csv)):
                if pred_csv is None or not pred_csv.exists():
                    continue
                df = load_prediction_csv(pred_csv)
                b = groupwise_prediction_metrics(df, ["cohort", "day"], include_inter_eye=(split == "control"))
                b["heldout_cohort"] = heldout
                b["split"] = split
                b["run_dir"] = str(rd)
                breakdown_rows.append(b)
        if breakdown_rows:
            bdf = pd.concat(breakdown_rows, ignore_index=True)
            braw = args.output.with_name(args.output.stem + "_cohort_day_rows.csv")
            bdf.to_csv(braw, index=False)
            print(f"[LOO] Saved cohort/day rows to {braw}")


if __name__ == "__main__":
    main()

