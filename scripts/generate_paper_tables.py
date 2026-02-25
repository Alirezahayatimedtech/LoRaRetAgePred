#!/usr/bin/env python3
"""Generate manuscript-ready tables from experiment outputs.

This script is intentionally tolerant: it emits available tables and skips
missing experiments with warnings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from paper_common import (  # noqa: E402
    collect_run_dirs,
    discover_run_files,
    groupwise_prediction_metrics,
    load_prediction_csv,
    read_metrics_summary,
    safe_mkdir_for_file,
)


def table1_control_performance(exp01_dir: Path, out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    exp01_dir = Path(exp01_dir)
    # Fast path: consume aggregate_cv outputs directly.
    agg_summary = exp01_dir / "summary.csv"
    agg_ctrl_rows = exp01_dir / "summary_control_cohort_day_fold_rows.csv"
    if agg_summary.exists():
        try:
            mdf = pd.read_csv(agg_summary)
            mdf = mdf[mdf["split"] == "control"].copy()
            if not mdf.empty:
                # Already aggregated mean/std table from aggregate_cv
                p = out_dir / "table1_control_cv_performance.csv"
                safe_mkdir_for_file(p)
                mdf.to_csv(p, index=False)
                outputs.append(p)
            if agg_ctrl_rows.exists():
                rows = pd.read_csv(agg_ctrl_rows)
                numeric = [c for c in rows.columns if c not in {"cohort", "day", "run_dir", "run_name", "fold", "metrics_file"} and pd.api.types.is_numeric_dtype(rows[c])]
                if numeric:
                    agg = rows.groupby(["cohort", "day"], dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
                    agg.columns = ["cohort", "day"] + [f"{a}_{b}" for a, b in agg.columns.tolist()[2:]]
                    p2 = out_dir / "table1_control_cohort_day_breakdown.csv"
                    agg.to_csv(p2, index=False)
                    outputs.append(p2)
                return outputs
        except Exception as e:
            print(f"[WARN] Failed direct aggregate load for EXP-01 ({e}); falling back to run-dir parsing.")

    runs = collect_run_dirs([exp01_dir])
    if not runs:
        print(f"[WARN] EXP-01 runs not found under {exp01_dir}")
        return outputs

    metric_rows = []
    breakdown_rows = []
    for rd in runs:
        rf = discover_run_files(rd)
        if rf.metrics_csv and rf.metrics_csv.exists():
            m = read_metrics_summary(rf.metrics_csv)
            m = m[m["split"] == "control"].copy()
            if not m.empty:
                m["run_name"] = rd.name
                metric_rows.append(m)
        if rf.control_pred_csv and rf.control_pred_csv.exists():
            df = load_prediction_csv(rf.control_pred_csv)
            b = groupwise_prediction_metrics(df, ["cohort", "day"], include_inter_eye=True)
            b["run_name"] = rd.name
            breakdown_rows.append(b)

    if metric_rows:
        mdf = pd.concat(metric_rows, ignore_index=True)
        numeric = [c for c in ("mae", "rmse", "r2", "pearson_r") if c in mdf.columns]
        summary = mdf[numeric].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
        p = out_dir / "table1_control_cv_performance.csv"
        safe_mkdir_for_file(p)
        summary.to_csv(p, index=False)
        outputs.append(p)
    if breakdown_rows:
        bdf = pd.concat(breakdown_rows, ignore_index=True)
        numeric = [c for c in bdf.columns if c not in {"cohort", "day", "run_name"} and pd.api.types.is_numeric_dtype(bdf[c])]
        agg = bdf.groupby(["cohort", "day"], dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["cohort", "day"] + [f"{a}_{b}" for a, b in agg.columns.tolist()[2:]]
        p = out_dir / "table1_control_cohort_day_breakdown.csv"
        agg.to_csv(p, index=False)
        outputs.append(p)
    return outputs


def table2_hls_rag(exp02_dir: Path, out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    exp02_dir = Path(exp02_dir)
    # Fast path: consume aggregate_cv outputs directly.
    agg_stress_rows = exp02_dir / "summary_stress_cohort_day_fold_rows.csv"
    if agg_stress_rows.exists():
        try:
            rows = pd.read_csv(agg_stress_rows)
            if "group" not in rows.columns:
                rows["group"] = "stress"
            numeric = [c for c in rows.columns if c not in {"cohort", "day", "group", "run_dir", "run_name", "fold", "metrics_file"} and pd.api.types.is_numeric_dtype(rows[c])]
            if numeric:
                agg = rows.groupby(["cohort", "group", "day"], dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
                agg.columns = ["cohort", "group", "day"] + [f"{a}_{b}" for a, b in agg.columns.tolist()[3:]]
                p = out_dir / "table2_hls_rag_by_cohort_day.csv"
                safe_mkdir_for_file(p)
                agg.to_csv(p, index=False)
                outputs.append(p)
                return outputs
        except Exception as e:
            print(f"[WARN] Failed direct aggregate load for EXP-02 ({e}); falling back to run-dir parsing.")

    runs = collect_run_dirs([exp02_dir])
    if not runs:
        print(f"[WARN] EXP-02 runs not found under {exp02_dir}")
        return outputs
    rows = []
    for rd in runs:
        rf = discover_run_files(rd)
        if not rf.stress_pred_csv or not rf.stress_pred_csv.exists():
            continue
        df = load_prediction_csv(rf.stress_pred_csv)
        b = groupwise_prediction_metrics(df, ["cohort", "day", "group"], include_inter_eye=True)
        b["run_name"] = rd.name
        rows.append(b)
    if not rows:
        return outputs
    df = pd.concat(rows, ignore_index=True)
    numeric = [c for c in df.columns if c not in {"cohort", "day", "group", "run_name"} and pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby(["cohort", "group", "day"], dropna=False)[numeric].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["cohort", "group", "day"] + [f"{a}_{b}" for a, b in agg.columns.tolist()[3:]]
    p = out_dir / "table2_hls_rag_by_cohort_day.csv"
    safe_mkdir_for_file(p)
    agg.to_csv(p, index=False)
    outputs.append(p)
    return outputs


def table3_inter_eye(mad_summary_csv: Path, out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not mad_summary_csv.exists():
        print(f"[WARN] MAD summary not found: {mad_summary_csv}")
        return outputs
    df = pd.read_csv(mad_summary_csv)
    p = out_dir / "table3_inter_eye_reliability.csv"
    safe_mkdir_for_file(p)
    df.to_csv(p, index=False)
    outputs.append(p)
    return outputs


def table4_backbone(backbone_cmp_csv: Path, out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not backbone_cmp_csv.exists():
        print(f"[WARN] Backbone comparison not found: {backbone_cmp_csv}")
        return outputs
    df = pd.read_csv(backbone_cmp_csv)
    p = out_dir / "table4_backbone_ablation.csv"
    safe_mkdir_for_file(p)
    df.to_csv(p, index=False)
    outputs.append(p)
    return outputs


def table5_lora_ablation(lora_ablation_csv: Path, out_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    if not lora_ablation_csv.exists():
        print(f"[WARN] LoRA ablation summary not found: {lora_ablation_csv}")
        return outputs
    df = pd.read_csv(lora_ablation_csv)
    p = out_dir / "table5_lora_adaptation_ablation.csv"
    safe_mkdir_for_file(p)
    df.to_csv(p, index=False)
    outputs.append(p)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate manuscript tables from experiment outputs")
    ap.add_argument("--exp01-dir", type=Path, default=Path("outputs/core/exp01_ctrl_cv"))
    ap.add_argument("--exp02-dir", type=Path, default=Path("outputs/core/exp02_hls_ood"))
    ap.add_argument("--mad-summary-csv", type=Path, default=Path("outputs/core/exp03_inter_eye/summary.csv"))
    ap.add_argument("--backbone-comparison-csv", type=Path, default=Path("outputs/ablation/backbone_comparison.csv"))
    ap.add_argument("--lora-ablation-csv", type=Path, default=Path("outputs/paper1/lora_ablation/summary.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/paper_ready/tables"))
    ap.add_argument("--all", action="store_true", help="Generate all available tables (default behavior)")
    args = ap.parse_args()

    out_dir = args.out_dir
    produced: List[Path] = []
    produced += table1_control_performance(args.exp01_dir, out_dir)
    produced += table2_hls_rag(args.exp02_dir, out_dir)
    produced += table3_inter_eye(args.mad_summary_csv, out_dir)
    produced += table4_backbone(args.backbone_comparison_csv, out_dir)
    produced += table5_lora_ablation(args.lora_ablation_csv, out_dir)

    index_csv = out_dir / "generated_tables_index.csv"
    safe_mkdir_for_file(index_csv)
    pd.DataFrame({"table_file": [str(p) for p in produced]}).to_csv(index_csv, index=False)
    print(f"[TABLES] Generated {len(produced)} table file(s). Index: {index_csv}")


if __name__ == "__main__":
    main()
