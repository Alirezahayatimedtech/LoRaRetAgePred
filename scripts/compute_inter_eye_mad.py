#!/usr/bin/env python3
"""Aggregate inter-eye reliability statistics from prediction outputs.

Supports directories containing paired inter-eye CSVs or raw prediction CSVs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from paper_common import (  # noqa: E402
    collect_run_dirs,
    discover_run_files,
    load_or_build_inter_eye,
    safe_mkdir_for_file,
)


def _normalize_groupby(group_by: str) -> List[str]:
    cols = [c.strip() for c in group_by.split(",") if c.strip()]
    aliases = {"condition": "group"}
    return [aliases.get(c, c) for c in cols]


def _summarize_group(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    vals = pd.to_numeric(df[metric_col], errors="coerce")
    g = df.assign(__metric=vals).dropna(subset=["__metric"])
    if g.empty:
        return pd.DataFrame()
    out = g.groupby([c for c in g.columns if c.startswith("__grp__")], dropna=False)["__metric"].agg(
        n_pairs="count",
        mad_mean="mean",
        mad_median="median",
        mad_q90=lambda s: float(s.quantile(0.90)),
        mad_q95=lambda s: float(s.quantile(0.95)),
        mad_q99=lambda s: float(s.quantile(0.99)),
        mad_max="max",
    ).reset_index()
    out.columns = [c.replace("__grp__", "") for c in out.columns]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute inter-eye MAD summaries from run outputs")
    ap.add_argument("--predictions-dir", type=Path, nargs="+", required=True, help="Run directories or roots containing run directories")
    ap.add_argument("--group-by", type=str, default="group,cohort,day", help="Comma-separated grouping columns (e.g. condition,cohort,day)")
    ap.add_argument("--metric-col", type=str, default="age_pred_inter_eye_abs", help="Metric column in paired inter-eye CSV")
    ap.add_argument("--output", type=Path, required=True, help="Output CSV summary path")
    ap.add_argument("--include-overall", action="store_true", help="Also append overall summary rows")
    args = ap.parse_args()

    run_dirs = collect_run_dirs(args.predictions_dir)
    # Fallback: user may pass a single run dir with no metrics_summary? include direct dirs.
    if not run_dirs:
        run_dirs = [p for p in args.predictions_dir if p.is_dir()]
    if not run_dirs:
        raise SystemExit("No run directories found.")

    group_cols = _normalize_groupby(args.group_by)
    rows = []
    for run_dir in run_dirs:
        rf = discover_run_files(run_dir)
        for split, pair_csv, pred_csv in (
            ("control", rf.control_inter_eye_csv, rf.control_pred_csv),
            ("stress", rf.stress_inter_eye_csv, rf.stress_pred_csv),
        ):
            pair = load_or_build_inter_eye(pair_csv, pred_csv)
            if pair is None or pair.empty:
                continue
            pair = pair.copy()
            pair["run_dir"] = str(run_dir)
            pair["run_name"] = run_dir.name
            pair["split"] = split
            for c in group_cols:
                if c not in pair.columns:
                    pair[c] = "Unknown"
                pair[f"__grp__{c}"] = pair[c]
            rows.append(pair)

    if not rows:
        raise SystemExit("No paired inter-eye data found.")
    all_pairs = pd.concat(rows, ignore_index=True)

    summaries = []
    for (run_name, split), sub in all_pairs.groupby(["run_name", "split"], dropna=False):
        out = _summarize_group(sub, args.metric_col)
        if out.empty:
            continue
        out["run_name"] = run_name
        out["split"] = split
        summaries.append(out)
        if args.include_overall:
            metric = pd.to_numeric(sub[args.metric_col], errors="coerce").dropna()
            if not metric.empty:
                summaries.append(pd.DataFrame([{
                    "run_name": run_name,
                    "split": split,
                    **{c: "ALL" for c in group_cols},
                    "n_pairs": int(metric.shape[0]),
                    "mad_mean": float(metric.mean()),
                    "mad_median": float(metric.median()),
                    "mad_q90": float(metric.quantile(0.90)),
                    "mad_q95": float(metric.quantile(0.95)),
                    "mad_q99": float(metric.quantile(0.99)),
                    "mad_max": float(metric.max()),
                }]))

    final = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if final.empty:
        raise SystemExit("Inter-eye summaries could not be computed.")

    # Column order
    first_cols = ["run_name", "split"] + group_cols
    rest = [c for c in final.columns if c not in first_cols]
    final = final[first_cols + rest]

    safe_mkdir_for_file(args.output)
    final.to_csv(args.output, index=False)
    print(f"[INTER-EYE] Saved summary to {args.output}")


if __name__ == "__main__":
    main()

