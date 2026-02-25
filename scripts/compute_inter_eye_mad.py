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

from paper_common import collect_run_dirs, discover_run_files, load_or_build_inter_eye, safe_mkdir_for_file  # noqa: E402


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


def _collect_paired_csvs(input_roots: List[Path]) -> List[Path]:
    files: List[Path] = []
    for root in input_roots:
        root = Path(root)
        if root.is_file() and root.name.endswith("_inter_eye_differences.csv"):
            files.append(root)
            continue
        if root.is_dir():
            files.extend(root.rglob("*_inter_eye_differences*.csv"))
    uniq = []
    seen = set()
    for p in sorted(files):
        name = p.name
        # Skip helper derivatives; keep the base paired CSVs only.
        if any(tag in name for tag in ("matched_view", "reliability", "_thresholds")):
            continue
        sp = str(p.resolve())
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p)
    return uniq


def _infer_split_from_pair_filename(path: Path) -> str:
    nm = path.name.lower()
    if "control" in nm:
        return "control"
    if "rag_experimental" in nm or "stress" in nm:
        return "stress"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute inter-eye MAD summaries from run outputs")
    ap.add_argument("--predictions-dir", type=Path, nargs="+", required=True, help="Run directories or roots containing run directories")
    ap.add_argument("--group-by", type=str, default="group,cohort,day", help="Comma-separated grouping columns (e.g. condition,cohort,day)")
    ap.add_argument("--metric-col", type=str, default="age_pred_inter_eye_abs", help="Metric column in paired inter-eye CSV")
    ap.add_argument("--output", type=Path, required=True, help="Output CSV summary path")
    ap.add_argument("--include-overall", action="store_true", help="Also append overall summary rows")
    args = ap.parse_args()

    group_cols = _normalize_groupby(args.group_by)
    rows = []
    # Prefer direct paired CSV discovery so fold-suffixed outputs are supported.
    paired_csvs = _collect_paired_csvs(args.predictions_dir)
    if paired_csvs:
        for pair_csv in paired_csvs:
            try:
                pair = pd.read_csv(pair_csv)
            except Exception as e:
                print(f"[WARN] Failed to read paired CSV {pair_csv}: {e}")
                continue
            if pair.empty:
                continue
            pair = pair.copy()
            pair["run_dir"] = str(pair_csv.parent)
            pair["run_name"] = pair_csv.parent.name
            pair["split"] = _infer_split_from_pair_filename(pair_csv)
            for c in group_cols:
                if c not in pair.columns:
                    pair[c] = "Unknown"
                pair[f"__grp__{c}"] = pair[c]
            rows.append(pair)
    else:
        run_dirs = collect_run_dirs(args.predictions_dir)
        # Fallback: user may pass a single run dir with no metrics_summary? include direct dirs.
        if not run_dirs:
            run_dirs = [p for p in args.predictions_dir if p.is_dir()]
        if not run_dirs:
            raise SystemExit("No run directories found.")
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
