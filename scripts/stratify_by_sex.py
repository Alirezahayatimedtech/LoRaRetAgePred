#!/usr/bin/env python3
"""Stratify prediction results by sex/cohort/condition/day."""

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

from paper_common import load_prediction_csv, safe_mkdir_for_file  # noqa: E402


def load_concat(paths: List[Path]) -> pd.DataFrame:
    dfs = [load_prediction_csv(Path(p)) for p in paths]
    if not dfs:
        raise SystemExit("No prediction CSVs provided.")
    return pd.concat(dfs, ignore_index=True)


def compute_group_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        s = sub.copy()
        s["age_true"] = pd.to_numeric(s["age_true"], errors="coerce")
        s["age_pred"] = pd.to_numeric(s["age_pred"], errors="coerce")
        s = s.dropna(subset=["age_true", "age_pred"])
        if s.empty:
            continue
        err = s["age_pred"] - s["age_true"]
        row = {c: v for c, v in zip(group_cols, key)}
        row.update({
            "n_rows": int(len(s)),
            "n_rats": int(s["rat_id"].astype(str).nunique()) if "rat_id" in s.columns else np.nan,
            "age_true_mean": float(s["age_true"].mean()),
            "age_pred_mean": float(s["age_pred"].mean()),
            "rag_mean": float(err.mean()),
            "rag_std": float(err.std(ddof=1)) if len(err) > 1 else 0.0,
            "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.mean(err**2))),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratify prediction outputs by sex")
    ap.add_argument("--predictions", type=Path, nargs="+", required=True, help="Prediction CSV(s)")
    ap.add_argument("--metadata", type=Path, default=None, help="Optional metadata CSV to fill missing sex (not required if predictions already include sex)")
    ap.add_argument("--group-by", type=str, default="cohort,sex,group,day", help="Comma-separated grouping columns")
    ap.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = ap.parse_args()

    df = load_concat(args.predictions)
    if "sex" not in df.columns and args.metadata:
        meta = pd.read_csv(args.metadata)
        if {"rat_id", "sex"} <= set(meta.columns):
            fill = meta[["rat_id", "sex"]].drop_duplicates("rat_id")
            df = df.merge(fill, on="rat_id", how="left", suffixes=("", "_meta"))
            if "sex" not in df.columns and "sex_meta" in df.columns:
                df["sex"] = df["sex_meta"]
            elif "sex_meta" in df.columns:
                df["sex"] = df["sex"].fillna(df["sex_meta"])
    if "sex" not in df.columns:
        raise SystemExit("Predictions do not contain `sex` and metadata fill did not provide it.")
    df["sex"] = df["sex"].fillna("Unknown").astype(str).str.strip()

    group_cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
    for c in group_cols:
        if c not in df.columns:
            raise SystemExit(f"Grouping column missing from dataframe: {c}")
    out = compute_group_metrics(df, group_cols)
    safe_mkdir_for_file(args.output)
    out.to_csv(args.output, index=False)
    print(f"[SEX] Saved stratified summary to {args.output}")


if __name__ == "__main__":
    main()

