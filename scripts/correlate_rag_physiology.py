#!/usr/bin/env python3
"""Correlate RAG with physiology variables after merging on identifiers."""

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


def _corr(a: pd.Series, b: pd.Series, method: str) -> float:
    if len(a) < 3:
        return np.nan
    return float(a.corr(b, method=method))


def main() -> None:
    ap = argparse.ArgumentParser(description="Correlate RAG with physiology variables")
    ap.add_argument("--predictions", type=Path, nargs="+", required=True, help="Prediction CSV(s) with RAG or age_true/age_pred")
    ap.add_argument("--physiology", type=Path, required=True, help="Physiology CSV")
    ap.add_argument("--variables", nargs="+", required=True, help="Physiology variables to correlate with RAG")
    ap.add_argument("--merge-keys", nargs="+", default=["rat_id", "day"], help="Merge keys (e.g., rat_id day eye)")
    ap.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = ap.parse_args()

    pred = pd.concat([load_prediction_csv(p) for p in args.predictions], ignore_index=True)
    phys = pd.read_csv(args.physiology)

    keys = [k for k in args.merge_keys if k in pred.columns and k in phys.columns]
    if not keys:
        raise SystemExit("No common merge keys found between predictions and physiology CSV.")

    merged = pred.merge(phys, on=keys, how="inner", suffixes=("", "_phys"))
    if merged.empty:
        raise SystemExit("No rows after merging predictions with physiology.")

    merged["RAG"] = pd.to_numeric(merged.get("RAG", merged["age_pred"] - merged["age_true"]), errors="coerce")
    rows = []
    for var in args.variables:
        if var not in merged.columns:
            print(f"[WARN] Physiology variable missing: {var}")
            continue
        sub = merged[["RAG", var]].copy()
        sub[var] = pd.to_numeric(sub[var], errors="coerce")
        sub = sub.dropna()
        if sub.empty:
            continue
        rows.append({
            "variable": var,
            "n": int(len(sub)),
            "pearson_r": _corr(sub["RAG"], sub[var], "pearson"),
            "spearman_r": _corr(sub["RAG"], sub[var], "spearman"),
            "rag_mean": float(sub["RAG"].mean()),
            "rag_std": float(sub["RAG"].std(ddof=1)) if len(sub) > 1 else 0.0,
            f"{var}_mean": float(sub[var].mean()),
            f"{var}_std": float(sub[var].std(ddof=1)) if len(sub) > 1 else 0.0,
        })

    out = pd.DataFrame(rows)
    safe_mkdir_for_file(args.output)
    out.to_csv(args.output, index=False)
    print(f"[CORR] Saved RAG-physiology correlations to {args.output}")


if __name__ == "__main__":
    main()

