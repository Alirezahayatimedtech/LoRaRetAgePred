#!/usr/bin/env python3
"""Plot longitudinal RAG trajectories from prediction CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
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
    df = pd.concat(dfs, ignore_index=True)
    # Normalize condition naming
    df["condition"] = df["group"].replace({"HLS (U)": "HLS"})
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot longitudinal RAG trajectories")
    ap.add_argument("--predictions", type=Path, nargs="+", required=True, help="Prediction CSVs (control and/or HLS)")
    ap.add_argument("--group-by", type=str, default="condition,cohort", help="Comma-separated grouping columns for separate lines")
    ap.add_argument("--output", type=Path, required=True, help="Output figure path (.png/.pdf)")
    ap.add_argument("--csv-out", type=Path, default=None, help="Optional aggregated trajectory CSV")
    ap.add_argument("--days", type=float, nargs="*", default=None, help="Optional day whitelist for plotting")
    args = ap.parse_args()

    df = load_concat(args.predictions)
    if args.days:
        keep = {float(x) for x in args.days}
        df = df[df["day"].astype(float).isin(keep)]
    if df.empty:
        raise SystemExit("No rows left after filtering.")

    group_cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
    for c in group_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing grouping column in predictions: {c}")

    agg = (
        df.groupby(group_cols + ["day"], dropna=False)["RAG"]
        .agg(rag_mean="mean", rag_std="std", n="count")
        .reset_index()
    )
    agg["rag_sem"] = agg["rag_std"] / np.sqrt(agg["n"].clip(lower=1))

    safe_mkdir_for_file(args.output)
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, sub in agg.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        label = " | ".join(f"{c}={v}" for c, v in zip(group_cols, key))
        sub = sub.sort_values("day")
        ax.plot(sub["day"], sub["rag_mean"], marker="o", label=label)
        ax.fill_between(
            sub["day"].to_numpy(dtype=float),
            (sub["rag_mean"] - sub["rag_sem"]).to_numpy(dtype=float),
            (sub["rag_mean"] + sub["rag_sem"]).to_numpy(dtype=float),
            alpha=0.15,
        )

    ax.axhline(0.0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("Day")
    ax.set_ylabel("RAG (pred - true, days)")
    ax.set_title("Longitudinal RAG Trajectories")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Saved longitudinal RAG plot to {args.output}")

    if args.csv_out:
        safe_mkdir_for_file(args.csv_out)
        agg.to_csv(args.csv_out, index=False)
        print(f"[DATA] Saved aggregated trajectory CSV to {args.csv_out}")


if __name__ == "__main__":
    main()

