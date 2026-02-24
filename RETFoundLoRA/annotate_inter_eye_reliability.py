#!/usr/bin/env python3
"""
Annotate paired inter-eye CSVs with reliability flags using control-derived thresholds.

Example:
  python RETFoundLoRA/annotate_inter_eye_reliability.py \
    --control-csv outputs/predictions/run/control_val_inter_eye_differences.csv \
    --target-csv outputs/predictions/run/control_val_inter_eye_differences.csv \
                 outputs/predictions/run/rag_experimental_inter_eye_differences.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_COL = "age_pred_inter_eye_abs"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _resolve_metric_col(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    # Fallbacks for schema drift.
    fallbacks = [
        "RAG_inter_eye_abs",
        "age_pred_inter_eye_abs",
        "inter_eye_abs",
    ]
    for c in fallbacks:
        if c in df.columns:
            return c
    raise ValueError(
        f"No inter-eye abs column found. Requested='{requested}', columns={list(df.columns)}"
    )


def _calc_thresholds(control_vals: pd.Series, q_warn: float, q_unrel: float, q_extreme: float) -> dict:
    vals = pd.to_numeric(control_vals, errors="coerce").dropna().astype(float)
    if vals.empty:
        raise ValueError("Control inter-eye values are empty after numeric conversion.")
    return {
        "metric": DEFAULT_COL,
        "n_control_pairs": int(vals.shape[0]),
        "q_warn": float(q_warn),
        "q_unreliable": float(q_unrel),
        "q_extreme": float(q_extreme),
        "thresh_warn": float(vals.quantile(q_warn)),
        "thresh_unreliable": float(vals.quantile(q_unrel)),
        "thresh_extreme": float(vals.quantile(q_extreme)),
        "control_mean": float(vals.mean()),
        "control_median": float(vals.median()),
        "control_max": float(vals.max()),
    }


def _annotate(df: pd.DataFrame, metric_col: str, th: dict) -> pd.DataFrame:
    out = df.copy()
    vals = pd.to_numeric(out[metric_col], errors="coerce").astype(float)
    out["inter_eye_metric_col"] = metric_col
    out["inter_eye_thresh_warn"] = th["thresh_warn"]
    out["inter_eye_thresh_unreliable"] = th["thresh_unreliable"]
    out["inter_eye_thresh_extreme"] = th["thresh_extreme"]
    out["inter_eye_flag_warn"] = vals > th["thresh_warn"]
    out["inter_eye_flag_unreliable"] = vals > th["thresh_unreliable"]
    out["inter_eye_flag_extreme"] = vals > th["thresh_extreme"]

    tier = np.full(len(out), "ok", dtype=object)
    tier[vals > th["thresh_warn"]] = "warn"
    tier[vals > th["thresh_unreliable"]] = "unreliable"
    tier[vals > th["thresh_extreme"]] = "extreme"
    tier[pd.isna(vals)] = "unknown"
    out["inter_eye_reliability_tier"] = tier
    return out


def _summarize(path: Path, df: pd.DataFrame, metric_col: str) -> dict:
    vals = pd.to_numeric(df[metric_col], errors="coerce")
    n = int(len(df))
    n_valid = int(vals.notna().sum())
    row = {
        "file": str(path),
        "n_rows": n,
        "n_valid_metric": n_valid,
        "metric_col": metric_col,
        "metric_mean": float(vals.mean()) if n_valid else float("nan"),
        "metric_median": float(vals.median()) if n_valid else float("nan"),
        "metric_max": float(vals.max()) if n_valid else float("nan"),
    }
    for c in ["inter_eye_flag_warn", "inter_eye_flag_unreliable", "inter_eye_flag_extreme"]:
        if c in df.columns:
            n_flag = int(pd.Series(df[c]).fillna(False).astype(bool).sum())
            row[c + "_n"] = n_flag
            row[c + "_pct"] = float(n_flag / n) if n else float("nan")
    if "mil_pair_low_conf_any" in df.columns and "age_pred_inter_eye_abs" in df.columns:
        low = pd.Series(df["mil_pair_low_conf_any"]).fillna(False).astype(bool)
        row["mil_pair_low_conf_any_n"] = int(low.sum())
        if low.any():
            row["metric_mean_low_conf"] = float(pd.to_numeric(df.loc[low, "age_pred_inter_eye_abs"], errors="coerce").mean())
        if (~low).any():
            row["metric_mean_not_low_conf"] = float(pd.to_numeric(df.loc[~low, "age_pred_inter_eye_abs"], errors="coerce").mean())
    return row


def _default_out_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-csv", type=Path, required=True)
    ap.add_argument("--target-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--column", type=str, default=DEFAULT_COL)
    ap.add_argument("--q-warn", type=float, default=0.90)
    ap.add_argument("--q-unreliable", type=float, default=0.95)
    ap.add_argument("--q-extreme", type=float, default=0.99)
    ap.add_argument("--out-suffix", type=str, default="_reliability")
    ap.add_argument("--summary-csv", type=Path, default=None)
    ap.add_argument("--thresholds-json", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.q_warn < args.q_unreliable < args.q_extreme < 1):
        raise SystemExit("Require 0 < q_warn < q_unreliable < q_extreme < 1")

    ctrl_df = _read_csv(args.control_csv)
    ctrl_col = _resolve_metric_col(ctrl_df, args.column)
    th = _calc_thresholds(ctrl_df[ctrl_col], args.q_warn, args.q_unreliable, args.q_extreme)
    th["metric"] = ctrl_col
    th["control_csv"] = str(args.control_csv)

    out_rows = []
    for p in args.target_csv:
        df = _read_csv(p)
        metric_col = _resolve_metric_col(df, args.column)
        annotated = _annotate(df, metric_col, th)
        out_p = p if args.overwrite else _default_out_path(p, args.out_suffix)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        annotated.to_csv(out_p, index=False)
        row = _summarize(out_p, annotated, metric_col)
        out_rows.append(row)
        print(
            f"[INTER-EYE][RELIABILITY] {p.name} -> {out_p.name} | "
            f"warn={row.get('inter_eye_flag_warn_n', 0)}/{row['n_rows']} | "
            f"unreliable={row.get('inter_eye_flag_unreliable_n', 0)}/{row['n_rows']} | "
            f"extreme={row.get('inter_eye_flag_extreme_n', 0)}/{row['n_rows']}"
        )

    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = args.control_csv.parent / "inter_eye_reliability_summary.csv"
    th_json = args.thresholds_json
    if th_json is None:
        th_json = args.control_csv.parent / "inter_eye_reliability_thresholds.json"

    pd.DataFrame(out_rows).to_csv(summary_csv, index=False)
    with open(th_json, "w", encoding="utf-8") as f:
        json.dump(th, f, indent=2)
    print(
        "[INTER-EYE][RELIABILITY] "
        f"thresholds from control ({ctrl_col}): "
        f"q{int(args.q_warn*100)}={th['thresh_warn']:.2f}, "
        f"q{int(args.q_unreliable*100)}={th['thresh_unreliable']:.2f}, "
        f"q{int(args.q_extreme*100)}={th['thresh_extreme']:.2f}"
    )
    print(f"[INTER-EYE][RELIABILITY] Saved summary to {summary_csv}")
    print(f"[INTER-EYE][RELIABILITY] Saved thresholds to {th_json}")


if __name__ == "__main__":
    main()
