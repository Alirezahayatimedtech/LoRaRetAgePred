#!/usr/bin/env python3
"""Summarize distillation runs relative to a baseline (control-priority metrics)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from paper_common import discover_run_files, read_metrics_summary, load_prediction_csv, build_inter_eye_pairs_from_predictions, safe_mkdir_for_file  # noqa: E402


def control_metrics(run_dir: Path) -> dict:
    rf = discover_run_files(run_dir)
    out = {"run_dir": str(run_dir)}
    if rf.metrics_csv and rf.metrics_csv.exists():
        m = read_metrics_summary(rf.metrics_csv)
        c = m[m.get("split", "").astype(str).str.lower() == "control"] if "split" in m.columns else m
        if not c.empty:
            row = c.iloc[0]
            for k in ("mae", "rmse", "r2", "pearson_r"):
                if k in row.index:
                    out[k] = float(row[k])
    pred = None
    for name in ("control_val_results.csv", "control_test_results.csv"):
        p = run_dir / name
        if p.exists():
            pred = p
            break
    if pred is not None:
        df = load_prediction_csv(pred)
        if {"day", "age_true", "age_pred"}.issubset(df.columns):
            x = df.copy()
            x["day"] = pd.to_numeric(x["day"], errors="coerce")
            x["age_true"] = pd.to_numeric(x["age_true"], errors="coerce")
            x["age_pred"] = pd.to_numeric(x["age_pred"], errors="coerce")
            d90 = x[x["day"] == 90].dropna(subset=["age_true", "age_pred"])
            if not d90.empty:
                out["day90_mae"] = float((d90["age_pred"] - d90["age_true"]).abs().mean())
        try:
            pair = build_inter_eye_pairs_from_predictions(df)
            s = pd.to_numeric(pair["age_pred_inter_eye_abs"], errors="coerce").dropna()
            if not s.empty:
                out["inter_eye_mean"] = float(s.mean())
                out["inter_eye_q95"] = float(s.quantile(0.95))
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", type=str, nargs="*", default=None)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    labels = args.labels or [p.name for p in args.inputs]
    if len(labels) != len(args.inputs):
        raise SystemExit("--labels length must match --inputs")
    rows = []
    for label, p in zip(labels, args.inputs):
        d = control_metrics(p)
        d["model"] = label
        rows.append(d)
    if args.baseline:
        b = control_metrics(args.baseline)
        b["model"] = "baseline"
        rows.insert(0, b)
    df = pd.DataFrame(rows)
    safe_mkdir_for_file(args.output)
    df.to_csv(args.output, index=False)
    print(f"[DISTILL] Saved {args.output}")


if __name__ == "__main__":
    main()

