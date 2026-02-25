#!/usr/bin/env python3
"""Compare backbone runs using metrics_summary.csv + inter-eye paired CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from paper_common import (  # noqa: E402
    discover_run_files,
    load_or_build_inter_eye,
    read_metrics_summary,
    summarize_inter_eye,
    safe_mkdir_for_file,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare backbone experiment outputs")
    ap.add_argument("--inputs", type=Path, nargs="+", required=True, help="Run directories (each containing metrics_summary.csv)")
    ap.add_argument("--labels", nargs="*", default=None, help="Optional labels matching --inputs order")
    ap.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.inputs):
        raise SystemExit("--labels must match number of --inputs")

    rows = []
    for idx, run_dir in enumerate(args.inputs):
        run_dir = Path(run_dir)
        label = args.labels[idx] if args.labels else run_dir.name
        rf = discover_run_files(run_dir)
        if rf.metrics_csv is None:
            print(f"[WARN] Missing metrics_summary.csv in {run_dir}; skipping")
            continue
        m = read_metrics_summary(rf.metrics_csv)
        row = {"model": label, "run_dir": str(run_dir)}
        for split in ("control", "stress"):
            sub = m[m["split"] == split]
            if sub.empty:
                continue
            s = sub.iloc[0]
            for col in ("n_rows", "mae", "rmse", "r2", "pearson_r", "spearman_r"):
                if col in s.index:
                    row[f"{split}_{col}"] = s[col]

        ctrl_ie = summarize_inter_eye(load_or_build_inter_eye(rf.control_inter_eye_csv, rf.control_pred_csv))
        st_ie = summarize_inter_eye(load_or_build_inter_eye(rf.stress_inter_eye_csv, rf.stress_pred_csv))
        for prefix, stats in (("control_ie", ctrl_ie), ("stress_ie", st_ie)):
            row[f"{prefix}_n_pairs"] = stats["n_pairs"]
            row[f"{prefix}_mean"] = stats["mean"]
            row[f"{prefix}_median"] = stats["median"]
            row[f"{prefix}_q95"] = stats["q95"]
            row[f"{prefix}_max"] = stats["max"]
        rows.append(row)

    if not rows:
        raise SystemExit("No comparable runs found.")
    out = pd.DataFrame(rows)
    safe_mkdir_for_file(args.output)
    out.to_csv(args.output, index=False)
    print(f"[COMPARE] Saved backbone comparison to {args.output}")


if __name__ == "__main__":
    main()

