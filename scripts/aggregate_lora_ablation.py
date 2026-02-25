#!/usr/bin/env python3
"""Aggregate RETFound adaptation ablation runs (LoRA vs full FT vs frozen head-only)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from paper_common import discover_run_files, read_metrics_summary, flatten_metric_summary_rows, safe_mkdir_for_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True, help="Run directories")
    ap.add_argument("--labels", type=str, nargs="*", default=None, help="Optional labels matching inputs")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    labels = args.labels or [p.name for p in args.inputs]
    if len(labels) != len(args.inputs):
        raise SystemExit("--labels must match --inputs length")

    rows = []
    for label, rd in zip(labels, args.inputs):
        rf = discover_run_files(rd)
        if rf.metrics_csv is None or not rf.metrics_csv.exists():
            rows.append({"model": label, "run_dir": str(rd), "available": False})
            continue
        m = flatten_metric_summary_rows(rd, read_metrics_summary(rf.metrics_csv))
        if m.empty:
            rows.append({"model": label, "run_dir": str(rd), "available": False})
            continue
        for _, r in m.iterrows():
            row = {"model": label, "run_dir": str(rd), "available": True}
            row.update(r.to_dict())
            rows.append(row)
    out = pd.DataFrame(rows)
    safe_mkdir_for_file(args.output)
    out.to_csv(args.output, index=False)
    print(f"[LORA-ABL] Saved {args.output}")


if __name__ == "__main__":
    main()

