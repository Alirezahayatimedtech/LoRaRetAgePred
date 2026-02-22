#!/usr/bin/env python3
"""
Run data-efficiency sweeps by subsampling the training set and capturing val MAE.
Uses the run.py CLI with --subset-size to control training rows.
"""

import argparse
import re
import subprocess
from pathlib import Path


def run_once(size, base_cmd):
    cmd = base_cmd.copy()
    label = str(size)
    if isinstance(size, int):
        cmd += ["--subset-size", str(size)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout
    stderr = proc.stderr
    val_mae = None
    m = re.findall(r"val_L1=([0-9.]+)", stdout)
    if m:
        val_mae = float(m[-1])
    print(f"[RUN {label}] exit={proc.returncode} val_L1={val_mae}")
    if proc.returncode != 0:
        print(stderr)
    return val_mae, stdout, stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bias-correction", action="store_true")
    ap.add_argument("--bias-correction-mode", type=str, default="poly2")
    ap.add_argument("--sizes", type=str, default="150,300,600,1200,full",
                    help="Comma-separated subset sizes (use 'full' for entire set)")
    ap.add_argument("--run-script", type=Path, default=Path("RETFoundLoRA/run.py"))
    args = ap.parse_args()

    sizes = []
    for tok in args.sizes.split(","):
        tok = tok.strip().lower()
        if tok in {"full", "all"}:
            sizes.append("full")
        else:
            try:
                sizes.append(int(tok))
            except ValueError:
                pass

    base_cmd = ["python", str(args.run_script),
                "--device", args.device,
                "--bias-correction",
                "--bias-correction-mode", args.bias_correction_mode]

    results = []
    for sz in sizes:
        cmd = base_cmd.copy()
        if isinstance(sz, int):
            cmd += ["--subset-size", str(sz), "--pred-csv", f"lora_age_outputs/pred_subset_{sz}.csv"]
        else:
            cmd += ["--pred-csv", "lora_age_outputs/pred_subset_full.csv"]
        mae, _, _ = run_once(sz, cmd)
        results.append((sz, mae))

    print("\n[SUMMARY] Sample Size -> Val_L1")
    for sz, mae in results:
        print(f"{sz}: {mae}")

    try:
        import matplotlib.pyplot as plt
        xs = [r[0] for r in results if isinstance(r[0], int) and r[1] is not None]
        ys = [r[1] for r in results if isinstance(r[0], int) and r[1] is not None]
        if xs and ys:
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xlabel("Training rows (subset size)")
            plt.ylabel("Val MAE (L1)")
            plt.title("Data Efficiency")
            plt.grid(True)
            plt.savefig("lora_age_outputs/data_efficiency.png", dpi=150, bbox_inches="tight")
            print("[PLOT] Saved lora_age_outputs/data_efficiency.png")
    except Exception as e:
        print(f"[PLOT] Skipped plotting ({e})")


if __name__ == "__main__":
    main()
