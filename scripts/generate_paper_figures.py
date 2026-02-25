#!/usr/bin/env python3
"""Generate manuscript figures from experiment outputs.

Figures are generated only when the required input files exist. Missing inputs
are skipped with warnings.
"""

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


def fig1_calibration_plots(ctrl_preds: Path, out_path: Path) -> bool:
    if not ctrl_preds.exists():
        print(f"[WARN] Missing control predictions for calibration: {ctrl_preds}")
        return False
    df = load_prediction_csv(ctrl_preds)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for cohort, sub in df.groupby("cohort", dropna=False):
        ax.scatter(sub["age_true"], sub["age_pred"], s=18, alpha=0.7, label=f"Cohort {cohort}")
    mn = float(min(df["age_true"].min(), df["age_pred"].min()))
    mx = float(max(df["age_true"].max(), df["age_pred"].max()))
    ax.plot([mn, mx], [mn, mx], "k--", lw=1)
    ax.set_xlabel("True age (days)")
    ax.set_ylabel("Predicted age (days)")
    ax.set_title("Control Calibration (Predicted vs True Age)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    safe_mkdir_for_file(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def fig2_rag_trajectories(hls_preds: Path, control_preds: Path, out_path: Path) -> bool:
    if not (hls_preds.exists() and control_preds.exists()):
        print("[WARN] Missing control/HLS predictions for longitudinal trajectories.")
        return False
    df = pd.concat([load_prediction_csv(control_preds), load_prediction_csv(hls_preds)], ignore_index=True)
    df["condition"] = df["group"].replace({"HLS (U)": "HLS"})
    agg = df.groupby(["cohort", "condition", "day"], dropna=False)["RAG"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    cohorts = sorted(agg["cohort"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, max(1, len(cohorts)), figsize=(5 * len(cohorts), 4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, cohort in zip(axes, cohorts):
        subc = agg[agg["cohort"].astype(str) == str(cohort)]
        for cond, sub in subc.groupby("condition", dropna=False):
            sub = sub.sort_values("day")
            ax.plot(sub["day"], sub["mean"], marker="o", label=str(cond))
            ax.fill_between(sub["day"], sub["mean"] - sub["sem"], sub["mean"] + sub["sem"], alpha=0.15)
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_title(f"Cohort {cohort}")
        ax.set_xlabel("Day")
    axes[0].set_ylabel("RAG (days)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Longitudinal RAG Trajectories (Control vs HLS)")
    fig.tight_layout()
    safe_mkdir_for_file(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def fig3_delta_rag_barchart(delta_csv: Path, out_path: Path) -> bool:
    if not delta_csv.exists():
        print(f"[WARN] Missing ΔRAG CSV for bar chart: {delta_csv}")
        return False
    df = pd.read_csv(delta_csv)
    # Try to normalize expected schema; fall back to plotting any 'Delta' column.
    delta_col = None
    for c in ["Delta_RAG", "delta_rag", "delta_RAG", "delta"]:
        if c in df.columns:
            delta_col = c
            break
    if delta_col is None:
        print("[WARN] Could not find delta column in ΔRAG CSV.")
        return False
    if "cohort" not in df.columns and "Cohort" in df.columns:
        df = df.rename(columns={"Cohort": "cohort"})
    if "day" not in df.columns and "Day" in df.columns:
        df = df.rename(columns={"Day": "day"})
    if "cohort" not in df.columns:
        df["cohort"] = "ALL"
    if "day" not in df.columns:
        df["day"] = 90
    day90 = df[df["day"].astype(float) == 90] if "day" in df.columns else df
    if day90.empty:
        day90 = df
    x = np.arange(len(day90))
    vals = day90[delta_col].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x, vals)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in day90["cohort"]])
    ax.set_ylabel("ΔRAG (days)")
    ax.set_title("ΔRAG by Cohort (Day 90)")
    fig.tight_layout()
    safe_mkdir_for_file(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def fig5_backbone_comparison(backbone_cmp_csv: Path, out_path: Path) -> bool:
    if not backbone_cmp_csv.exists():
        print(f"[WARN] Missing backbone comparison CSV: {backbone_cmp_csv}")
        return False
    df = pd.read_csv(backbone_cmp_csv)
    if "model" not in df.columns or "control_mae" not in df.columns:
        print("[WARN] backbone comparison CSV missing `model`/`control_mae` columns.")
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(df))
    ax.bar(x, df["control_mae"].astype(float), label="Control MAE")
    if "stress_mae" in df.columns:
        ax.plot(x, df["stress_mae"].astype(float), "o-", color="tab:red", label="Stress MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].astype(str), rotation=15, ha="right")
    ax.set_ylabel("MAE (days)")
    ax.set_title("Backbone Comparison")
    ax.legend()
    fig.tight_layout()
    safe_mkdir_for_file(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate manuscript figures from saved experiment outputs")
    ap.add_argument("--control-preds", type=Path, default=Path("outputs/core/exp01_ctrl_cv/fold0/control_val_results.csv"))
    ap.add_argument("--hls-preds", type=Path, default=Path("outputs/core/exp02_hls_ood/rag_experimental_results.csv"))
    ap.add_argument("--delta-rag-csv", type=Path, default=Path("outputs/core/exp02_hls_ood/rag_results_delta.csv"))
    ap.add_argument("--backbone-comparison-csv", type=Path, default=Path("outputs/ablation/backbone_comparison.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/paper_ready/figures"))
    ap.add_argument("--all", action="store_true", help="Generate all available figures (default behavior)")
    args = ap.parse_args()

    out_dir = args.out_dir
    produced: List[Path] = []
    if fig1_calibration_plots(args.control_preds, out_dir / "figure1_calibration_plots.png"):
        produced.append(out_dir / "figure1_calibration_plots.png")
    if fig2_rag_trajectories(args.hls_preds, args.control_preds, out_dir / "figure2_longitudinal_rag_trajectories.png"):
        produced.append(out_dir / "figure2_longitudinal_rag_trajectories.png")
    if fig3_delta_rag_barchart(args.delta_rag_csv, out_dir / "figure3_delta_rag_barchart.png"):
        produced.append(out_dir / "figure3_delta_rag_barchart.png")
    if fig5_backbone_comparison(args.backbone_comparison_csv, out_dir / "figure5_backbone_comparison.png"):
        produced.append(out_dir / "figure5_backbone_comparison.png")

    index_path = out_dir / "generated_figures_index.csv"
    safe_mkdir_for_file(index_path)
    pd.DataFrame({"figure_file": [str(p) for p in produced]}).to_csv(index_path, index=False)
    print(f"[FIGURES] Generated {len(produced)} figure file(s). Index: {index_path}")


if __name__ == "__main__":
    main()

