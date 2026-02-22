#!/usr/bin/env python3
"""
Longitudinal Analysis of Retinal Aging.
Plots RAG trajectories over time (Suspension vs Recovery Phases) for selected cohorts.
Focuses on HLS vs Controls within cohorts to avoid batch effects.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def load_and_normalize(csv_paths, keep_cohorts):
    # 1. Load
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    # 2. Filter Cohorts
    df["cohort"] = df["cohort"].astype(str).str.strip()
    df = df[df["cohort"].isin(keep_cohorts)]

    # 3. Calculate Raw RAG
    df["RAG"] = df["age_pred"] - df["age_true"]

    # 4. Normalize by Cohort (center controls to ~0)
    norm_rows = []
    print(f"[INFO] Normalizing cohorts {keep_cohorts} using control medians...")
    for coh in keep_cohorts:
        subset = df[df["cohort"] == coh].copy()
        controls = subset[subset["group"] == "Controls"]
        if controls.empty:
            baseline = 0.0
        else:
            baseline = controls["RAG"].median()
        subset["RAG_Norm"] = subset["RAG"] - baseline
        norm_rows.append(subset)
    df_norm = pd.concat(norm_rows, ignore_index=True)

    # 5. Filter Groups of interest
    df_norm = df_norm[df_norm["group"].isin(["Controls", "HLS (U)"])]
    return df_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", nargs="+", required=True, help="Prediction CSVs from run.py")
    parser.add_argument("--cohorts", nargs="*", default=["1", "3"], help="Cohort IDs to include")
    args = parser.parse_args()

    df = load_and_normalize([Path(p) for p in args.pred_csv], args.cohorts)

    # Define phases
    df["Phase"] = df["day"].apply(lambda d: "Recovery (>90d)" if d > 90 else "Suspension (<=90d)")

    # Stats: Recovery only
    rec_hls = df[(df["group"] == "HLS (U)") & (df["Phase"] == "Recovery (>90d)")]
    rec_ctrl = df[(df["group"] == "Controls") & (df["Phase"] == "Recovery (>90d)")]
    rec_hls_subj = rec_hls.groupby(["rat_id"])["RAG_Norm"].mean()
    rec_ctrl_subj = rec_ctrl.groupby(["rat_id"])["RAG_Norm"].mean()
    if len(rec_hls_subj) > 0 and len(rec_ctrl_subj) > 0:
        t_stat, p_val = ttest_ind(rec_hls_subj, rec_ctrl_subj, equal_var=False)
    else:
        t_stat, p_val = float("nan"), float("nan")

    print("-" * 60)
    print(f"[STATS] Recovery Phase (Day > 90) Analysis (Cohorts: {args.cohorts})")
    print(f"   HLS Mean RAG: {rec_hls_subj.mean():.2f} days" if len(rec_hls_subj) else "   HLS Mean RAG: n/a")
    print(f"   Ctrl Mean RAG: {rec_ctrl_subj.mean():.2f} days" if len(rec_ctrl_subj) else "   Ctrl Mean RAG: n/a")
    print(f"   P-Value (HLS vs Control): {p_val:.4f}")
    print("-" * 60)

    # Plot (matplotlib only, no seaborn dependency)
    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(True, alpha=0.3)

    # Compute mean and stderr per day/group
    summary = df.groupby(["group", "day"])["RAG_Norm"].agg(["mean", "count", "std"]).reset_index()
    summary["se"] = summary["std"] / summary["count"].pow(0.5)
    colors = {"Controls": "green", "HLS (U)": "red"}

    for grp, grp_df in summary.groupby("group"):
        grp_df = grp_df.sort_values("day")
        plt.plot(grp_df["day"], grp_df["mean"], label=grp, color=colors.get(grp, None), marker="o")
        plt.fill_between(
            grp_df["day"],
            grp_df["mean"] - grp_df["se"],
            grp_df["mean"] + grp_df["se"],
            color=colors.get(grp, None),
            alpha=0.2,
        )

    plt.axvline(x=90, color="gray", linestyle="--", alpha=0.5)
    ymax = df["RAG_Norm"].max() if not df.empty else 0
    plt.text(91, ymax, "Recovery Start", fontsize=10, color="gray")

    plt.title("Longitudinal Retinal Aging: HLS vs Recovery", fontsize=14)
    plt.ylabel("Normalized Retinal Age Gap (Days)", fontsize=12)
    plt.xlabel("Days", fontsize=12)
    plt.legend(title="Group")
    plt.tight_layout()
    out_path = Path("figure_longitudinal_recovery.png")
    plt.savefig(out_path)
    print(f"[DONE] Saved plot to {out_path}")


if __name__ == "__main__":
    main()
