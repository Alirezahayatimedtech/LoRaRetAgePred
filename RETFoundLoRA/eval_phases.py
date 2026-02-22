#!/usr/bin/env python3
"""
Two-Phase Statistical Evaluation (Suspension vs Recovery).
Generates the specific P-values needed to prove "Reversibility".
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp


def load_and_normalize(csv_paths, keep_cohorts):
    # 1. Load and Filter
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df["cohort"] = df["cohort"].astype(str).str.strip()
    df = df[df["cohort"].isin(keep_cohorts)]

    # 2. Calculate RAG
    df["RAG"] = df["age_pred"] - df["age_true"]

    # 3. Normalize (Cohort-Specific): controls <=90d set to ~0
    norm_rows = []
    print(f"[INFO] Normalizing cohorts {keep_cohorts}...")
    for coh in keep_cohorts:
        subset = df[df["cohort"] == coh].copy()
        controls = subset[(subset["group"] == "Controls") & (subset["day"] <= 90)]
        if controls.empty:
            print(f"[WARN] Cohort {coh}: No baseline controls <=90d found. Skipping normalization for this cohort.")
            subset["RAG_Norm"] = subset["RAG"]
        else:
            baseline = controls["RAG"].median()
            subset["RAG_Norm"] = subset["RAG"] - baseline
        norm_rows.append(subset)
    df_norm = pd.concat(norm_rows, ignore_index=True)
    return df_norm[df_norm["group"].isin(["Controls", "HLS (U)"])]


def run_stats_for_phase(df, phase_name, min_day, max_day, baseline_mode="concurrent"):
    """
    Calculates P-value for a specific time window.
    baseline_mode: 'concurrent' (vs controls) or 'zero' (vs 0.0 one-sample).
    """
    subset = df[(df["day"] >= min_day) & (df["day"] <= max_day)]

    # Split Groups
    hls = subset[subset["group"] == "HLS (U)"]
    ctrl = subset[subset["group"] == "Controls"]

    # Aggregate per Rat (Subject Level)
    hls_subj = hls.groupby(["rat_id"])["RAG_Norm"].mean()
    ctrl_subj = ctrl.groupby(["rat_id"])["RAG_Norm"].mean()

    mean_hls = hls_subj.mean() if not hls_subj.empty else float("nan")
    mean_ctrl = ctrl_subj.mean() if not ctrl_subj.empty else 0.0

    if len(hls_subj) < 2:
        return phase_name, mean_hls, mean_ctrl, float("nan"), "Insufficient Data"

    if baseline_mode == "concurrent" and len(ctrl_subj) > 2:
        stat, p = ttest_ind(hls_subj, ctrl_subj, equal_var=False)
        test_type = "T-Test vs Controls"
    else:
        # Fallback if controls are missing in this phase
        stat, p = ttest_1samp(hls_subj, 0.0)
        test_type = "1-Sample T-Test vs 0"

    return phase_name, mean_hls, mean_ctrl, p, test_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-csv", nargs="+", required=True, help="Prediction CSVs from run.py")
    parser.add_argument("--cohorts", nargs="*", default=["1", "3"], help="Cohort IDs to include")
    args = parser.parse_args()

    df = load_and_normalize([Path(p) for p in args.pred_csv], args.cohorts)

    print("\n" + "=" * 80)
    print(f"TWO-PHASE ANALYSIS (Cohorts: {args.cohorts})")
    print(f"Goal: Prove 'Damage' in Phase 1 and 'Recovery' in Phase 2")
    print("=" * 80)

    # Phase 1: Suspension (14-90d) - expect P < 0.05
    p1 = run_stats_for_phase(df, "Suspension (14-90d)", 14, 90, baseline_mode="concurrent")

    # Phase 2: Recovery (>90d) - expect P > 0.05 (recovery)
    p2 = run_stats_for_phase(df, "Recovery (>90d)", 91, 180, baseline_mode="zero")

    # Report
    print(f"{'Phase':<20} | {'HLS RAG':<10} | {'Ctrl RAG':<10} | {'P-Value':<10} | {'Conclusion'}")
    print("-" * 80)
    for name, h_mu, c_mu, p, test in [p1, p2]:
        sig_str = "**SIG**" if p < 0.05 else "n.s."
        print(f"{name:<20} | {h_mu:>6.2f}d    | {c_mu:>6.2f}d    | {p:.4f} {sig_str} | {test}")
    print("-" * 80)


if __name__ == "__main__":
    main()
