#!/usr/bin/env python3
"""
Unified evaluation/analysis suite for RETFound LoRA age prediction.

Choose a task with --task:
  - auroc          : Global AUROC (cohort-centered, subject-level)
  - auroc-cohort   : Per-cohort AUROC breakdown
  - phases         : Suspension vs Recovery phase P-values
  - plot           : Longitudinal plot (saves PNG)
  - mad            : Inter-eye MAD (overall + per-day) for specified groups

Examples:
  python eval_suite.py --task auroc \
    --pred-csv lora_age_outputs/control_test_results.csv lora_age_outputs/rag_experimental_results.csv \
    --min-day 0 --exclude-recovery

  python eval_suite.py --task auroc-cohort \
    --pred-csv lora_age_outputs/control_test_results.csv lora_age_outputs/rag_experimental_results.csv \
    --min-day 0 --exclude-recovery

  python eval_suite.py --task phases \
    --pred-csv lora_age_outputs/control_test_results.csv lora_age_outputs/rag_experimental_results.csv \
    --cohorts 1 3

  python eval_suite.py --task plot \
    --pred-csv lora_age_outputs/control_test_results.csv lora_age_outputs/rag_experimental_results.csv \
    --cohorts 1 3
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_1samp
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def apply_control_day_anchor(df: pd.DataFrame, control_mask: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Subtract mean Control RAG per day so Controls are centered at 0 for each sampled day.
    Useful when linear bias correction leaves residual per-day offsets.
    """
    controls = df[control_mask]
    if controls.empty:
        print("[ANCHOR] No control rows available; skipping control-day anchoring.")
        return df, pd.Series(dtype=float)
    baselines = controls.groupby("day")["RAG"].mean()
    df = df.copy()
    df["RAG"] = df["RAG"] - df["day"].map(baselines).fillna(0.0)
    print(f"[ANCHOR] Applied control day-level anchoring on {len(baselines)} day(s):")
    print(baselines.to_string())
    return df, baselines


# ------------- MAD (inter-eye) -------------
def run_mad(args):
    csv_paths = flatten_paths(args.pred_csv)
    dfs = [load_df(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    if args.mad_groups:
        df = df[df["group"].isin(args.mad_groups)]
        print(f"[MAD] Groups kept: {args.mad_groups} -> N={len(df)} rows")

    pivot = df.pivot_table(
        index=["rat_id", "day"],
        columns="eye",
        values="age_pred",
        aggfunc="mean",
    )
    if not {"OD", "OS"} <= set(pivot.columns):
        raise SystemExit("[MAD] Need both OD and OS entries to compute inter-eye MAD.")
    diffs = (pivot["OD"] - pivot["OS"]).abs().dropna()
    overall_mad = float(diffs.mean()) if len(diffs) else float("nan")
    print(f"[MAD] Inter-eye MAD: {overall_mad:.3f} days" if overall_mad == overall_mad else "[MAD] Inter-eye MAD: NaN")

    per_day = (
        pivot.dropna(subset=["OD", "OS"])
        .reset_index()
        .assign(abs_diff=lambda x: (x["OD"] - x["OS"]).abs())
        .groupby("day")["abs_diff"]
        .agg(mad_days="mean", n_pairs="count")
        .reset_index()
    )
    if not per_day.empty:
        print("[MAD] Per-day MAD (days, mad_days, n_pairs):")
        print(per_day.to_string(index=False))


# ------------- Common loaders -------------
def load_df(path: Path, force_control_label: str = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"rat_id", "group", "day", "age_true", "age_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing columns: {missing}")
    df["group"] = df["group"].astype(str).str.strip()
    if force_control_label:
        df["group"] = force_control_label
    df["cohort"] = df.get("cohort", "Unknown")
    df["cohort"] = df["cohort"].fillna("Unknown").astype(str).str.strip()
    df["sex"] = df.get("sex", "Unknown")
    df["sex"] = df["sex"].fillna("Unknown").astype(str).str.strip()
    df["__source"] = str(path)
    df["RAG"] = df["age_pred"] - df["age_true"]
    return df


def flatten_paths(paths: List[List[Path]]) -> List[Path]:
    return [p for sub in paths for p in sub]


# ------------- AUROC (global) -------------
def run_auroc(args):
    csv_paths = flatten_paths(args.pred_csv)
    control_source_names = {Path(s).name for s in args.control_sources}

    dfs = []
    for p in csv_paths:
        force_control = args.controls_label if p.name in control_source_names else None
        dfs.append(load_df(p, force_control_label=force_control))
    df = pd.concat(dfs, ignore_index=True)

    # optional filters
    if args.filter_cohorts is not None:
        df = df[df["cohort"].astype(str).isin(args.filter_cohorts)]
    if args.filter_sex is not None:
        df = df[df["sex"].str.lower() == args.filter_sex.lower()]
    if args.exclude_recovery:
        df = df[~df["group"].isin(["Recovery", "High_CO2_Recovery"])]

    # day filter
    df = df[df["day"] > args.min_day + EPS]

    control_set = {args.controls_label.strip()}
    control_set.update({g.strip() for g in args.extra_control_groups})
    disease_set = {args.hls_label.strip()}
    if not args.strict_hls_only:
        disease_set.update({g.strip() for g in args.extra_disease_groups})

    control_sources = {Path(s).name for s in args.control_sources}
    df["__is_control"] = df["group"].isin(control_set) | df["__source"].apply(lambda s: Path(s).name in control_sources)
    df["__is_disease"] = df["group"].isin(disease_set)

    df = df[df["__is_control"] | df["__is_disease"]]
    if args.control_day_anchor:
        df, _ = apply_control_day_anchor(df, df["__is_control"])
    if df.empty or df["__is_disease"].nunique() == 1:
        raise SystemExit("No rows or no class variety after filtering.")

    # Cohort-specific centering (controls mean -> 0) unless skipped
    ctrl_mask = df["__is_control"]
    if ctrl_mask.any():
        if getattr(args, "skip_cohort_center", False):
            print("[INFO] Skipping cohort centering (requested).")
        else:
            baselines = df[ctrl_mask].groupby("cohort")["RAG"].mean()
            df["RAG_raw"] = df["RAG"]
            df["RAG"] = df["RAG"] - df["cohort"].map(baselines).fillna(0.0)
            print("[INFO] Cohort control baselines (RAG):")
            print(baselines.to_string())

    if args.show_delta:
        tbl = (
            df[df["__is_control"] | df["__is_disease"]]
            .groupby(["day", "__is_disease"])["RAG"]
            .mean()
            .unstack(fill_value=np.nan)
        )
        tbl.columns = ["Control_mean", "Disease_mean"]
        tbl["Delta_RAG"] = tbl["Disease_mean"] - tbl["Control_mean"]
        print("[DELTA] Mean RAG by day (disease - control):")
        print(tbl.round(3).to_string())

    df_subject = df.groupby(["rat_id", "day", "__is_disease"])["RAG"].mean().reset_index()
    df_subject["label"] = df_subject["__is_disease"].astype(int)
    if df_subject["label"].nunique() < 2:
        raise SystemExit("Need both classes for AUROC after aggregation.")

    auroc = roc_auc_score(df_subject["label"], df_subject["RAG"])

    print("[INFO] group counts after filtering:")
    print(df["group"].value_counts())
    print("[INFO] day range:", float(df["day"].min()), "to", float(df["day"].max()))
    print("[INFO] source counts:")
    print(df["__source"].value_counts())
    print(f"[INFO] Aggregated {len(df)} eyes into {len(df_subject)} subject-day rows.")
    print(f"[AUROC] RAG AUROC (Controls vs disease, day > {args.min_day}): {auroc:.4f}")


# ------------- AUROC per cohort -------------
def get_auroc(df_subset: pd.DataFrame, control_label: str, disease_label: str):
    df_binary = df_subset[df_subset["group"].isin([control_label, disease_label])].copy()
    df_binary["label"] = (df_binary["group"] == disease_label).astype(int)
    if df_binary["label"].nunique() < 2:
        return None, 0, "N/A"
    df_subject = df_binary.groupby(["rat_id", "day", "label"])["RAG"].mean().reset_index()
    n_subjects = len(df_subject)
    try:
        score = roc_auc_score(df_subject["label"], df_subject["RAG"])
    except ValueError:
        return None, n_subjects, "Error"
    direction = "Older"
    if score < 0.5:
        score = 1.0 - score
        direction = "Younger"
    return score, n_subjects, direction


def run_auroc_cohort(args):
    csv_paths = flatten_paths(args.pred_csv)
    dfs = [load_df(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df["RAG"] = df["age_pred"] - df["age_true"]
    df = df[df["day"] > args.min_day + EPS]
    if args.exclude_recovery:
        df = df[~df["group"].isin(["Recovery", "High_CO2_Recovery"])]
    if args.control_day_anchor:
        ctrl_mask = df["group"].isin(["Controls", "High_CO2_Controls"])
        df, _ = apply_control_day_anchor(df, ctrl_mask)

    cohorts = sorted(df["cohort"].unique())
    print(f"\n[INFO] Found cohorts: {cohorts}")
    print(f"{'Cohort':<10} | {'Baseline (Control)':<22} | {'Disease Group':<20} | {'N (Subj)':<8} | {'AUROC':<6} | {'Effect'}")
    print("-" * 100)

    for coh in cohorts:
        df_coh = df[df["cohort"] == coh].copy()
        groups = df_coh["group"].unique().tolist()
        baseline = None
        if "Controls" in groups:
            baseline = "Controls"
        elif "High_CO2_Controls" in groups:
            baseline = "High_CO2_Controls"
        if baseline is None:
            continue
        disease_groups = [g for g in groups if g != baseline]
        for dis in disease_groups:
            score, n, direction = get_auroc(df_coh, baseline, dis)
            if score is not None:
                print(f"{coh:<10} | {baseline:<22} | {dis:<20} | {n:<8} | {score:.3f}  | {direction}")
    print("-" * 100)


# ------------- Phases (Suspension vs Recovery) -------------
def load_and_normalize_phases(csv_paths: List[Path], keep_cohorts: List[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df["cohort"] = df["cohort"].astype(str).str.strip()
    df = df[df["cohort"].isin(keep_cohorts)]
    df["RAG"] = df["age_pred"] - df["age_true"]
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


def run_stats_for_phase(df: pd.DataFrame, phase_name: str, min_day: float, max_day: float, baseline_mode: str):
    subset = df[(df["day"] >= min_day) & (df["day"] <= max_day)]
    hls = subset[subset["group"] == "HLS (U)"]
    ctrl = subset[subset["group"] == "Controls"]
    hls_subj = hls.groupby(["rat_id"])["RAG_Norm"].mean()
    ctrl_subj = ctrl.groupby(["rat_id"])["RAG_Norm"].mean()
    mean_hls = hls_subj.mean() if not hls_subj.empty else float("nan")
    mean_ctrl = ctrl_subj.mean() if not ctrl_subj.empty else 0.0
    if len(hls_subj) < 2:
        return phase_name, mean_hls, mean_ctrl, float("nan"), "Insufficient Data"
    if baseline_mode == "concurrent" and len(ctrl_subj) > 2:
        _, p = ttest_ind(hls_subj, ctrl_subj, equal_var=False)
        test_type = "T-Test vs Controls"
    else:
        _, p = ttest_1samp(hls_subj, 0.0)
        test_type = "1-Sample T-Test vs 0"
    return phase_name, mean_hls, mean_ctrl, p, test_type


def run_phases(args):
    df = load_and_normalize_phases(flatten_paths(args.pred_csv), args.cohorts)
    print("\n" + "=" * 80)
    print(f"TWO-PHASE ANALYSIS (Cohorts: {args.cohorts})")
    print("=" * 80)
    p1 = run_stats_for_phase(df, "Suspension (14-90d)", 14, 90, baseline_mode="concurrent")
    p2 = run_stats_for_phase(df, "Recovery (>90d)", 91, 180, baseline_mode="zero")
    print(f"{'Phase':<20} | {'HLS RAG':<10} | {'Ctrl RAG':<10} | {'P-Value':<10} | {'Conclusion'}")
    print("-" * 80)
    for name, h_mu, c_mu, p, test in [p1, p2]:
        sig_str = "**SIG**" if p < 0.05 else "n.s."
        print(f"{name:<20} | {h_mu:>6.2f}d    | {c_mu:>6.2f}d    | {p:.4f} {sig_str} | {test}")
    print("-" * 80)


# ------------- Longitudinal Plot -------------
def load_and_normalize_plot(csv_paths: List[Path], keep_cohorts: List[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df["cohort"] = df["cohort"].astype(str).str.strip()
    df = df[df["cohort"].isin(keep_cohorts)]
    df["RAG"] = df["age_pred"] - df["age_true"]
    norm_rows = []
    print(f"[INFO] Normalizing cohorts {keep_cohorts} using control medians...")
    for coh in keep_cohorts:
        subset = df[df["cohort"] == coh].copy()
        controls = subset[subset["group"] == "Controls"]
        baseline = controls["RAG"].median() if not controls.empty else 0.0
        subset["RAG_Norm"] = subset["RAG"] - baseline
        norm_rows.append(subset)
    df_norm = pd.concat(norm_rows, ignore_index=True)
    df_norm = df_norm[df_norm["group"].isin(["Controls", "HLS (U)"])]
    df_norm["Phase"] = df_norm["day"].apply(lambda d: "Recovery (>90d)" if d > 90 else "Suspension (<=90d)")
    return df_norm


def run_plot(args):
    df = load_and_normalize_plot(flatten_paths(args.pred_csv), args.cohorts)
    # Stats for recovery
    rec_hls = df[(df["group"] == "HLS (U)") & (df["Phase"] == "Recovery (>90d)")]
    rec_ctrl = df[(df["group"] == "Controls") & (df["Phase"] == "Recovery (>90d)")]
    rec_hls_subj = rec_hls.groupby(["rat_id"])["RAG_Norm"].mean()
    rec_ctrl_subj = rec_ctrl.groupby(["rat_id"])["RAG_Norm"].mean()
    if len(rec_hls_subj) > 0 and len(rec_ctrl_subj) > 0:
        _, p_val = ttest_ind(rec_hls_subj, rec_ctrl_subj, equal_var=False)
    else:
        p_val = float("nan")

    print("-" * 60)
    print(f"[STATS] Recovery Phase (Day > 90) Analysis (Cohorts: {args.cohorts})")
    print(f"   HLS Mean RAG: {rec_hls_subj.mean():.2f} days" if len(rec_hls_subj) else "   HLS Mean RAG: n/a")
    print(f"   Ctrl Mean RAG: {rec_ctrl_subj.mean():.2f} days" if len(rec_ctrl_subj) else "   Ctrl Mean RAG: n/a")
    print(f"   P-Value (HLS vs Control): {p_val:.4f}")
    print("-" * 60)

    # Plot with matplotlib
    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(True, alpha=0.3)
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


# ------------- Main dispatcher -------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified evaluation suite")
    p.add_argument("--task", choices=["auroc", "auroc-cohort", "phases", "plot", "mad"], required=True)
    p.add_argument("--pred-csv", type=Path, nargs="+", action="append", required=True,
                   help="Prediction CSVs (per rat/eye/day) from run.py; repeat flag allowed")
    # Common filters
    p.add_argument("--min-day", type=float, default=0.0, help="Exclude days <= this value")
    p.add_argument("--exclude-recovery", action="store_true", help="Drop Recovery/High_CO2_Recovery rows (where applicable)")
    p.add_argument("--control-day-anchor", action="store_true",
                   help="Center RAG per day using Control means (after linear bias correction)")
    p.add_argument("--filter-cohorts", type=str, nargs="*", default=None, help="Optional cohort filter (auroc)")
    p.add_argument("--filter-sex", type=str, default=None, help="Optional sex filter (auroc)")
    p.add_argument("--show-delta", action="store_true", help="Print per-day mean RAG for control vs disease and their delta")
    # AUROC-specific
    p.add_argument("--controls-label", type=str, default="Controls")
    p.add_argument("--hls-label", type=str, default="HLS (U)")
    p.add_argument("--extra-control-groups", type=str, nargs="*", default=[])
    p.add_argument("--extra-disease-groups", type=str, nargs="*", default=[])
    p.add_argument("--control-sources", type=str, nargs="*", default=[])
    p.add_argument("--strict-hls-only", action="store_true")
    p.add_argument("--skip-cohort-center", action="store_true", help="Do not subtract cohort control means before AUROC")
    # Cohort/phase/plot
    p.add_argument("--cohorts", nargs="*", default=["1", "3"], help="Cohort IDs for phases/plot tasks")
    # MAD
    p.add_argument("--mad-groups", type=str, nargs="*", default=["Controls"], help="Groups to include for MAD (default: Controls)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.task == "auroc":
        run_auroc(args)
    elif args.task == "auroc-cohort":
        run_auroc_cohort(args)
    elif args.task == "phases":
        run_phases(args)
    elif args.task == "plot":
        run_plot(args)
    elif args.task == "mad":
        run_mad(args)
    else:
        raise SystemExit("Unknown task")


if __name__ == "__main__":
    main()
