#!/usr/bin/env python3
"""
Compute ΔRAG per rat (subtract Day 0 baseline) and plot trajectories by cohort.
Outputs:
  - outputs/predictions/rag_results_delta.csv
  - outputs/predictions/FINAL_DELTA_RAG_PLOT.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    pred_dir = Path("outputs/predictions")
    exp_csv = pred_dir / "rag_experimental_results.csv"
    ctrl_csv = pred_dir / "control_test_results.csv"
    out_csv = pred_dir / "rag_results_delta.csv"
    out_plot = pred_dir / "FINAL_DELTA_RAG_PLOT.png"

    missing = [p for p in (exp_csv, ctrl_csv) if not p.exists()]
    if missing:
        print(f"[ERROR] Missing input files: {', '.join(str(p) for p in missing)}")
        sys.exit(1)

    print(f"[INFO] Loading {exp_csv} and {ctrl_csv}")
    df_exp = pd.read_csv(exp_csv)
    df_ctrl = pd.read_csv(ctrl_csv)
    df = pd.concat([df_ctrl, df_exp], ignore_index=True)
    print(f"[INFO] Total rows: {len(df)}")

    # Baseline RAG per rat at day 0 (mean if duplicates)
    baseline_df = df[df["day"] == 0]
    rat_baselines = baseline_df.groupby("rat_id")["RAG"].mean().to_dict()

    def get_delta(row):
        baseline = rat_baselines.get(row["rat_id"])
        return row["RAG"] - baseline if baseline is not None else None

    df["Delta_RAG"] = df.apply(get_delta, axis=1)

    n_before = len(df)
    df_clean = df.dropna(subset=["Delta_RAG"])
    dropped = n_before - len(df_clean)
    if dropped:
        print(f"[WARN] Dropped {dropped} rows (no Day 0 baseline for those rats).")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_csv, index=False)
    print(f"[DATA] Saved ΔRAG data to {out_csv}")

    # Plot
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    palette = {
        "Controls": "blue",
        "HLS (U)": "red",
        "Recovery": "green",
        "High_CO2_Controls": "cyan",
        "High_CO2_HLS": "orange",
        "High_CO2_Recovery": "purple",
    }

    g = sns.FacetGrid(
        df_clean,
        col="cohort",
        col_wrap=2,
        height=5,
        aspect=1.2,
        sharex=True,
        sharey=True,
    )
    g.map_dataframe(
        sns.lineplot,
        x="day",
        y="Delta_RAG",
        hue="group",
        style="group",
        markers=True,
        dashes=False,
        err_style="bars",
        errorbar="se",
        linewidth=2.5,
        palette=palette,
    )

    for ax in g.axes.flat:
        ax.axhline(0, color="gray", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_ylabel("Δ Retinal Age Gap (Days)")
        ax.set_xlabel("Days")

    g.add_legend(title="Group", adjust_subtitles=True)
    g.fig.suptitle("Change in Retinal Age Gap (Baseline Corrected)", y=1.02)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    print(f"[DONE] Plot saved to {out_plot}")


if __name__ == "__main__":
    main()
