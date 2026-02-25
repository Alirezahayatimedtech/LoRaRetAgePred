#!/usr/bin/env python3
"""Assemble a compact Paper #1 (control-only) results bundle from completed runs.

This script does not rerun experiments. It copies/symlinks existing summaries,
tables, and figures into ``outputs/paper1`` and writes an execution status report
that marks supported/unsupported steps in the current codebase.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import numpy as np


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, summary: list[dict], required: bool = True) -> bool:
    if not src.exists():
        summary.append({"action": "copy", "src": str(src), "dst": str(dst), "status": "missing"})
        if required:
            return False
        return False
    ensure_parent(dst)
    shutil.copy2(src, dst)
    summary.append({"action": "copy", "src": str(src), "dst": str(dst), "status": "ok"})
    return True


def write_text(dst: Path, text: str, summary: list[dict]) -> None:
    ensure_parent(dst)
    dst.write_text(text)
    summary.append({"action": "write", "dst": str(dst), "status": "ok"})


def build_distillation_summary(root: Path, out_csv: Path, summary: list[dict]) -> None:
    pred_root = root / "outputs" / "predictions"
    rows = []

    def _load_metrics(run_dir: Path) -> dict | None:
        p = run_dir / "metrics_summary.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        if "split" in df.columns:
            c = df[df["split"].astype(str).str.lower() == "control"]
            if not c.empty:
                return c.iloc[0].to_dict()
        return df.iloc[0].to_dict() if not df.empty else None

    def _control_pred_csv(run_dir: Path) -> Path | None:
        for name in ("control_val_results.csv", "control_test_results.csv", "control_results.csv"):
            p = run_dir / name
            if p.exists():
                return p
        return None

    def _inter_eye_stats(run_dir: Path) -> tuple[float | None, float | None]:
        p = None
        for name in (
            "control_val_inter_eye_differences.csv",
            "control_test_inter_eye_differences.csv",
            "control_inter_eye_differences.csv",
        ):
            cand = run_dir / name
            if cand.exists():
                p = cand
                break
        if p is not None:
            df = pd.read_csv(p)
            if "age_pred_inter_eye_abs" in df.columns and not df.empty:
                s = pd.to_numeric(df["age_pred_inter_eye_abs"], errors="coerce").dropna()
                if not s.empty:
                    return float(s.mean()), float(s.quantile(0.95))

        # Fallback: build paired inter-eye absolute differences from per-eye control predictions.
        pred = _control_pred_csv(run_dir)
        if pred is None:
            return None, None
        df = pd.read_csv(pred)
        need = {"rat_id", "day", "eye", "age_pred"}
        if not need.issubset(df.columns):
            return None, None
        x = df.copy()
        x["eye"] = x["eye"].astype(str).str.upper().str.strip()
        x["age_pred"] = pd.to_numeric(x["age_pred"], errors="coerce")
        x = x.dropna(subset=["age_pred"])
        if x.empty:
            return None, None
        per_eye = x.groupby(["rat_id", "day", "eye"], as_index=False)["age_pred"].mean()
        piv = per_eye.pivot_table(index=["rat_id", "day"], columns="eye", values="age_pred", aggfunc="first").reset_index()
        if not {"OD", "OS"}.issubset(piv.columns):
            return None, None
        s = (pd.to_numeric(piv["OD"], errors="coerce") - pd.to_numeric(piv["OS"], errors="coerce")).abs().dropna()
        if s.empty:
            return None, None
        return float(s.mean()), float(s.quantile(0.95))

    def _day90_mae(run_dir: Path) -> float | None:
        p = _control_pred_csv(run_dir)
        if p is None:
            return None
        df = pd.read_csv(p)
        if not {"day", "age_true", "age_pred"}.issubset(df.columns):
            return None
        d = df.copy()
        d["day"] = pd.to_numeric(d["day"], errors="coerce")
        d["age_true"] = pd.to_numeric(d["age_true"], errors="coerce")
        d["age_pred"] = pd.to_numeric(d["age_pred"], errors="coerce")
        d = d[(d["day"] == 90) & d["age_true"].notna() & d["age_pred"].notna()]
        if d.empty:
            return None
        return float((d["age_pred"] - d["age_true"]).abs().mean())

    runs = [
        ("xception_baseline", pred_root / "xception_e40_lr1e3_d0090_c123", "baseline"),
        ("xception_human_retfound_a03", pred_root / "xception_e40_lr1e3_d0090_c123_distill_retfoundfeat_a03_b8", "human_teacher"),
        ("xception_rat_retfound_a01", pred_root / "xception_e40_lr1e3_d0090_c123_distill_ratretfoundfeat_a01_b8", "rat_teacher"),
        ("xception_rat_retfound_a005", pred_root / "xception_e40_lr1e3_d0090_c123_distill_ratretfoundfeat_a005_b8", "rat_teacher"),
    ]
    for label, run_dir, teacher_type in runs:
        m = _load_metrics(run_dir)
        ie_mean, ie_q95 = _inter_eye_stats(run_dir)
        row = {
            "model": label,
            "teacher_type": teacher_type,
            "run_dir": str(run_dir),
            "available": bool(m is not None),
            "mae": None,
            "rmse": None,
            "r2": None,
            "pearson_r": None,
            "inter_eye_mean": ie_mean,
            "inter_eye_q95": ie_q95,
            "day90_mae": _day90_mae(run_dir),
            "note": "",
        }
        if m is None:
            row["note"] = "run not found (optional distillation variant not executed)"
        else:
            for k in ("mae", "rmse", "r2", "pearson_r"):
                if k in m:
                    row[k] = float(m[k])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    ensure_parent(out_csv)
    out_df.to_csv(out_csv, index=False)
    summary.append({"action": "write", "dst": str(out_csv), "status": "ok", "rows": len(out_df)})


def build_execution_report(root: Path, out_md: Path, summary: list[dict]) -> None:
    txt = f"""# Paper #1 (Control-Only Age Prediction) Execution Report

This bundle was assembled from completed experiments in the current codebase.

## Scope
- Dataset: OSD-679 (rat OCT)
- Cohorts: 1/2/3 (Cohort 4 excluded)
- Primary days: 0 and 90 (control-priority protocol)
- Rat-level splits

## Executed / Reused Results

### EXP-01 (Control-only CV, RETFound+MAE+MIL+LoRA4)
- Source: `outputs/core/exp01_ctrl_cv/`
- Aggregated summary: `outputs/paper1/exp01_retfound_lora/summary.csv`
- Status: completed and reused

### EXP-02 (Backbone ablation, supported subset)
- Source: `outputs/ablation/`
- RETFound + Xception completed
- Random ViT baseline not implemented in current `run.py`
- Aggregated summary: `outputs/paper1/ablation/backbone_comparison.csv`

### EXP-03 (LoRA vs Full FT vs Frozen head-only)
- Status: not fully runnable in current codebase as specified
- Reason: clean full RETFound fine-tuning path (MIL) is not exposed via `run.py`; current implementation is PEFT-focused (LoRA/frozen variants).
- Placeholder note written under `outputs/paper1/lora_ablation/README_skipped.txt`

### EXP-04 (Inter-eye reliability)
- Source: `outputs/core/exp03_inter_eye/summary.csv`
- Status: completed and reused

### EXP-05 (Saliency)
- Source: `outputs/core/exp04_saliency/day90_examples/`
- Status: completed and reused
- Note: CLS-only checkpoints use gradient-saliency fallback (patched script)

### EXP-06 (Optional distillation summary)
- Built from existing Xception + distillation runs
- Summary: `outputs/paper1/distillation_summary.csv`
- Includes baseline, human-teacher α=0.3, rat-adapted-teacher α=0.1
- Rat-adapted α=0.05 row is marked unavailable if not run

## Added Generalization Experiment (LOCO Cohort)
- True LOCO support was patched into `RETFoundLoRA/run.py` and `RETFoundLoRA/preprocess_age_lora.py`
- Outputs: `outputs/generalization/exp07_loo_mae50_mil_lora4_d0090/`
- Aggregate: `outputs/generalization/exp07_loo_mae50_mil_lora4_d0090/loo_summary.csv`

## Main Claims Supported by Current Results
1. On the narrow control-only day 0/90 benchmark, Xception is a stronger baseline than the current RETFound+MIL configuration (backbone ablation).
2. RETFound pipelines remain useful for broader analyses (CV, saliency, OOD/HLS summaries) and for structured experimentation (MIL, LoRA, SSL variants).
3. Human RETFound feature distillation into Xception (α=0.3) degrades control-priority performance; rat-adapted teacher distillation (α=0.1) improves RMSE/R² vs human-teacher distill but does not beat plain Xception on control MAE/inter-eye reliability.
4. LOCO cohort results confirm severe cross-cohort generalization failure for held-out cohort 3 when training excludes the older age regime (age extrapolation dominates).

## Reproducibility Notes
- Generated `outputs/` are local experiment artifacts; only compact summaries/reports are intended for GitHub.
- This report reflects the current codebase behavior, not the exact hypothetical CLI from the provided runbook.
"""
    write_text(out_md, txt, summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out-root", type=Path, default=Path("outputs/paper1"))
    args = ap.parse_args()

    root = args.root.resolve()
    out_root = (root / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    actions: list[dict] = []

    # Core summaries
    copy_file(root / "outputs/core/exp01_ctrl_cv/summary.csv", out_root / "exp01_retfound_lora/summary.csv", actions)
    copy_file(root / "outputs/ablation/backbone_comparison.csv", out_root / "ablation/backbone_comparison.csv", actions)
    copy_file(root / "outputs/core/exp03_inter_eye/summary.csv", out_root / "exp04_inter_eye/summary.csv", actions)

    # Saliency pointer (avoid copying ~1GB)
    saliency_note = (
        "Saliency outputs were generated and retained in-place to avoid duplicating large files.\n"
        "Source directory: outputs/core/exp04_saliency/day90_examples/\n"
        "Use scripts/save_saliency_subset.py and scripts/generate_paper_figures.py for regeneration.\n"
    )
    write_text(out_root / "exp05_saliency/README.txt", saliency_note, actions)

    # Unsupported/skipped EXP-03 note
    lora_note = (
        "EXP-03 (LoRA vs full fine-tuning vs frozen head-only) was not executed as specified.\n"
        "Reason: a clean full RETFound fine-tuning path in the MIL pipeline is not exposed in current run.py.\n"
        "Existing results focus on LoRA-based PEFT and backbone ablations.\n"
    )
    write_text(out_root / "lora_ablation/README_skipped.txt", lora_note, actions)

    # Distillation summary (optional experiment summary from existing runs)
    build_distillation_summary(root, out_root / "distillation_summary.csv", actions)

    # Copy paper-ready tables / figures generated earlier
    tables_src = root / "outputs/paper_ready/tables"
    figs_src = root / "outputs/paper_ready/figures"
    if tables_src.exists():
        for p in sorted(tables_src.glob("*.csv")):
            copy_file(p, out_root / "tables" / p.name, actions, required=False)
    if figs_src.exists():
        for p in sorted(figs_src.glob("*.png")):
            copy_file(p, out_root / "figures" / p.name, actions, required=False)
        copy_file(figs_src / "generated_figures_index.csv", out_root / "figures/generated_figures_index.csv", actions, required=False)

    # LOCO aggregate copies (newly executed)
    loco_root = root / "outputs/generalization/exp07_loo_mae50_mil_lora4_d0090"
    for name in ("loo_summary.csv", "loo_summary_rows.csv", "loo_summary_cohort_day_rows.csv", "loo_summary_with_inter_eye.csv"):
        copy_file(loco_root / name, out_root / "exp07_loo_cohort" / name, actions, required=False)

    # Execution report + manifest
    build_execution_report(root, out_root / "EXECUTION_REPORT.md", actions)
    ensure_parent(out_root / "bundle_manifest.json")
    (out_root / "bundle_manifest.json").write_text(json.dumps({"actions": actions}, indent=2))

    print(f"[BUNDLE] Wrote compact Paper #1 bundle to {out_root}")
    print(f"[BUNDLE] Manifest: {out_root / 'bundle_manifest.json'}")


if __name__ == "__main__":
    main()
