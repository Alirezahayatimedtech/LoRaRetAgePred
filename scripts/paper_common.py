#!/usr/bin/env python3
"""Shared helpers for paper-style experiment aggregation scripts.

These utilities are intentionally tolerant to output variations across runs
(RETFound, Xception, MIL/non-MIL, control-only runs, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd


CONTROL_CSV_CANDIDATES = (
    "control_val_results.csv",
    "control_test_results.csv",
    "control_results.csv",
)
STRESS_CSV_CANDIDATES = (
    "rag_experimental_results.csv",
    "stress_results.csv",
    "test_results.csv",
)
INTER_EYE_SUFFIX = "_inter_eye_differences.csv"


@dataclass
class RunFiles:
    run_dir: Path
    metrics_csv: Optional[Path]
    control_pred_csv: Optional[Path]
    stress_pred_csv: Optional[Path]
    control_inter_eye_csv: Optional[Path]
    stress_inter_eye_csv: Optional[Path]


def _first_existing(parent: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = parent / name
        if p.exists():
            return p
    return None


def discover_run_files(run_dir: Path) -> RunFiles:
    run_dir = Path(run_dir)
    metrics_csv = run_dir / "metrics_summary.csv"
    if not metrics_csv.exists():
        metrics_csv = None

    control_pred_csv = _first_existing(run_dir, CONTROL_CSV_CANDIDATES)
    stress_pred_csv = _first_existing(run_dir, STRESS_CSV_CANDIDATES)

    def paired_from_pred(pred_path: Optional[Path]) -> Optional[Path]:
        if pred_path is None:
            return None
        if "_results" in pred_path.name:
            paired = pred_path.with_name(pred_path.name.replace("_results", "_inter_eye_differences", 1))
        else:
            paired = pred_path.with_name(pred_path.stem + INTER_EYE_SUFFIX)
        return paired if paired.exists() else None

    return RunFiles(
        run_dir=run_dir,
        metrics_csv=metrics_csv,
        control_pred_csv=control_pred_csv,
        stress_pred_csv=stress_pred_csv,
        control_inter_eye_csv=paired_from_pred(control_pred_csv),
        stress_inter_eye_csv=paired_from_pred(stress_pred_csv),
    )


def collect_run_dirs(root_dirs: Sequence[Path], metrics_name: str = "metrics_summary.csv") -> List[Path]:
    found: List[Path] = []
    for root in root_dirs:
        root = Path(root)
        if root.is_file() and root.name == metrics_name:
            found.append(root.parent)
            continue
        if root.is_dir():
            for p in root.rglob(metrics_name):
                found.append(p.parent)
    # stable unique
    uniq = []
    seen = set()
    for p in sorted(found):
        sp = str(p.resolve())
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p)
    return uniq


def infer_fold_index(path: Path) -> Optional[int]:
    text = str(path)
    for pat in (r"/fold[_-]?(\d+)(?:/|$)", r"_fold(\d+)\b"):
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return None


def infer_heldout_cohort(path: Path) -> Optional[str]:
    m = re.search(r"loo[_-]?cohort(?:_|)?(\d+)", str(path), flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"cohort(\d+)", str(path), flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def normalize_pred_df(df: pd.DataFrame, source: Optional[Path] = None) -> pd.DataFrame:
    out = df.copy()
    if "group" in out.columns:
        out["group"] = out["group"].astype(str).str.strip()
    if "cohort" in out.columns:
        out["cohort"] = out["cohort"].astype(str).str.strip()
    else:
        out["cohort"] = "Unknown"
    if "sex" in out.columns:
        out["sex"] = out["sex"].astype(str).str.strip()
    else:
        out["sex"] = "Unknown"
    if "RAG" not in out.columns and {"age_pred", "age_true"} <= set(out.columns):
        out["RAG"] = pd.to_numeric(out["age_pred"], errors="coerce") - pd.to_numeric(out["age_true"], errors="coerce")
    if source is not None:
        out["__source"] = str(source)
    return out


def load_prediction_csv(path: Path) -> pd.DataFrame:
    return normalize_pred_df(pd.read_csv(path), source=path)


def build_inter_eye_pairs_from_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    required = {"rat_id", "day", "eye", "age_pred"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction dataframe missing required columns for inter-eye pairing: {sorted(missing)}")

    df = pred_df.copy()
    df["eye"] = df["eye"].astype(str).str.upper().str.strip()

    agg_cols = {}
    for col in ("age_pred", "age_true", "RAG"):
        if col in df.columns:
            agg_cols[col] = "mean"
    for col in ("group", "sex", "cohort"):
        if col in df.columns:
            agg_cols[col] = "first"
    for col in ("mil_bag_n_kept", "mil_bag_n_raw", "mil_bag_n_qc_dropped", "mil_low_conf_bag"):
        if col in df.columns:
            agg_cols[col] = "mean" if col != "mil_low_conf_bag" else "max"

    per_eye = df.groupby(["rat_id", "day", "eye"], as_index=False).agg(agg_cols)
    piv = per_eye.pivot_table(index=["rat_id", "day"], columns="eye", values=list(agg_cols.keys()), aggfunc="first")
    # flatten columns -> field_EYE
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    pair = piv.reset_index()

    # require both eyes
    if "age_pred_OD" in pair.columns and "age_pred_OS" in pair.columns:
        pair = pair.dropna(subset=["age_pred_OD", "age_pred_OS"]).copy()
        pair["age_pred_inter_eye_signed_OD_minus_OS"] = pd.to_numeric(pair["age_pred_OD"], errors="coerce") - pd.to_numeric(pair["age_pred_OS"], errors="coerce")
        pair["age_pred_inter_eye_abs"] = pair["age_pred_inter_eye_signed_OD_minus_OS"].abs()
    else:
        pair["age_pred_inter_eye_signed_OD_minus_OS"] = np.nan
        pair["age_pred_inter_eye_abs"] = np.nan

    if "RAG_OD" in pair.columns and "RAG_OS" in pair.columns:
        pair["RAG_inter_eye_signed_OD_minus_OS"] = pd.to_numeric(pair["RAG_OD"], errors="coerce") - pd.to_numeric(pair["RAG_OS"], errors="coerce")
        pair["RAG_inter_eye_abs"] = pair["RAG_inter_eye_signed_OD_minus_OS"].abs()

    # unified convenience columns
    for col in ("group", "sex", "cohort"):
        od = f"{col}_OD"
        os = f"{col}_OS"
        if od in pair.columns:
            pair[col] = pair[od]
        elif os in pair.columns:
            pair[col] = pair[os]

    if "mil_bag_n_kept_OD" in pair.columns and "mil_bag_n_kept_OS" in pair.columns:
        pair["mil_min_bag_n_kept"] = np.minimum(
            pd.to_numeric(pair["mil_bag_n_kept_OD"], errors="coerce"),
            pd.to_numeric(pair["mil_bag_n_kept_OS"], errors="coerce"),
        )
    return pair


def load_or_build_inter_eye(pair_csv: Optional[Path], pred_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if pair_csv and pair_csv.exists():
        return pd.read_csv(pair_csv)
    if pred_csv and pred_csv.exists():
        return build_inter_eye_pairs_from_predictions(load_prediction_csv(pred_csv))
    return None


def summarize_inter_eye(pair_df: Optional[pd.DataFrame], metric_col: str = "age_pred_inter_eye_abs") -> Dict[str, float]:
    if pair_df is None or pair_df.empty or metric_col not in pair_df.columns:
        return {
            "n_pairs": 0,
            "mean": np.nan,
            "median": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
        }
    vals = pd.to_numeric(pair_df[metric_col], errors="coerce").dropna().astype(float)
    if vals.empty:
        return {
            "n_pairs": 0,
            "mean": np.nan,
            "median": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "max": np.nan,
        }
    return {
        "n_pairs": int(vals.shape[0]),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "q95": float(vals.quantile(0.95)),
        "q99": float(vals.quantile(0.99)),
        "max": float(vals.max()),
    }


def compute_error_metrics(df: pd.DataFrame) -> Dict[str, float]:
    out = df.copy()
    out["age_true"] = pd.to_numeric(out["age_true"], errors="coerce")
    out["age_pred"] = pd.to_numeric(out["age_pred"], errors="coerce")
    out = out.dropna(subset=["age_true", "age_pred"])
    if out.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "rag_mean": np.nan, "rag_std": np.nan}
    err = out["age_pred"] - out["age_true"]
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((out["age_true"] - out["age_true"].mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {
        "n": int(len(out)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "rag_mean": float(err.mean()),
        "rag_std": float(err.std(ddof=1)) if len(err) > 1 else 0.0,
    }


def groupwise_prediction_metrics(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    include_inter_eye: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    valid_cols = [c for c in group_cols if c in df.columns]
    if not valid_cols:
        valid_cols = []
    grouped = [((), df)] if not valid_cols else list(df.groupby(valid_cols, dropna=False))
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row: Dict[str, object] = {}
        for c, v in zip(valid_cols, key):
            row[c] = v
        row.update(compute_error_metrics(sub))
        rows.append(row)
    out = pd.DataFrame(rows)
    if include_inter_eye and {"rat_id", "day", "eye", "age_pred"} <= set(df.columns):
        pair = build_inter_eye_pairs_from_predictions(df)
        pair_group_cols = [c for c in group_cols if c in pair.columns]
        pair_grouped = [((), pair)] if not pair_group_cols else list(pair.groupby(pair_group_cols, dropna=False))
        ie_rows: List[Dict[str, object]] = []
        for key, sub in pair_grouped:
            if not isinstance(key, tuple):
                key = (key,)
            row: Dict[str, object] = {}
            for c, v in zip(pair_group_cols, key):
                row[c] = v
            stats = summarize_inter_eye(sub)
            row.update(
                inter_eye_n_pairs=stats["n_pairs"],
                inter_eye_mean=stats["mean"],
                inter_eye_median=stats["median"],
                inter_eye_q95=stats["q95"],
                inter_eye_max=stats["max"],
            )
            ie_rows.append(row)
        ie_df = pd.DataFrame(ie_rows)
        if not out.empty:
            out = out.merge(ie_df, on=pair_group_cols, how="left") if pair_group_cols else out.join(ie_df)
        else:
            out = ie_df
    return out


def read_metrics_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "split" not in df.columns:
        # infer split from file path if older format
        df = df.copy()
        df["split"] = df["file"].astype(str).str.contains("control", case=False).map({True: "control", False: "stress"})
    return df


def safe_mkdir_for_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def flatten_metric_summary_rows(run_dir: Path, metrics_df: pd.DataFrame, model_label: Optional[str] = None) -> pd.DataFrame:
    out = metrics_df.copy()
    out["run_dir"] = str(run_dir)
    out["run_name"] = run_dir.name
    out["model_label"] = model_label or run_dir.name
    out["fold"] = infer_fold_index(run_dir)
    out["heldout_cohort"] = infer_heldout_cohort(run_dir)
    return out

