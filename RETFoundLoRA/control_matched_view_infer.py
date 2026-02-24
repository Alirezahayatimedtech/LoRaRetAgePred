#!/usr/bin/env python3
"""
Control-only matched-view MIL re-inference for inter-eye consistency analysis.

Given a paired control inter-eye CSV (OD/OS predictions per rat/day), this script:
1) Reconstructs the exact source image bags from metadata.
2) Restricts each OD/OS pair to the intersection of view families with equal counts.
3) Re-runs MIL inference on these matched subsets.
4) Writes a paired CSV with matched-view predictions and control-derived q95/q99 flags.

This is a post-hoc analysis utility for diagnosing/controling inter-eye inconsistency.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch

CUR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUR_DIR.parent
for p in (CUR_DIR, REPO_ROOT):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from config import BACKBONE_CKPT, CSV_PATH, IMAGE_TYPES, COHORTS_TO_KEEP  # noqa: E402
from retfound_lora_age_pred import RETFoundLoRAAgePred  # noqa: E402
from data_prep_age_lora import load_metadata, make_transform, _mil_view_group_from_row  # noqa: E402
from utils import normalize_eye_side  # noqa: E402


def _normalize_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["eye"] = df.apply(
        lambda r: normalize_eye_side(r.get("eye"), r.get("image_path", ""), r.get("material_type", "")),
        axis=1,
    )
    if "sex" in df.columns:
        df["sex"] = df["sex"].fillna("Unknown").astype(str).str.strip()
    else:
        df["sex"] = "Unknown"
    if "cohort_number" in df.columns:
        def _norm_cohort(v):
            if pd.isna(v):
                return None
            s = str(v).strip()
            if not s:
                return None
            try:
                fv = float(s)
                return str(int(fv)) if fv.is_integer() else str(fv)
            except Exception:
                return s
        cn = df["cohort_number"].apply(_norm_cohort)
        if "cohort" in df.columns:
            df["cohort"] = cn.fillna(df["cohort"])
        else:
            df["cohort"] = cn
    if "cohort" in df.columns:
        df["cohort"] = df["cohort"].fillna("Unknown").astype(str).str.strip()
    else:
        df["cohort"] = "Unknown"
    return df


def _load_model(args) -> RETFoundLoRAAgePred:
    model = RETFoundLoRAAgePred(
        ckpt_path=args.backbone_ckpt,
        img_size=args.img_size,
        global_pool=False,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_blocks=args.lora_blocks,
        lora_dropout=args.lora_dropout,
        upsample_factor=args.upsample_factor,
        keep_spatial_tokens=args.keep_spatial_tokens,
        use_pre_adapter=args.input_pre_adapter,
        pre_adapter_hidden_dim=args.input_pre_adapter_hidden,
        use_mil_attention=True,
        mil_attn_dim=args.mil_attn_dim,
        mil_hidden_dim=args.mil_hidden_dim,
        ordinal_num_bins=0,
        ordinal_aux_hidden_dim=0,
    )
    ckpt = torch.load(args.load_lora, map_location="cpu")
    if not (isinstance(ckpt, dict) and "backbone_lora" in ckpt):
        raise SystemExit(f"Unexpected checkpoint format (missing backbone_lora): {args.load_lora}")
    # Unmerged before loading LoRA deltas.
    model.backbone.train()
    model.backbone.load_state_dict(ckpt["backbone_lora"], strict=False)
    model.backbone.eval()
    if "head" in ckpt and ckpt["head"] is not None:
        model.head.load_state_dict(ckpt["head"], strict=False)
    if hasattr(model, "pre_adapter") and model.pre_adapter is not None and "pre_adapter" in ckpt:
        model.pre_adapter.load_state_dict(ckpt["pre_adapter"], strict=False)
    if hasattr(model, "mil_head") and model.mil_head is not None and "mil_head" in ckpt:
        model.mil_head.load_state_dict(ckpt["mil_head"], strict=False)
    model.eval()
    return model


def _safe_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _day_key(v) -> int:
    return int(round(float(v)))


def _pair_rows_schema(df: pd.DataFrame) -> pd.DataFrame:
    req = {"rat_id", "day"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Pair CSV missing required columns: {missing}")
    # Keep as-is; script handles multiple schema variants.
    return df.copy()


def _group_rows_by_view(df_eye: pd.DataFrame) -> Dict[str, List[dict]]:
    rows = []
    for _, r in df_eye.iterrows():
        d = r.to_dict()
        d["__view_key"] = _mil_view_group_from_row(r)
        rows.append(d)
    rows.sort(key=lambda x: str(x.get("image_path", "")))
    buckets: Dict[str, List[dict]] = {}
    for r in rows:
        buckets.setdefault(str(r["__view_key"]), []).append(r)
    return buckets


def _load_img_tensor(path: str, tf_eval, canonicalize_os_to_od: bool, eye: str):
    with Image.open(Path(path)).convert("RGB") as im:
        if canonicalize_os_to_od and str(eye).strip().upper() == "OS":
            im = ImageOps.mirror(im)
        return tf_eval(im)


@torch.no_grad()
def _predict_bag(model: RETFoundLoRAAgePred, img_paths: List[str], eye: str, tf_eval, device: torch.device, canonicalize_os_to_od: bool = False) -> float:
    if not img_paths:
        return float("nan")
    imgs = [_load_img_tensor(p, tf_eval, canonicalize_os_to_od, eye) for p in img_paths]
    batch = torch.stack(imgs, dim=0).to(device, non_blocking=True)
    feats = model.extract_image_features(batch)
    pred, _ = model.mil_predict_from_features(feats)
    return float(pred.view(-1)[0].detach().cpu().item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pair-csv", type=Path, required=True, help="Existing paired control inter-eye CSV")
    p.add_argument("--load-lora", type=Path, required=True, help="MIL+LoRA checkpoint")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--thresholds-json", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=CSV_PATH, help="Metadata CSV")
    p.add_argument("--backbone-ckpt", type=Path, default=BACKBONE_CKPT)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--upsample-factor", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-blocks", type=int, default=4)
    p.add_argument("--lora-dropout", type=float, default=0.20)
    p.add_argument("--mil-attn-dim", type=int, default=256)
    p.add_argument("--mil-hidden-dim", type=int, default=512)
    p.add_argument("--keep-spatial-tokens", action="store_true")
    p.add_argument("--input-pre-adapter", action="store_true")
    p.add_argument("--input-pre-adapter-hidden", type=int, default=16)
    p.add_argument("--canonicalize-os-to-od", action="store_true", help="Mirror OS before inference (off by default)")
    p.add_argument("--min-common-images", type=int, default=1)
    p.add_argument("--q95", type=float, default=0.95)
    p.add_argument("--q99", type=float, default=0.99)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if args.out_csv is None:
        args.out_csv = args.pair_csv.with_name(args.pair_csv.stem + "_matched_view.csv")
    if args.summary_json is None:
        args.summary_json = args.out_csv.with_name(args.out_csv.stem + "_summary.json")
    if args.thresholds_json is None:
        args.thresholds_json = args.out_csv.with_name(args.out_csv.stem + "_thresholds.json")

    print(f"[LOAD] Pair CSV: {args.pair_csv}")
    pair_df = _pair_rows_schema(pd.read_csv(args.pair_csv))
    print(f"[LOAD] Pair rows: {len(pair_df)}")

    print("[LOAD] Metadata...")
    meta = load_metadata(
        csv_path=args.csv,
        image_types=IMAGE_TYPES,
        day_whitelist=None,
        include_recovery_days=True,
        cohorts_to_keep=COHORTS_TO_KEEP,
        exclude_recovery_paths=False,
        verbose=False,
    )
    meta = _normalize_meta(meta)
    # Restrict to control rows only (utility is for control paired analysis).
    if "group_norm" in meta.columns:
        meta = meta[meta["group_norm"].astype(str).str.strip() == "Controls"].copy()

    # Index metadata by rat/day/eye for fast lookup.
    meta = meta.copy()
    meta["__day_key"] = meta["day"].apply(_day_key)
    key_to_rows: Dict[Tuple[str, int, str], pd.DataFrame] = {}
    for (rat, dayk, eye), grp in meta.groupby(["rat_id", "__day_key", "eye"], sort=False):
        key_to_rows[(str(rat), int(dayk), str(eye).upper())] = grp.reset_index(drop=True)

    print("[LOAD] Model...")
    model = _load_model(args).to(device)
    model.eval()
    tf_eval = make_transform(img_size=args.img_size, train=False)

    out_rows = []
    n_missing = 0
    n_no_shared = 0
    n_too_small = 0

    for _, prow in pair_df.iterrows():
        rat = str(prow.get("rat_id"))
        dayk = _day_key(prow.get("day"))
        od_df = key_to_rows.get((rat, dayk, "OD"))
        os_df = key_to_rows.get((rat, dayk, "OS"))

        base_row = {
            "rat_id": rat,
            "day": float(dayk),
            "age_true": _safe_float(prow.get("age_true", prow.get("age_true_OD", np.nan))),
            "age_true_OD": _safe_float(prow.get("age_true_OD", np.nan)),
            "age_true_OS": _safe_float(prow.get("age_true_OS", np.nan)),
            "age_pred_OD_orig": _safe_float(prow.get("age_pred_OD", np.nan)),
            "age_pred_OS_orig": _safe_float(prow.get("age_pred_OS", np.nan)),
            "orig_inter_eye_abs": _safe_float(prow.get("age_pred_inter_eye_abs", np.nan)),
            "cohort": str(prow.get("cohort", prow.get("cohort_OD", ""))),
            "group": str(prow.get("group", prow.get("group_OD", ""))),
            "sex_OD": str(prow.get("sex_OD", "")),
            "sex_OS": str(prow.get("sex_OS", "")),
        }

        if od_df is None or os_df is None or od_df.empty or os_df.empty:
            n_missing += 1
            base_row.update(
                {
                    "matched_age_pred_OD": float("nan"),
                    "matched_age_pred_OS": float("nan"),
                    "matched_inter_eye_signed_OD_minus_OS": float("nan"),
                    "matched_inter_eye_abs": float("nan"),
                    "raw_n_OD": int(len(od_df)) if od_df is not None else 0,
                    "raw_n_OS": int(len(os_df)) if os_df is not None else 0,
                    "matched_n_OD": 0,
                    "matched_n_OS": 0,
                    "shared_view_count": 0,
                    "shared_views": "",
                    "match_status": "missing_eye_bag",
                }
            )
            out_rows.append(base_row)
            continue

        od_buckets = _group_rows_by_view(od_df)
        os_buckets = _group_rows_by_view(os_df)
        shared_views = sorted(set(od_buckets.keys()) & set(os_buckets.keys()))
        selected_od: List[dict] = []
        selected_os: List[dict] = []
        per_view_counts = {}
        for vk in shared_views:
            od_rows = od_buckets[vk]
            os_rows = os_buckets[vk]
            n_keep = min(len(od_rows), len(os_rows))
            if n_keep <= 0:
                continue
            per_view_counts[vk] = int(n_keep)
            selected_od.extend(od_rows[:n_keep])
            selected_os.extend(os_rows[:n_keep])

        if not shared_views or (len(selected_od) == 0 or len(selected_os) == 0):
            n_no_shared += 1
            base_row.update(
                {
                    "matched_age_pred_OD": float("nan"),
                    "matched_age_pred_OS": float("nan"),
                    "matched_inter_eye_signed_OD_minus_OS": float("nan"),
                    "matched_inter_eye_abs": float("nan"),
                    "raw_n_OD": int(len(od_df)),
                    "raw_n_OS": int(len(os_df)),
                    "matched_n_OD": 0,
                    "matched_n_OS": 0,
                    "shared_view_count": 0,
                    "shared_views": "",
                    "match_status": "no_shared_views",
                }
            )
            out_rows.append(base_row)
            continue

        if min(len(selected_od), len(selected_os)) < int(args.min_common_images):
            n_too_small += 1
            status = "matched_too_small"
        else:
            status = "ok"

        od_paths = [str(r["image_path"]) for r in selected_od]
        os_paths = [str(r["image_path"]) for r in selected_os]

        if status == "ok":
            pred_od = _predict_bag(model, od_paths, "OD", tf_eval, device, args.canonicalize_os_to_od)
            pred_os = _predict_bag(model, os_paths, "OS", tf_eval, device, args.canonicalize_os_to_od)
            signed = float(pred_od - pred_os)
            absdiff = float(abs(signed))
        else:
            pred_od = pred_os = signed = absdiff = float("nan")

        base_row.update(
            {
                "matched_age_pred_OD": pred_od,
                "matched_age_pred_OS": pred_os,
                "matched_inter_eye_signed_OD_minus_OS": signed,
                "matched_inter_eye_abs": absdiff,
                "delta_inter_eye_abs_vs_orig": (absdiff - base_row["orig_inter_eye_abs"]) if (not math.isnan(absdiff) and not math.isnan(base_row["orig_inter_eye_abs"])) else float("nan"),
                "raw_n_OD": int(len(od_df)),
                "raw_n_OS": int(len(os_df)),
                "matched_n_OD": int(len(od_paths)),
                "matched_n_OS": int(len(os_paths)),
                "shared_view_count": int(len(shared_views)),
                "shared_views": "|".join(shared_views),
                "shared_view_counts_json": json.dumps(per_view_counts, sort_keys=True),
                "match_status": status,
            }
        )
        out_rows.append(base_row)

    out_df = pd.DataFrame(out_rows)

    valid = out_df["match_status"].eq("ok") & out_df["matched_inter_eye_abs"].notna()
    vals = pd.to_numeric(out_df.loc[valid, "matched_inter_eye_abs"], errors="coerce").dropna()
    if len(vals):
        q95 = float(vals.quantile(args.q95))
        q99 = float(vals.quantile(args.q99))
    else:
        q95 = q99 = float("nan")
    out_df["matched_control_q95_thresh"] = q95
    out_df["matched_control_q99_thresh"] = q99
    out_df["matched_inter_eye_flag_unreliable_q95"] = pd.to_numeric(out_df["matched_inter_eye_abs"], errors="coerce") > q95 if np.isfinite(q95) else False
    out_df["matched_inter_eye_flag_extreme_q99"] = pd.to_numeric(out_df["matched_inter_eye_abs"], errors="coerce") > q99 if np.isfinite(q99) else False
    if np.isfinite(q95):
        tier = np.where(pd.to_numeric(out_df["matched_inter_eye_abs"], errors="coerce") > q99, "extreme",
               np.where(pd.to_numeric(out_df["matched_inter_eye_abs"], errors="coerce") > q95, "unreliable", "ok"))
    else:
        tier = np.array(["unknown"] * len(out_df), dtype=object)
    tier[pd.to_numeric(out_df["matched_inter_eye_abs"], errors="coerce").isna()] = "unknown"
    out_df["matched_inter_eye_reliability_tier"] = tier

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    summary = {
        "pair_csv": str(args.pair_csv),
        "checkpoint": str(args.load_lora),
        "n_pairs_input": int(len(pair_df)),
        "n_pairs_output": int(len(out_df)),
        "n_missing_eye_bag": int(n_missing),
        "n_no_shared_views": int(n_no_shared),
        "n_too_small": int(n_too_small),
        "n_ok_pairs": int(valid.sum()),
        "orig_mean_inter_eye_abs": float(pd.to_numeric(out_df["orig_inter_eye_abs"], errors="coerce").mean()),
        "matched_mean_inter_eye_abs": float(pd.to_numeric(out_df.loc[valid, "matched_inter_eye_abs"], errors="coerce").mean()) if valid.any() else float("nan"),
        "orig_median_inter_eye_abs": float(pd.to_numeric(out_df["orig_inter_eye_abs"], errors="coerce").median()),
        "matched_median_inter_eye_abs": float(pd.to_numeric(out_df.loc[valid, "matched_inter_eye_abs"], errors="coerce").median()) if valid.any() else float("nan"),
        "orig_max_inter_eye_abs": float(pd.to_numeric(out_df["orig_inter_eye_abs"], errors="coerce").max()),
        "matched_max_inter_eye_abs": float(pd.to_numeric(out_df.loc[valid, "matched_inter_eye_abs"], errors="coerce").max()) if valid.any() else float("nan"),
        "matched_control_q95": q95,
        "matched_control_q99": q99,
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(args.thresholds_json, "w", encoding="utf-8") as f:
        json.dump({"metric": "matched_inter_eye_abs", "q95": q95, "q99": q99, "n_valid_pairs": int(len(vals))}, f, indent=2)

    print(f"[SAVE] Matched-view pairs: {args.out_csv} (N={len(out_df)})")
    print(f"[SAVE] Summary: {args.summary_json}")
    print(f"[SAVE] Thresholds: {args.thresholds_json}")
    print(
        "[MATCHED][CONTROL] "
        f"orig mean|OD-OS|={summary['orig_mean_inter_eye_abs']:.2f} -> "
        f"matched mean|OD-OS|={summary['matched_mean_inter_eye_abs']:.2f} "
        f"(ok_pairs={summary['n_ok_pairs']})"
    )
    print(
        "[MATCHED][CONTROL] "
        f"q95={q95:.2f} q99={q99:.2f} | "
        f"unreliable={int(out_df['matched_inter_eye_flag_unreliable_q95'].fillna(False).sum())}/{len(out_df)} "
        f"extreme={int(out_df['matched_inter_eye_flag_extreme_q99'].fillna(False).sum())}/{len(out_df)}"
    )


if __name__ == "__main__":
    main()
