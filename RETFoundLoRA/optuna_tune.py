#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for RETFound LoRA age regression.

Example:
  python3 RETFoundLoRA/optuna_tune.py --n-trials 20 --epochs 10 --batch-size 16
"""

import argparse
import subprocess
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def parse_args():
    p = argparse.ArgumentParser(description="Optuna tuner for RETFoundLoRA")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16, help="Default batch size if batch-size tuning is disabled")
    p.add_argument("--tune-batch-size", action="store_true", default=True, help="Include batch size in search (2–16)")
    p.add_argument("--fixed-batch-size", action="store_false", dest="tune_batch_size", help="Disable batch-size tuning")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--metric", type=str, default="r2", choices=["mae", "rmse", "r2"])
    p.add_argument("--study-name", type=str, default="retfoundlora_optuna")
    p.add_argument("--storage", type=str, default="sqlite:///outputs/optuna/retfoundlora.db")
    p.add_argument("--reset-study", action="store_true",
                   help="Delete existing study storage before running (use with caution)")
    p.add_argument("--direction", type=str, default="minimize", choices=["minimize", "maximize"],
                   help="Direction for Optuna; default minimize for bias-aware objective")
    p.add_argument("--device", type=str, default=None, help="Override device for run.py (e.g., cpu)")
    p.add_argument("--early-stop-patience", type=int, default=6)
    p.add_argument("--trials-csv", type=Path, default=Path("outputs/optuna/optuna_trials.csv"))
    p.add_argument("--best-csv", type=Path, default=Path("outputs/optuna/best_trial.csv"),
                   help="Write/update best trial immediately when improved")
    p.add_argument("--controls-only", action="store_true", default=True,
                   help="Tune on Controls only (default: on)")
    p.add_argument("--include-stress", action="store_false", dest="controls_only",
                   help="Include stress groups in each trial")
    p.add_argument("--use-bias-correction", action="store_true",
                   help="Enable bias correction during tuning (off by default to avoid leakage on val)")
    return p.parse_args()


def bias_aware_score(control_pred_csv: Path):
    """
    Compute bias-aware objective:
      score = MAE_after_correction + 20 * |ADC|
      ADC = corr(age_true, RAG) after linear bias correction.
    Lower is better.
    """
    if not control_pred_csv.exists():
        return float("inf"), None, None
    df = pd.read_csv(control_pred_csv)
    required = {"age_true", "age_pred"}
    if df.empty or not required.issubset(df.columns):
        return float("inf"), None, None
    y_true = df["age_true"].to_numpy(dtype=float)
    y_pred = df["age_pred"].to_numpy(dtype=float)
    # Fit bias correction on controls
    alpha, beta = fit_linear_correction(y_true, y_pred)
    y_corr = apply_correction(y_true, y_pred, alpha, beta)
    rag = y_corr - y_true
    mae = float(np.mean(np.abs(rag)))
    adc = float(abs(pearsonr(y_true, rag)[0]))
    score = mae + 20.0 * adc
    row = {
        "mae_corrected": mae,
        "adc_abs": adc,
        "alpha": alpha,
        "beta": beta,
        "score": score,
    }
    return score, row, df


def main():
    args = parse_args()
    try:
        import optuna
    except Exception:
        raise SystemExit("Optuna is not installed. Install with `pip install optuna`.")

    out_root = Path("outputs/optuna")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.reset_study:
        m = re.match(r"sqlite:///([^?]+)", args.storage)
        if m:
            db_path = Path(m.group(1))
            if db_path.exists():
                try:
                    os.remove(db_path)
                    print(f"[RESET] Removed existing Optuna DB at {db_path}")
                except Exception as e:
                    print(f"[RESET] Could not remove Optuna DB {db_path}: {e}")

    best_score = float("inf") if args.direction == "minimize" else -float("inf")

    def objective(trial: "optuna.Trial") -> float:
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        lora_rank = trial.suggest_categorical("lora_rank", [4, 8, 16, 32])
        lora_alpha = trial.suggest_categorical("lora_alpha", [16.0, 32.0, 64.0])
        lora_dropout = trial.suggest_float("lora_dropout", 0.05, 0.30)
        label_noise = trial.suggest_float("label_noise_std", 0.5, 3.0)
        aug_level = trial.suggest_categorical("aug_level", ["low", "medium", "high"])
        batch_size = (
            trial.suggest_int("batch_size", 2, 16)
            if args.tune_batch_size
            else args.batch_size
        )

        trial_dir = out_root / f"trial{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = trial_dir / "metrics.csv"
        control_pred_csv = trial_dir / "control_val_results.csv"
        save_lora = trial_dir / "weights.skip"
        pred_csv = trial_dir / "predictions.skip"

        cmd = [
            "python3", "RETFoundLoRA/run.py",
            "--epochs", str(args.epochs),
            "--batch-size", str(batch_size),
            "--img-size", str(args.img_size),
            "--lr", str(lr),
            "--lora-rank", str(lora_rank),
            "--lora-alpha", str(lora_alpha),
            "--lora-dropout", str(lora_dropout),
            "--label-noise-std", str(label_noise),
            "--aug-level", aug_level,
            "--early-stop-patience", str(args.early_stop_patience),
            "--metrics-csv", str(metrics_csv),
            "--save-lora", str(save_lora),
            "--pred-csv", str(pred_csv),
            "--no-save-lora",
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.controls_only:
            # empty test-groups => Controls-only split/metrics
            cmd += ["--test-groups"]
        # avoid disk errors on weight save during tuning
        if not args.use_bias_correction:
            cmd += ["--no-bias-correction"]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
        rc = proc.wait()
        if rc != 0:
            # Penalize failed trials heavily instead of pruning
            return float("inf")

        score, row, _ = bias_aware_score(control_pred_csv)
        trial.set_user_attr("control_pred_csv", str(control_pred_csv))
        if row:
            for k, v in row.items():
                trial.set_user_attr(f"control_{k}", v)
        nonlocal best_score
        is_better = (args.direction == "minimize" and score < best_score) or (
            args.direction == "maximize" and score > best_score
        )
        if is_better:
            best_score = score
            args.best_csv.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "trial": trial.number,
                "metric": args.metric,
                "direction": args.direction,
                "value": score,
                "metrics_csv": str(metrics_csv),
                "weights": str(save_lora),
                "preds": str(pred_csv),
                "lr": lr,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "label_noise_std": label_noise,
                "aug_level": aug_level,
                "batch_size": batch_size,
            }
            if row:
                for k, v in row.items():
                    record[f"control_{k}"] = v
            pd.DataFrame([record]).to_csv(args.best_csv, index=False)
            print(f"[OPTUNA] Updated best -> {args.best_csv} (value={score:.4f})")
        return score

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("[OPTUNA] Best trial:")
    print(study.best_trial)

    df = study.trials_dataframe()
    args.trials_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.trials_csv, index=False)
    print(f"[OPTUNA] Saved trials to {args.trials_csv}")


if __name__ == "__main__":
    main()
