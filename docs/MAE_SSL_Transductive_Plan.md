# MAE SSL Transductive Plan (RETFound -> Rat OCT)

## Goal

Improve cross-cohort and cross-domain age regression by doing MAE-style continued pretraining on unlabeled rat OCT, then re-running the current best supervised setup (`MIL + LoRA-4 + mild`).

This is a domain adaptation step, not supervised age training.

## Protocol Labels

- `Strict SSL`: unlabeled images from train-split rats only (no val/test rats)
- `Transductive SSL`: unlabeled images from all rats, including evaluation rats (no age labels used)

This repo currently supports both manifest types and makes the protocol explicit in filenames.

## What Is Implemented Here

- `RETFoundLoRA/build_ssl_manifests.py`
  - builds strict and transductive unlabeled manifests using the same metadata/split logic as supervised training
- `RETFoundLoRA/launch_mae_ssl_adapt.py`
  - prepares MAE-compatible `ImageFolder`
  - supports `--force-rgb-export` for grayscale OCT compatibility
  - supports `--skip-unreadable` to log and skip corrupt source files during export
  - autodetects common MAE CLI flags from `main_pretrain.py -h`
  - can generate a launch script or execute directly

## Status Update (Completed Transductive MAE Run)

A full transductive MAE adaptation run has been completed locally:
- backend: `facebookresearch/mae` (`main_pretrain.py`) patched with `--init_ckpt` for RETFound init
- init: RETFound-Large checkpoint
- data: transductive unlabeled rat OCT manifest (`Controls + HLS`, cohorts `1/2/3`, all ages)
- duration: `50` epochs (`0..49`)

Notes from the run:
- `7071 / 7073` images exported for SSL
- `2` unreadable images skipped (logged via `--skip-unreadable`)
- final MAE train loss (epoch 49): `0.1003`

Resulting MAE checkpoint (example path used downstream):
- `outputs/ssl_adapt/mae_transductive_c123_allages_fbmae_run50_live/mae_run/output_resume_from_e0/checkpoint-49.pth`

## Current Unlabeled Manifests (Example)

- Strict (no rat leakage): `outputs/ssl_manifests/.../ssl_strict_train_rats_unlabeled.csv`
- Transductive (all rats): `outputs/ssl_manifests/.../ssl_transductive_all_rats_unlabeled.csv`

Transductive is the recommended fast-progress path for domain adaptation experiments, but it must be reported as transductive.

## Recommended MAE Repo

Use the official Facebook MAE repo (`facebookresearch/mae`) as the pretraining backend.

Notes:
- Official `main_pretrain.py` is pretraining-only and usually does not expose an init-checkpoint flag.
- In local experiments, we patched MAE pretraining to add `--init_ckpt` so continued pretraining can start from RETFound weights.

This patch is external to this repo (in your local MAE clone), so keep that noted in experiments.

## Recommended 4090 Settings (ViT-L / 224)

Start conservative and stable:
- `model = vit_large_patch16`
- `input_size = 224`
- `mask_ratio = 0.75`
- `batch_size = 16`
- `accum_iter = 4`
- `epochs = 50` (quick signal), then `100+` if promising
- `blr = 5e-4` (continue-pretrain is often more stable than `1e-3`)
- `num_workers = 8`

## Example Transductive MAE Launch

```bash
python3 RETFoundLoRA/launch_mae_ssl_adapt.py \
  --manifest outputs/ssl_manifests/c123_allages_controls_hls_strict_vs_transductive/ssl_transductive_all_rats_unlabeled.csv \
  --out-dir outputs/ssl_adapt/mae_transductive_c123_allages_fbmae_run50 \
  --pretrain-script /home/alireza/Code/mae/main_pretrain.py \
  --python-bin /home/alireza/Code/mae/.venv_mae/bin/python \
  --launch-backend python-module \
  --auto-init-ckpt-arg \
  --force-rgb-export \
  --epochs 50 \
  --batch-size 16 \
  --accum-iter 4 \
  --num-workers 8 \
  --blr 5e-4 \
  --execute
```

## After MAE Adaptation (Supervised Comparison)

Keep the supervised pipeline fixed and only swap the backbone initialization:

1. Baseline: original RETFound initialization + `MIL + LoRA-4 + mild`
2. Adapted: MAE-adapted RETFound initialization + same supervised config

Compare on:
- Control MAE/RMSE and inter-eye `|OD-OS|`
- HLS MAE/RMSE
- cohort/day slices (especially harder cohorts / later days)

### Current Best MAE-Adapted Supervised Result (Transductive, `<=90` Protocol)

Using the MAE-adapted backbone checkpoint above with the supervised setting:
- `MIL + LoRA-4`
- `mil_attn_dim=256`, `mil_hidden_dim=512`
- `aug_level=mild`, `--no-photometric-aug`
- `--day-whitelist 0 7 14 28 90`
- `--control-eval-days 0 90`
- `--lr 1e-4`, `--epochs 40`

Observed metrics:
- Control (`day 0/90`): `MAE=35.04`, `RMSE=57.75`, `R²=0.587`
- HLS (`0/7/14/28/90`): `MAE=25.94`, `RMSE=42.70`, `R²=0.783`
- Control inter-eye mean `|OD-OS|`: `16.41`
- HLS inter-eye mean `|OD-OS|`: `13.05`

This outperformed the closest available non-SSL staged `<=90` comparator in both accuracy and inter-eye metrics, but should still be labeled **transductive SSL**.

## Reporting Guidance

When presenting results, label clearly:
- `No SSL baseline`
- `Strict SSL (unlabeled, train rats only)` or
- `Transductive SSL (unlabeled, all rats incl. eval)`

Do not mix these protocols in one headline number without the label.
