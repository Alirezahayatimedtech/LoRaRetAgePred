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
  - autodetects common MAE CLI flags from `main_pretrain.py -h`
  - can generate a launch script or execute directly

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

## Reporting Guidance

When presenting results, label clearly:
- `No SSL baseline`
- `Strict SSL (unlabeled, train rats only)` or
- `Transductive SSL (unlabeled, all rats incl. eval)`

Do not mix these protocols in one headline number without the label.
