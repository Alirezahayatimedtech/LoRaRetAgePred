# LoRaRetAgePred

RETFound-based OCT age regression pipeline with:
- LoRA fine-tuning for ViT RETFound backbones
- attention-MIL for many images/views per eye (`bag = rat_id, eye, day`)
- deterministic robust intensity normalization (train + eval)
- control/HLS evaluation and inter-eye reliability analysis
- SSL tooling for MAE-style domain adaptation on unlabeled rat OCT

## Repo Contents

- `RETFoundLoRA/`: training/eval code, RETFound+LoRA model, MIL mode, diagnostics, SSL manifest/launcher tools
- `data_prep_age_lora.py`: shared metadata loading, transforms, image datasets, MIL bag datasets

## Current Best Supervised Setup (No SSL)

Best performing configuration so far in this project:
- `MIL + LoRA` with `lora_blocks=4`
- `--mil-attn-dim 256 --mil-hidden-dim 512`
- `--aug-level mild`
- `--no-photometric-aug`
- train on all ages, control eval restricted to day `0/90`
- `--no-bias-correction` for honest model selection

```bash
python3 RETFoundLoRA/run.py \
  --mil-attention \
  --no-mil-freeze-backbone \
  --lora-blocks 4 \
  --mil-attn-dim 256 \
  --mil-hidden-dim 512 \
  --epochs 20 \
  --all-ages \
  --control-eval-days 0 90 \
  --aug-level mild \
  --no-photometric-aug \
  --no-bias-correction \
  --post-control-inter-eye-analysis \
  --post-control-matched-view
```

## Inter-Eye Reliability Tools

Added utilities for control-first reliability analysis:
- `RETFoundLoRA/annotate_inter_eye_reliability.py`: annotate paired OD/OS CSVs with control-derived q95/q99 flags
- `RETFoundLoRA/control_matched_view_infer.py`: control-only matched-view MIL re-inference diagnostic
- `RETFoundLoRA/run.py` post-steps:
  - `--post-control-inter-eye-analysis`
  - `--post-control-matched-view`

## SSL / MAE Domain Adaptation (Next Major Plan)

We are moving toward **MAE-style continued pretraining on unlabeled rat OCT** starting from RETFound weights, then re-running the same supervised `MIL + LoRA` pipeline.

Why:
- reduce human -> rat OCT domain gap
- improve cohort generalization (especially harder cohorts / older ages)
- use all unlabeled OCT without age-label leakage

Plan details and exact commands are in:
- `docs/MAE_SSL_Transductive_Plan.md`

### Included SSL tooling

- `RETFoundLoRA/build_ssl_manifests.py`
  - builds strict vs transductive unlabeled manifests from the current split logic
- `RETFoundLoRA/launch_mae_ssl_adapt.py`
  - manifest -> MAE-compatible `ImageFolder`
  - optional RGB export for grayscale OCT
  - `--skip-unreadable` support for corrupt/unreadable source images (logs skipped files and continues)
  - MAE CLI arg autodetection (`main_pretrain.py -h`)
  - launch script generation / execution

## Current Best MAE-Adapted Setup (Transductive SSL, <=90-Day Supervised Protocol)

This is the current best **transductive SSL** result in this repo (unlabeled MAE adaptation on all rats, then supervised `MIL + LoRA` age regression).

Protocol:
- MAE continued pretraining on unlabeled rat OCT (Controls + HLS, cohorts `1/2/3`, all ages), reported as **transductive SSL**
- supervised training on days `0/7/14/28/90` only
- control evaluation restricted to day `0/90`

Supervised command (MAE-adapted backbone init):

```bash
python3 RETFoundLoRA/run.py \
  --mil-attention \
  --no-mil-freeze-backbone \
  --lora-blocks 4 \
  --mil-attn-dim 256 \
  --mil-hidden-dim 512 \
  --epochs 40 \
  --lr 1e-4 \
  --day-whitelist 0 7 14 28 90 \
  --control-eval-days 0 90 \
  --aug-level mild \
  --no-photometric-aug \
  --no-bias-correction \
  --backbone-ckpt outputs/ssl_adapt/mae_transductive_c123_allages_fbmae_run50_live/mae_run/output_resume_from_e0/checkpoint-49.pth \
  --post-control-inter-eye-analysis
```

Observed results (this protocol):
- Control (`day 0/90`): `MAE=35.04`, `RMSE=57.75`, `R²=0.587`
- HLS (`0/7/14/28/90`): `MAE=25.94`, `RMSE=42.70`, `R²=0.783`
- Control inter-eye mean `|OD-OS|`: `16.41`
- HLS inter-eye mean `|OD-OS|`: `13.05`

Caveat:
- This result is **transductive SSL** and should be reported separately from strict train-rats-only SSL.
- It beat the closest available non-SSL `<=90` staged comparator in both accuracy and inter-eye metrics, but a strict from-scratch non-SSL `<=90` ablation is still recommended.

## Current Best MAE-Adapted Setup (Control-Priority, Day 0/90 Only Supervised Protocol)

Use this configuration when the primary objective is **Controls day 0/90** accuracy and inter-eye consistency (not broader HLS generalization across more days).

Protocol:
- MAE continued pretraining on unlabeled rat OCT (Controls + HLS, cohorts `1/2/3`, all ages), reported as **transductive SSL**
- supervised training on days `0/90` only
- control and HLS evaluation both restricted to day `0/90` (via `--day-whitelist 0 90`)

Supervised command (MAE-adapted backbone init):

```bash
python3 RETFoundLoRA/run.py \
  --mil-attention \
  --no-mil-freeze-backbone \
  --lora-blocks 4 \
  --mil-attn-dim 256 \
  --mil-hidden-dim 512 \
  --epochs 40 \
  --lr 1e-4 \
  --day-whitelist 0 90 \
  --control-eval-days 0 90 \
  --cohorts 1 2 3 \
  --aug-level mild \
  --no-photometric-aug \
  --no-bias-correction \
  --backbone-ckpt outputs/ssl_adapt/mae_transductive_c123_allages_fbmae_run50_live/mae_run/output_resume_from_e0/checkpoint-49.pth \
  --post-control-inter-eye-analysis
```

Observed results (control-priority protocol):
- Control (`day 0/90`): `MAE=34.90`, `RMSE=58.08`, `R²=0.583`
- HLS (`day 0/90`): `MAE=31.78`, `RMSE=50.16`, `R²=0.709`
- Control inter-eye mean `|OD-OS|`: `13.98`
- HLS inter-eye mean `|OD-OS|`: `13.19`

Checkpoint:
- `outputs/checkpoints/retfound_mil_e40_lora4_attn256_h512_mild_mae50transductive_lr1e4_d0090_c123.pt`

Tradeoff vs the `<=90` supervised protocol:
- Better control metrics and control inter-eye consistency
- Worse HLS accuracy (and slightly worse HLS inter-eye) than training on `0/7/14/28/90`

## Bias Correction Warning

For fair model comparison and deployable inference behavior, use:

```bash
--no-bias-correction
```

The current linear bias-correction path is retrospective and can use true age during correction application.

## Notes

- `--batch-size` in MIL mode means **bags per batch**, not images per batch.
- `--control-eval-days 0 90` only restricts the control evaluation path (not HLS days).
- `--no-photometric-aug` disables train-time photometric augmentation but keeps fixed robust intensity normalization.
