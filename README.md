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

## Paper #1 Status (Control-Only Age Prediction)

Completed in the current codebase:
- control-only 3-fold CV pipeline (RETFound MAE-transductive + MIL + LoRA4)
- backbone ablation (`RETFound`, `Xception`, `Random ViT`)
- RETFound adaptation ablation (`LoRA` vs `full fine-tune` vs `frozen head-only`)
- inter-eye reliability aggregation
- saliency export (CLS-only checkpoints use gradient-saliency fallback)
- LOCO cohort generalization runs (held-out cohort `1`, `2`, `3`)
- paper-ready table/figure generation scripts and compact execution bundle tooling

Local reporting artifacts (not pushed to GitHub):
- `outputs/paper1/EXECUTION_REPORT.md`
- `docs/Paper1_ControlOnly_Methods_Results_Claims.md`
- `outputs/paper1/ablation/backbone_comparison.csv`
- `outputs/paper1/lora_ablation/summary.csv`

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

## SSL / MAE Domain Adaptation (Implemented)

The repo now includes and has been used for **MAE-style continued pretraining on unlabeled rat OCT** starting from RETFound weights, followed by supervised `MIL + LoRA` runs.

Why:
- reduce human -> rat OCT domain gap
- improve cohort generalization (especially harder cohorts / older ages)
- use all unlabeled OCT without age-label leakage

Protocol details and exact commands are in:
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

## Xception Baseline (Matched Control-Only Protocol, Day 0/90)

To test whether RETFound provides a real benefit on the narrow control-priority slice, a matched Xception baseline was run with the same split/protocol:

- cohorts `1/2/3`
- train/eval on day `0/90`
- rat-level splits
- `aug_level=mild`, `--no-photometric-aug`
- `--no-bias-correction`

Important note:
- `lr=1e-4` underfit badly for Xception in this setup.
- A fairer Xception baseline used `--lr 1e-3`.

Observed results (Xception, `lr=1e-3`):
- Control (`day 0/90`): `MAE=23.76`, `RMSE=39.16`, `R²=0.810`
- HLS (`day 0/90`): `MAE=25.08`, `RMSE=43.57`, `R²=0.781`
- Control inter-eye mean `|OD-OS|`: `13.20`
- Control day 90 inter-eye mean `|OD-OS|`: `15.41`

Artifacts:
- Checkpoint: `outputs/checkpoints/xception_e40_lr1e3_d0090_c123.pt`
- Metrics: `outputs/predictions/xception_e40_lr1e3_d0090_c123/metrics_summary.csv`

Interpretation:
- On this **narrow day 0/90 control-priority protocol**, the simple Xception baseline outperformed the current RETFound+MIL setup.
- This does **not** prove RETFound is unhelpful in general; it means the RETFound advantage was not observed on this slice/protocol.

## Backbone Ablation (Completed, Control-Priority Day 0/90)

Fair comparison (same split/protocol, day `0/90`, control-priority):

| Backbone | Control MAE | Control R² | HLS MAE | HLS R² |
|---|---:|---:|---:|---:|
| RETFound (MAE-transductive) + MIL + LoRA4 | 34.75 | 0.581 | 30.30 | 0.729 |
| Xception | **23.44** | **0.791** | **26.97** | **0.732** |
| Random ViT (frozen head-only baseline) | 70.46 | 0.090 | 71.21 | 0.128 |

Summary:
- Xception is the stronger backbone baseline on this narrow control-only day `0/90` slice.
- Random ViT performs poorly, which supports the value of pretrained features in general.

## RETFound Adaptation Ablation (Completed)

RETFound-based adaptation comparison on the same control-priority day `0/90` protocol:

| Method | Control MAE | HLS MAE | Control R² | HLS R² |
|---|---:|---:|---:|---:|
| Full fine-tune (RETFound backbone trainable) | **26.87** | **23.31** | **0.662** | **0.750** |
| LoRA (current control-priority default) | 34.90 | 31.78 | 0.582 | 0.709 |
| Frozen backbone + head only | 55.26 | 55.85 | 0.264 | 0.295 |

Summary:
- In this specific day `0/90` control-priority setting, **full fine-tuning beats LoRA**.
- Frozen head-only underfits severely.

## Xception + RETFound Feature Distillation (Control-Only, Day 0/90)

A feature-level distillation experiment was added for the Xception baseline:

- Student: Xception
- Teacher: frozen RETFound checkpoint (feature-only teacher)
- Distillation loss: MSE on L2-normalized features
- `--skip-stress-eval` used (control-only experiment)

Run settings (first test):
- `--distill-alpha 0.3`
- `--distill-feature-only`
- `--distill-teacher-ckpt RETFound_MAE_Model/RETFound_mae_natureOCT.pth`

Observed result (control-only):
- Control (`day 0/90`): `MAE=26.06`, `RMSE=39.37`, `R²=0.808`
- Control inter-eye mean `|OD-OS|`: `20.30`
- Control day 90 inter-eye mean `|OD-OS|`: `25.57`

Compared to plain Xception (`lr=1e-3`, same protocol):
- Control MAE worsened (`23.76 -> 26.06`)
- Control inter-eye worsened (`13.20 -> 20.30`)

Conclusion (for `alpha=0.3`):
- RETFound feature distillation did **not** improve Xception on this control-only day `0/90` protocol.
- If distillation is revisited, reduce `alpha` (e.g. `0.05–0.1`) and keep it feature-only.

Artifacts:
- Checkpoint: `outputs/checkpoints/xception_e40_lr1e3_d0090_c123_distill_retfoundfeat_a03_b8.pt`
- Metrics: `outputs/predictions/xception_e40_lr1e3_d0090_c123_distill_retfoundfeat_a03_b8/metrics_summary.csv`
- Comparison table: `outputs/predictions/compare_xception_vs_xception_distill_retfoundfeat_day0090_controlonly.csv`

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

## Reproducing Paper #1 Results

### Prerequisites

- Python `3.9+`
- PyTorch `2.x` with CUDA (tested on CUDA `11.x/12.x` setups)
- Install project dependencies (environment varies by local setup):

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present in your local clone, use the environment described in `environment.yml` (local project root) and install the package set used for the runs in this repo (`torch`, `torchvision`, `timm`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`, etc.).

### Run Core Paper #1 Pipeline

```bash
./run_paper1_all.sh
```

This executes the current paper pipeline wrapper (skip-if-done behavior), including:
- control-only CV
- backbone ablation
- LOCO cohort generalization
- paper table/figure generation
- compact paper bundle assembly

### Key Outputs

- `outputs/paper1/exp01_retfound_lora/summary.csv` — main control-only CV summary
- `outputs/paper1/ablation/backbone_comparison.csv` — backbone ablation summary
- `outputs/paper1/lora_ablation/summary.csv` — RETFound adaptation ablation (LoRA/full FT/frozen head)
- `outputs/paper1/exp04_inter_eye/summary.csv` — inter-eye reliability summary
- `outputs/paper1/tables/` — manuscript-ready tables (CSV outputs from generator)
- `outputs/paper1/figures/` — generated figure assets
- `outputs/paper1/EXECUTION_REPORT.md` — executed methods/results/claims audit

### Data Access (OSD-679)

OSD-679 data is not bundled in this repository. Request access via NASA GeneLab:

- https://genelab-data.ndc.nasa.gov/genelab/accession/OSD-679

### Notes on Reproducibility

- All supervised experiments use rat-level splits (no rat overlap across train/val/test within runs).
- MAE-based SSL adaptation results are explicitly labeled **transductive** when unlabeled images from evaluation rats are used.
- For fair model comparisons, use `--no-bias-correction` unless you are reporting retrospective calibrated metrics separately.
