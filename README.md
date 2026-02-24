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
  - MAE CLI arg autodetection (`main_pretrain.py -h`)
  - launch script generation / execution

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
