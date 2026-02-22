# LoRaRetAgePred

RETFound-based age regression pipeline with:
- LoRA fine-tuning options
- deterministic robust intensity normalization
- attention-MIL baseline (bag = `rat_id, eye, day`)
- control/HLS evaluation + inter-eye analysis workflow

## Repo Contents

- `RETFoundLoRA/`: training, evaluation, RETFound+LoRA model, MIL mode, trainer
- `data_prep_age_lora.py`: shared metadata loading, transforms, datasets, bag dataset

## Recommended Simple Baseline (MIL, Frozen RETFound)

This is the simplest workable mode for many images per case without fusion.

- Uses attention-MIL over all images in each `(rat_id, eye, day)` bag
- Disables fusion modes automatically
- Freezes RETFound backbone in MIL baseline (`lora_blocks=0` forced)

```bash
python3 RETFoundLoRA/run.py \
  --mil-attention \
  --epochs 30 \
  --early-stop-patience 1000 \
  --all-ages \
  --control-eval-days 0 90 \
  --no-photometric-aug \
  --no-bias-correction \
  --batch-size 4 \
  --save-lora outputs/checkpoints/retfound_mil_e30_frozen_retfound.pt \
  --pred-csv outputs/predictions/retfound_mil_e30_frozen_retfound/predictions.csv \
  --metrics-csv outputs/predictions/retfound_mil_e30_frozen_retfound/metrics_summary.csv
```

## Notes

- `--batch-size` in MIL mode means **bags per batch**, not images per batch.
- `--control-eval-days 0 90` restricts only control evaluation outputs/metrics, not HLS test days.
- `--no-photometric-aug` disables train-time photometric augmentation but keeps fixed robust intensity normalization.

## Bias Correction Warning

For fair model comparison and deployable inference behavior, use:

```bash
--no-bias-correction
```

The current linear bias-correction path is a retrospective calibration method and can use true age during correction application.

## Quick Smoke Test (MIL)

```bash
python3 RETFoundLoRA/run.py \
  --mil-attention \
  --epochs 1 \
  --batch-size 4 \
  --all-ages \
  --control-eval-days 0 90 \
  --no-photometric-aug \
  --save-lora outputs/checkpoints/mil_smoke_1epoch.pt \
  --pred-csv outputs/predictions/mil_smoke_1epoch/predictions.csv \
  --metrics-csv outputs/predictions/mil_smoke_1epoch/metrics_summary.csv
```

