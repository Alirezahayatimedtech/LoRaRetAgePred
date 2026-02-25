#!/usr/bin/env bash
set -euo pipefail

# Backbone ablation runner (current codebase support)
# Supported today: RETFound + Xception
# Not supported yet in run.py: random ViT backbone (draft EXP-05 placeholder)

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PY="${RUN_PY:-RETFoundLoRA/run.py}"
OUT_ROOT="${OUT_ROOT:-outputs}"
CSV_PATH="${CSV_PATH:-}"
BACKBONE_CKPT="${BACKBONE_CKPT:-outputs/ssl_adapt/mae_transductive_c123_allages_fbmae_run50_live/mae_run/output_resume_from_e0/checkpoint-49.pth}"

mkdir -p "${OUT_ROOT}/ablation"

COMMON=(
  --cohorts 1 2 3
  --day-whitelist 0 90
  --control-eval-days 0 90
  --train-groups Controls
  --test-groups "HLS (U)"
  --aug-level mild
  --no-photometric-aug
  --no-bias-correction
  --post-control-inter-eye-analysis
)
if [[ -n "${CSV_PATH}" ]]; then
  COMMON+=(--csv "${CSV_PATH}")
fi

echo "=== RETFound (control-priority, MAE-transductive) ==="
"${PYTHON_BIN}" "${RUN_PY}" \
  --model-type retfound \
  --backbone-ckpt "${BACKBONE_CKPT}" \
  --mil-attention --no-mil-freeze-backbone \
  --lora-blocks 4 --mil-attn-dim 256 --mil-hidden-dim 512 \
  --lr 1e-4 --epochs 40 \
  --save-lora "${OUT_ROOT}/ablation/retfound_mil_lora4_d0090_c123.pt" \
  --pred-csv "${OUT_ROOT}/ablation/retfound_mil_lora4_d0090_c123/predictions.csv" \
  --metrics-csv "${OUT_ROOT}/ablation/retfound_mil_lora4_d0090_c123/metrics_summary.csv" \
  "${COMMON[@]}"

echo "=== Xception baseline ==="
"${PYTHON_BIN}" "${RUN_PY}" \
  --model-type xception \
  --lr 1e-3 --epochs 40 \
  --save-lora "${OUT_ROOT}/ablation/xception_d0090_c123.pt" \
  --pred-csv "${OUT_ROOT}/ablation/xception_d0090_c123/predictions.csv" \
  --metrics-csv "${OUT_ROOT}/ablation/xception_d0090_c123/metrics_summary.csv" \
  "${COMMON[@]}"

echo "=== Compare supported backbones ==="
"${PYTHON_BIN}" scripts/compare_backbones.py \
  --inputs \
    "${OUT_ROOT}/ablation/retfound_mil_lora4_d0090_c123" \
    "${OUT_ROOT}/ablation/xception_d0090_c123" \
  --labels RETFound_MIL_LoRA4 Xception \
  --output "${OUT_ROOT}/ablation/backbone_comparison.csv"

echo "[NOTE] Random ViT baseline is not implemented in current run.py (model-type supports: retfound, xception)."

