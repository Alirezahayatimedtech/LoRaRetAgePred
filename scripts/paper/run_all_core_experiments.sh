#!/usr/bin/env bash
set -euo pipefail

# Core experiment driver for OSD-679 paper protocol (current codebase version).
# Usage:
#   bash scripts/paper/run_all_core_experiments.sh exp01_ctrl_cv
#   bash scripts/paper/run_all_core_experiments.sh exp02_hls_ood
#   bash scripts/paper/run_all_core_experiments.sh exp03_inter_eye
#   bash scripts/paper/run_all_core_experiments.sh exp04_saliency

EXP_ID="${1:-all}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PY="${RUN_PY:-RETFoundLoRA/run.py}"
OUT_ROOT="${OUT_ROOT:-outputs}"
CSV_PATH="${CSV_PATH:-}"
BACKBONE_CKPT="${BACKBONE_CKPT:-outputs/ssl_adapt/mae_transductive_c123_allages_fbmae_run50_live/mae_run/output_resume_from_e0/checkpoint-49.pth}"

COMMON_FLAGS=(
  --cohorts 1 2 3
  --aug-level mild
  --no-photometric-aug
  --no-bias-correction
  --post-control-inter-eye-analysis
  --post-control-matched-view
)

if [[ -n "${CSV_PATH}" ]]; then
  COMMON_FLAGS+=(--csv "${CSV_PATH}")
fi

run_exp01_ctrl_cv() {
  local out_dir="${OUT_ROOT}/core/exp01_ctrl_cv/retfound_mae50_mil_lora4_d0090"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" "${RUN_PY}" \
    --model-type retfound \
    --backbone-ckpt "${BACKBONE_CKPT}" \
    --mil-attention --no-mil-freeze-backbone \
    --lora-blocks 4 --mil-attn-dim 256 --mil-hidden-dim 512 \
    --day-whitelist 0 90 \
    --control-eval-days 0 90 \
    --train-groups Controls \
    --test-groups "HLS (U)" \
    --kfolds 3 --run-all-folds --fold-seed 42 \
    --epochs 40 --lr 1e-4 --early-stop-patience 10 \
    --save-lora "${out_dir}/checkpoint.pt" \
    --pred-csv "${out_dir}/predictions.csv" \
    --metrics-csv "${out_dir}/metrics_summary.csv" \
    "${COMMON_FLAGS[@]}"

  "${PYTHON_BIN}" scripts/aggregate_cv.py \
    --input-dir "${out_dir}" \
    --metrics mae rmse r2 pearson_r \
    --emit-breakdowns \
    --output "${OUT_ROOT}/core/exp01_ctrl_cv/summary.csv"
}

run_exp02_hls_ood() {
  local in_dir="${OUT_ROOT}/core/exp01_ctrl_cv"
  local out_dir="${OUT_ROOT}/core/exp02_hls_ood"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" scripts/aggregate_cv.py \
    --input-dir "${in_dir}" \
    --metrics mae rmse r2 pearson_r \
    --emit-breakdowns \
    --output "${out_dir}/summary.csv"

  # Optional ΔRAG preparation if control and HLS prediction CSVs are available in a selected run.
  echo "[EXP-02] HLS OOD aggregation complete. Use RETFoundLoRA/plot_delta_rag.py on a chosen run dir for ΔRAG CSV/plot."
}

run_exp03_inter_eye() {
  local out_dir="${OUT_ROOT}/core/exp03_inter_eye"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" scripts/compute_inter_eye_mad.py \
    --predictions-dir "${OUT_ROOT}/core/exp01_ctrl_cv" "${OUT_ROOT}/core/exp02_hls_ood" \
    --group-by group,cohort,day \
    --include-overall \
    --output "${out_dir}/summary.csv"
}

run_exp04_saliency() {
  # Example / representative export; customize cohort/day/group and lora checkpoint path as needed.
  local out_dir="${OUT_ROOT}/core/exp04_saliency/day90_examples"
  mkdir -p "${out_dir}"
  echo "[EXP-04] Running representative saliency export subset (customize lora-ckpt path if needed)."
  "${PYTHON_BIN}" scripts/save_saliency_subset.py \
    ${CSV_PATH:+--csv "${CSV_PATH}"} \
    --cohorts 1 2 3 \
    --days 90 \
    --groups Controls "HLS (U)" \
    --lora-ckpt "${OUT_ROOT}/checkpoints/retfound_mil_e40_lora4_attn256_h512_mild_mae50transductive_lr1e4_d0090_c123.pt" \
    --backbone-ckpt "${BACKBONE_CKPT}" \
    --out-dir "${out_dir}"
}

case "${EXP_ID}" in
  exp01_ctrl_cv) run_exp01_ctrl_cv ;;
  exp02_hls_ood) run_exp02_hls_ood ;;
  exp03_inter_eye) run_exp03_inter_eye ;;
  exp04_saliency) run_exp04_saliency ;;
  all)
    run_exp01_ctrl_cv
    run_exp02_hls_ood
    run_exp03_inter_eye
    run_exp04_saliency
    ;;
  *)
    echo "Unknown EXP_ID: ${EXP_ID}" >&2
    echo "Expected: exp01_ctrl_cv | exp02_hls_ood | exp03_inter_eye | exp04_saliency | all" >&2
    exit 2
    ;;
esac

