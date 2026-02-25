#!/usr/bin/env bash
set -euo pipefail

# Paper #1 (Control-only age prediction) wrapper for the current codebase.
# This is a pragmatic wrapper around the scripts already implemented in scripts/paper/.
# It skips completed steps when expected outputs already exist.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "Paper #1 Control-Only Pipeline (current codebase)"
echo "================================================="

FREE_GB="$(df -BG . | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
echo "[DISK] Free space: ${FREE_GB}G"
if [[ "${FREE_GB}" -lt 10 ]]; then
  echo "[DISK] Low free space detected (<10G). Running checkpoint cleanup dry-run..."
  python3 scripts/paper/cleanup_checkpoints.py --dry-run || true
fi

mkdir -p outputs/paper1

run_if_missing() {
  local marker="$1"; shift
  local label="$1"; shift
  if [[ -f "${marker}" ]]; then
    echo "[SKIP] ${label} (found ${marker})"
  else
    echo "[RUN ] ${label}"
    "$@"
  fi
}

# Core experiments (EXP-01..04 in current wrapper)
run_if_missing "outputs/core/exp01_ctrl_cv/summary.csv" \
  "EXP-01..04 core pipeline (CV + OOD agg + inter-eye + saliency)" \
  bash scripts/paper/run_all_core_experiments.sh all

# Backbone ablation (EXP-02 in paper1 runbook)
run_if_missing "outputs/ablation/backbone_comparison.csv" \
  "Backbone ablation (RETFound vs Xception; current codebase support)" \
  bash scripts/paper/run_backbone_ablation.sh

# LOCO cohort generalization (additional high-value experiment; patched support)
run_if_missing "outputs/generalization/exp07_loo_mae50_mil_lora4_d0090/loo_summary.csv" \
  "LOCO cohort generalization (held-out cohorts 1/2/3)" \
  bash -lc '
    set -euo pipefail
    ROOT=outputs/generalization/exp07_loo_mae50_mil_lora4_d0090
    mkdir -p "$ROOT"
    for heldout in 1 2 3; do
      if [ "$heldout" = "1" ]; then train_cohorts=(2 3); fi
      if [ "$heldout" = "2" ]; then train_cohorts=(1 3); fi
      if [ "$heldout" = "3" ]; then train_cohorts=(1 2); fi
      OUT="$ROOT/loo_cohort${heldout}"
      if [ -f "$OUT/metrics_summary.csv" ]; then
        echo "[SKIP] LOCO heldout cohort ${heldout} (metrics_summary.csv exists)"
        continue
      fi
      python3 RETFoundLoRA/run.py \
        --mil-attention --no-mil-freeze-backbone --lora-blocks 4 \
        --mil-attn-dim 256 --mil-hidden-dim 512 \
        --epochs 40 --early-stop-patience 1000 \
        --day-whitelist 0 90 --control-eval-days 0 90 \
        --cohorts 1 2 3 \
        --train-cohorts "${train_cohorts[@]}" --test-cohorts "$heldout" \
        --train-groups Controls --test-groups Controls "HLS (U)" \
        --aug-level mild --no-photometric-aug --no-bias-correction \
        --lr 1e-4 \
        --backbone-ckpt outputs/ssl_adapt/mae_transductive_c123_allages_fbmae_run50_live/mae_run/output_resume_from_e0/checkpoint-49.pth \
        --save-lora "$OUT/checkpoint.pt" \
        --pred-csv "$OUT/predictions.csv" \
        --metrics-csv "$OUT/metrics_summary.csv"
    done
    python3 scripts/aggregate_loo.py --input-dir "$ROOT" --output "$ROOT/loo_summary.csv" --emit-breakdowns
  '

# Paper tables / figures (already supported)
run_if_missing "outputs/paper_ready/tables/generated_tables_index.csv" \
  "Generate paper-ready tables" \
  python3 scripts/generate_paper_tables.py --all

run_if_missing "outputs/paper_ready/figures/generated_figures_index.csv" \
  "Generate paper-ready figures" \
  python3 scripts/generate_paper_figures.py --all

# Compact paper1 bundle/report
echo "[RUN ] Build compact Paper #1 bundle"
python3 scripts/paper/build_paper1_bundle.py

echo
echo "Done."
echo "Bundle: outputs/paper1/"
echo "Execution report: outputs/paper1/EXECUTION_REPORT.md"

