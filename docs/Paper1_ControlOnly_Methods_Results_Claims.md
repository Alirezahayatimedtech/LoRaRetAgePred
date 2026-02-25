# Paper #1 (Control-Only Age Prediction): Methods, Results, and Claims

This document records the **executed** protocol and results for the current codebase (not the hypothetical CLI shown in the draft runbook).

## Method (Executed)

### Dataset / Scope
- OSD-679 rat OCT
- Cohorts: `1, 2, 3` (Cohort 4 excluded)
- Primary control-priority protocol: **Day 0 and Day 90**
- Rat-level splits (no rat overlap across train/val/test within runs)

### Core pipeline (current codebase)
- Core batch runner: `scripts/paper/run_all_core_experiments.sh`
  - EXP-01-like: 3-fold control-only CV (RETFound MAE-transductive + MIL + LoRA4)
  - EXP-02-like: HLS OOD aggregation from folds
  - EXP-03-like: inter-eye reliability aggregation
  - EXP-04-like: saliency export (gradient-saliency fallback for CLS-only checkpoints)
- Backbone ablation runner: `scripts/paper/run_backbone_ablation.sh`
  - Completed: `RETFound`, `Xception`, `vit_random`
- Tables/Figures:
  - `scripts/generate_paper_tables.py --all`
  - `scripts/generate_paper_figures.py --all`
- Compact packaging:
  - `scripts/paper/build_paper1_bundle.py`

### Added experiment (patched)
- **True LOCO cohort support** was added to:
  - `RETFoundLoRA/run.py`
  - `RETFoundLoRA/preprocess_age_lora.py`
- New CLI:
  - `--train-cohorts ...`
  - `--test-cohorts ...`
- LOCO runs use held-out Controls + HLS in `test_groups`, with held-out Controls routed to `ctrl_test` reporting.

## What Was Completed vs. Skipped

### Completed / Reused
- EXP-01 (control-only CV): completed
- EXP-02 (backbone ablation, supported subset): completed (`RETFound`, `Xception`)
- EXP-04 (inter-eye reliability): completed
- EXP-05 (saliency export): completed
- EXP-06 optional distillation summary: summarized from completed distillation runs
- Additional LOCO cohort generalization: completed (all held-out cohorts 1/2/3)

### Remaining Gaps (Current Codebase)
- The runbook's exact hypothetical YAML/CLI interface was not used literally.
- EXP-06 optional distillation `α=0.05` rat-teacher row remains unrun (summary marks it unavailable).

## Key Results (Executed Runs)

### 1) Control-only 3-fold CV aggregate (RETFound MAE-transductive + MIL + LoRA4, day 0/90)
Source: `outputs/core/exp01_ctrl_cv/summary.csv`

| Split | MAE mean ± SD | RMSE mean ± SD | R² mean ± SD | Pearson r mean ± SD |
|---|---:|---:|---:|---:|
| Control | `26.20 ± 5.03` | `43.24 ± 5.38` | `0.744 ± 0.049` | `0.868 ± 0.024` |
| HLS | `30.34 ± 1.12` | `46.38 ± 1.36` | `0.751 ± 0.015` | `0.882 ± 0.003` |

### 2) Backbone ablation (control-priority day 0/90 protocol)
Source: `outputs/paper1/ablation/backbone_comparison.csv`

| Model | Control MAE | Control R² | HLS MAE | HLS R² | Control inter-eye mean | HLS inter-eye mean |
|---|---:|---:|---:|---:|---:|---:|
| RETFound (MAE-transductive) + MIL + LoRA4 | `34.75` | `0.581` | `30.30` | `0.729` | `15.71` | `15.68` |
| Xception | **`23.44`** | **`0.791`** | **`26.97`** | **`0.732`** | `19.13` | `16.13` |
| Random ViT (frozen head-only baseline) | `70.46` | `0.090` | `71.21` | `0.128` | `5.47` | `5.97` |

### 3) RETFound adaptation ablation (LoRA vs full fine-tune vs frozen head-only)
Source: `outputs/paper1/lora_ablation/summary.csv`

Control/HLS summary (day 0/90):
- `full_ft`: Control MAE `26.87`, HLS MAE `23.31` (best of the three)
- `lora`: Control MAE `34.90`, HLS MAE `31.78`
- `frozen_head`: Control MAE `55.26`, HLS MAE `55.85`

Interpretation:
- In this narrow control-priority protocol, full RETFound fine-tuning outperformed LoRA and frozen-head-only.
- Frozen head-only underfits severely.

### 4) Inter-eye reliability aggregate (core run aggregation)
Source: `outputs/core/exp03_inter_eye/summary.csv`

Selected rows (overall):
- Control overall mean MAD (`age_pred_inter_eye_abs`): `14.22` days
- Stress overall mean MAD: `13.32` days

Per-cohort/day breakdown is preserved in:
- `outputs/paper1/exp04_inter_eye/summary.csv`

### 5) Distillation summary (Xception student, control-only day 0/90)
Source: `outputs/paper1/distillation_summary.csv`

| Model | MAE | RMSE | R² | Inter-eye mean | Inter-eye Q95 | Day90 MAE |
|---|---:|---:|---:|---:|---:|---:|
| Xception baseline | **`23.76`** | `39.16` | `0.810` | **`13.20`** | **`43.47`** | `33.65` |
| + Human RETFound teacher (feature distill, α=0.3) | `26.06` | `39.37` | `0.808` | `20.30` | `59.21` | `32.68` |
| + Rat-adapted RETFound teacher (feature distill, α=0.1) | `24.76` | **`37.37`** | **`0.827`** | `17.65` | `58.72` | **`32.32`** |

Interpretation:
- Human-teacher distillation degrades control-priority performance.
- Rat-adapted teacher improves RMSE/R² and Day90 MAE vs human-teacher distill, but still does **not** beat plain Xception on overall control MAE and inter-eye reliability.

### 6) LOCO cohort generalization (added experiment)
Source: `outputs/generalization/exp07_loo_mae50_mil_lora4_d0090/loo_summary_with_inter_eye.csv`

| Held-out Cohort | Split | MAE | RMSE | R² | Inter-eye mean | Inter-eye Q95 |
|---|---|---:|---:|---:|---:|---:|
| 1 | Control | `50.97` | `60.96` | `-0.93` | `23.22` | `85.52` |
| 1 | HLS | `56.23` | `65.36` | `-1.19` | `18.43` | `67.89` |
| 2 | Control | `40.85` | `55.27` | `-0.58` | `2.84` | `8.13` |
| 2 | HLS | `40.26` | `54.95` | `-0.56` | `2.55` | `6.80` |
| 3 | Control | `137.39` | `144.90` | `-10.66` | `3.18` | `6.11` |
| 3 | HLS | `144.42` | `151.36` | `-10.55` | `3.13` | `6.04` |

Interpretation:
- Held-out cohort 3 collapses (absolute age extrapolation failure; train cohorts 1/2 only see younger age regime).
- Low inter-eye error in held-out cohort 3 is **not** success; it reflects consistent bilateral underprediction.

## Claims (Supported by Current Results)

1. **Xception is a strong control-only baseline on day 0/90** and outperforms the current RETFound+MIL setup on the narrow control-priority slice.
2. **RETFound pipelines remain experimentally valuable** for CV, OOD/HLS summaries, saliency, MIL/LoRA/SSL ablations, and structured generalization studies.
3. **Under the current day 0/90 control-priority RETFound protocol, full fine-tuning beats LoRA PEFT**, while frozen-head-only clearly underfits.
4. **Feature distillation must be teacher/domain-aware**:
   - human RETFound teacher features hurt Xception (α=0.3),
   - rat-adapted RETFound teacher features partially recover performance (better RMSE/R²), but do not surpass Xception on control MAE/inter-eye.
5. **Cross-cohort generalization is constrained by age-regime coverage**:
   - LOCO cohort 3 failure indicates severe age extrapolation/domain-shift when the older regime is absent from training.

## Locations of Paper-Ready Assets

- Tables: `outputs/paper1/tables/`
- Figures: `outputs/paper1/figures/`
- Compact execution bundle: `outputs/paper1/`
- Local execution report: `outputs/paper1/EXECUTION_REPORT.md`
