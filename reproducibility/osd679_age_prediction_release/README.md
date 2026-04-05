# OSD-679 Age-Prediction Reproducibility Bundle

This folder contains a polished, GitHub-friendly supplementary data package for the Brown Norway rat retinal age-prediction experiments.

It is designed to support manuscript reproduction without redistributing the raw OSD-679 image payload.

## Contents

- `reproducibility/osd679_age_prediction_release/Supplementary_Data_1_Image_to_Age_Mapping.xlsx`: Excel workbook with the full image-to-age mapping, OCT-eligible subsets, and the primary day 0/90 manifests.
- `reproducibility/osd679_age_prediction_release/Supplementary_Data_2_Benchmark_Splits_and_Results.xlsx`: Excel workbook with cohort counts, benchmark results, split definitions, and supplementary tables.
- `reproducibility/osd679_age_prediction_release/Supplementary_Data_3_Qualitative_Examples.xlsx`: Excel workbook with best/worst qualitative example metadata and image-level review indices.
- `reproducibility/osd679_age_prediction_release/csv/osd679_c123_all_mapped_images_minimal.csv`: CSV companion for the cleaned Cohort 1-3 image-to-age mapping.
- `reproducibility/osd679_age_prediction_release/csv/osd679_paper1_controls_day0_day90_manifest.csv`: CSV companion for the primary control day 0/90 benchmark manifest.
- `reproducibility/osd679_age_prediction_release/csv/osd679_paper1_controls_hls_day0_day90_manifest.csv`: CSV companion for the broader Controls + HLS (U) day 0/90 evaluation universe.
- `reproducibility/osd679_age_prediction_release/csv/osd679_paper1_control_cv_fold_definitions.csv`: Rat-level 3-fold cross-validation definitions for the primary control benchmark.
- `reproducibility/osd679_age_prediction_release/csv/osd679_paper1_control_performance_by_cohort_day.csv`: Per-cohort, per-day control performance table used in the manuscript.
- `reproducibility/osd679_age_prediction_release/csv/osd679_paper1_backbone_ablation_mainpaper.csv`: Main-text backbone ablation table (RETFound + LoRA vs Xception + GAP).
- `reproducibility/osd679_age_prediction_release/csv/osd679_paper1_best_worst_control_examples.csv`: Best/worst qualitative sample manifest for the Xception control review set.

## Notes

- Raw OCT images are not redistributed here. `image_path_relative` values are pointers into a local OSD-679-style directory layout.
- The primary benchmark subset corresponds to Cohorts 1-3, Controls only, image types `BScanThumb` + `REGAVG`, and study days 0 and 90.
- The broader `Controls + HLS (U)` subset is included because it underlies the control-vs-stress evaluation universe.
- `chronological_age_days` preserves the raw metadata-derived age, whereas `benchmark_age_days` reflects the age implied by the benchmark day label used in the paper protocol.
- Rat-level cross-validation folds are provided for the primary control benchmark.
- The scratch/random ViT baseline is retained in the supplementary result tables as a negative-control architecture check only.

## Data access

OSD-679 data access should be requested via NASA GeneLab / the Open Science Data Repository. This repository only provides the derived mapping tables, split definitions, and result summaries used in the paper.

## Regeneration

This bundle is generated from the local metadata/results cache with:

`python3 scripts/paper/build_reproducibility_bundle.py`
