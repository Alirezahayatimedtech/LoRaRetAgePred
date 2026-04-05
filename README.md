# LoRaRetAgePred

Brown Norway rat OCT age-prediction workspace built around `RETFoundLoRA`, matched CNN baselines, and manuscript-oriented reproducibility assets for the OSD-679 study.

## Main components

- `RETFoundLoRA/`: RETFound + LoRA age-regression pipeline, evaluation helpers, manuscript assets, and experimental runners.
- `scripts/paper/`: scripts used to build paper tables, figures, reviewer-mode manuscript copies, and the reproducibility bundle.
- `reproducibility/osd679_age_prediction_release/`: polished supplementary-data package for the age-prediction paper.

## OSD-679 reproducibility bundle

The publishable bundle is located at:

- `reproducibility/osd679_age_prediction_release/`

It includes:

- `Supplementary_Data_1_Image_to_Age_Mapping.xlsx`
  - cleaned image-to-age mapping for Cohorts 1-3
  - OCT-eligible subset manifests
  - primary paper subset manifests for `Controls` day `0/90` and `Controls + HLS (U)` day `0/90`
- `Supplementary_Data_2_Benchmark_Splits_and_Results.xlsx`
  - cohort/sample counts
  - control CV results
  - cohort/day performance tables
  - backbone ablation summaries
  - split definitions and supplementary tables
- `Supplementary_Data_3_Qualitative_Examples.xlsx`
  - best/worst control example manifests
  - selected image-level review metadata used for qualitative figure assembly
- CSV companions for the most useful release tables
- `bundle_manifest.json`
- a local regeneration script: `scripts/paper/build_reproducibility_bundle.py`

## Rebuild the bundle

```bash
python3 scripts/paper/build_reproducibility_bundle.py
```

## Data access

The raw OSD-679 image payload is not redistributed in this repository.

This repo provides:

- relative image-path manifests
- rat-level split definitions
- benchmark-facing tables and summaries
- qualitative sample indices used in the manuscript

To fully reproduce the experiments, pair this repository with a local OSD-679 checkout obtained through NASA GeneLab / the Open Science Data Repository.

## Notes

- `chronological_age_days` preserves the raw metadata-derived age.
- `benchmark_age_days` reflects the age implied by the benchmark day label used in the paper protocol.
- The scratch/random ViT baseline is kept only as a supplementary negative-control architecture check, not as a main competing model.
