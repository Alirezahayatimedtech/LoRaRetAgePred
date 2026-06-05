# LoRaRetAgePred

Brown Norway rat OCT age-prediction workspace built around `RETFoundLoRA`, matched CNN baselines, and manuscript-oriented reproducibility assets for the OSD-679 study.

## Main components

- `RETFoundLoRA/`: RETFound + LoRA age-regression pipeline, evaluation helpers, manuscript assets, and experimental runners.
- `scripts/paper/`: scripts used to build paper tables, figures, reviewer-mode manuscript copies, and the reproducibility bundle.
- `reproducibility/osd679_age_prediction_release/`: polished supplementary-data package for the age-prediction paper.
- `docs/rag_delta_biomarker_analysis.md`: current RAG/DeltaRAG biomarker-analysis summary across Cohorts 1-3 and all available days.
- `docs/rag_method_recommendations.md`: recommended statistical framing, interpretation limits, and spaceflight-use-case notes.
- `results/rag_delta_biomarker/`: compact RAG/DeltaRAG result tables.

## RAG / DeltaRAG biomarker analysis

The current RAG analysis uses the image-only RETFound-LoRA NatureOCT model trained on controls only, evaluated with 3-fold rat-level cross-validation across Cohorts 1-3 and all available days.

Key result:

- Control out-of-fold age MAE was about 31-34 days across folds.
- DeltaRAG did not strongly separate HLS from controls; day-90 AUROC was about 0.51 and day-14 was only a weak trend.
- The strongest biological signal was recovery-phase RAG versus temporal choroidal thickness in HLS rats: residualized Pearson r about -0.85, BH-adjusted q about 0.0036.
- Biomarker-fusion age modeling was stopped after weak partial results and is documented only as exploratory because it creates circularity for biomarker-correlation claims.

Start here:

- `docs/rag_delta_biomarker_analysis.md`
- `docs/rag_method_recommendations.md`
- `results/rag_delta_biomarker/README.md`

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
- RAG/DeltaRAG biomarker-analysis summaries

To fully reproduce the experiments, pair this repository with a local OSD-679 checkout obtained through NASA GeneLab / the Open Science Data Repository.

## Notes

- `chronological_age_days` preserves the raw metadata-derived age.
- `benchmark_age_days` reflects the age implied by the benchmark day label used in the paper protocol.
- The scratch/random ViT baseline is kept only as a supplementary negative-control architecture check, not as a main competing model.
- For biomarker-discovery claims, use image-only RAG as the primary endpoint; biomarker-input age models are exploratory only.
