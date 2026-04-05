# slit-to-oct

A collection of experiments predicting OCT-derived anterior segment metrics from slit-lamp images. Includes:
- Data prep scripts to map slit-lamp images to CASIA OCT biometrics and cluster/label views.
- Multi-view regression models (early/late fusion, MIL) for targets like CCT, AOD/TISA/TIA, ACD.
- Utilities for anatomical normalization (nasal/temporal), evaluation, and visualization (attention maps, scatter plots).
- Slit-lamp labeling + active-learning workflow documented in `slit-project/labeling_readme.md`.

This repo currently hosts the submodule `RETFound_MAE` for related MAE work; main code and data are kept locally/offline for now.

## RETFound Age-Prediction Reproducibility Bundle

This workspace also contains the Brown Norway rat retinal age-prediction pipeline under `RETFoundLoRA/` and a polished supplementary-data package for the OSD-679 manuscript under:

- `reproducibility/osd679_age_prediction_release/`

That bundle includes:

- image-to-age mapping tables derived from `metadata/image_age_mapping.csv`
- the primary paper subset manifests for `Controls` day `0/90` and `Controls + HLS (U)` day `0/90`
- rat-level 3-fold split definitions
- benchmark / supplementary result tables
- qualitative best/worst example manifests used for manuscript figures

Key files:

- `reproducibility/osd679_age_prediction_release/Supplementary_Data_1_Image_to_Age_Mapping.xlsx`
- `reproducibility/osd679_age_prediction_release/Supplementary_Data_2_Benchmark_Splits_and_Results.xlsx`
- `reproducibility/osd679_age_prediction_release/Supplementary_Data_3_Qualitative_Examples.xlsx`
- `reproducibility/osd679_age_prediction_release/README.md`

The raw OSD-679 image payload is not redistributed here. The reproducibility bundle provides relative image paths, split definitions, and manuscript-facing tables so the analysis can be reconstructed against a local OSD-679 checkout.

To regenerate the bundle locally, run:

```bash
python3 scripts/paper/build_reproducibility_bundle.py
```
