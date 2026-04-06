# Control Best/Worst Case Assets

This folder contains curated qualitative-control examples from the primary Xception control benchmark.

Folders:
- `manuscript_best5/`: five manuscript-ready composite PNG panels for the top-5 best control predictions.
- `best5_overlay_side_by_side_magma/`: per-image original-versus-magma overlay panels for the best-control cases.
- `worst5_original_images/`: original OCT images for the top-5 worst control predictions.
- `restored_case_assets/`: full per-sample review folders for both best and worst cases.

Inside each `restored_case_assets/.../rankXX_*` folder:
- `panel.png`: composite sample panel.
- `sample_summary.csv`: sample-level metadata and error.
- `original_images/`: original OCT inputs used by the model.
- `magma_overlays/`: saliency overlays.
- `side_by_side/`: original-versus-overlay image pairs.
- `overlay_side_by_side_magma/`: manuscript-style overlay pair exports.

Filename convention:
- `best_rankXX_<rat>_<eye>_day<d>_mae<abs_error>_<orig_name>.png`
- `worst_rankXX_<rat>_<eye>_day<d>_mae<abs_error>_<orig_name>.BMP`

`mae` here is the per-sample absolute error: `|age_pred - age_true|` for the selected control sample.
