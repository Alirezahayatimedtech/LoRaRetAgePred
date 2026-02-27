# Control Best/Worst Case Assets

- `best5_overlay_side_by_side_magma/`: side-by-side original+overlay saliency images for top-5 best control samples (Xception run).
- `worst5_original_images/`: original OCT images for top-5 worst control samples.

Filename convention:
- `best_rankXX_<rat>_<eye>_day<d>_mae<abs_error>_<orig_name>.png`
- `worst_rankXX_<rat>_<eye>_day<d>_mae<abs_error>_<orig_name>.BMP`

`mae` here is per-sample absolute error: `|age_pred - age_true|` for that selected control sample.
