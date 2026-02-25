## Paper Experiment Shell Wrappers (OSD-679)

These scripts run the **runnable** parts of the manuscript protocol against the current `RETFoundLoRA/run.py` CLI.

### Environment Variables (optional overrides)
- `PYTHON_BIN` (default: `python3`)
- `RUN_PY` (default: `RETFoundLoRA/run.py`)
- `BACKBONE_CKPT` (default: MAE-transductive RETFound checkpoint)
- `OUT_ROOT` (default: `outputs`)
- `CSV_PATH` (default: use `run.py` default metadata csv)

### Usage
Core experiments:
```bash
bash scripts/paper/run_all_core_experiments.sh exp01_ctrl_cv
bash scripts/paper/run_all_core_experiments.sh exp02_hls_ood
bash scripts/paper/run_all_core_experiments.sh exp03_inter_eye
bash scripts/paper/run_all_core_experiments.sh exp04_saliency
```

Backbone ablation:
```bash
bash scripts/paper/run_backbone_ablation.sh
```

### Storage / Checkpoint cleanup policy
Current repo state does not require cleanup for script preparation, but for large sweeps:
- Keep:
  - best checkpoints
  - checkpoints tied to figures/tables in manuscript
  - latest training checkpoints for resumability
- Safe to remove first:
  - smoke-test checkpoints (`*smoke*`)
  - failed/abandoned ablation checkpoints
  - duplicate checkpoints with clearly worse metrics and no dependent analyses

