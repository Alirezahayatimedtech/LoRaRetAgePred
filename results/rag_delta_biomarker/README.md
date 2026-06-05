# RAG / DeltaRAG Result Tables

These tables summarize the full 3-fold RETFound-LoRA NatureOCT RAG analysis across OSD-679 Cohorts 1-3 and all available study days.

## Main Tables

- `retfound_lora_natureoct_full_cv_metrics.csv`: fold-level age-prediction metrics for out-of-fold controls and HLS evaluation rows.
- `coverage_by_group_day.csv`: rat/day coverage for Controls and HLS.
- `deltarag_group_separation.csv`: HLS-vs-control DeltaRAG separation by follow-up day.
- `top10_rag_correlations.csv`: strongest raw and day-centered RAG biomarker correlations.
- `top10_delta_correlations.csv`: strongest DeltaRAG versus delta-biomarker correlations.
- `top10_residualized_correlations.csv`: strongest biomarker correlations after residualizing for basic covariates.

## Full Correlation Tables

The full correlation tables were generated locally from the same analysis run and are intentionally not committed because they are large. The compact top-10 tables above preserve the ranked findings used in the documentation.

## Exploratory Stopped Run

- `biomarker_fusion_stopped_partial_metrics.csv`

This table contains completed folds from the exploratory model that fed ocular biomarkers into the age-prediction head. It is included for transparency only. It should not be used as the main biomarker-discovery result because using biomarkers as model inputs and then correlating RAG with those same biomarkers is circular.
