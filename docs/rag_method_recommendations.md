# RAG Method Recommendations

## Recommended Primary Method

Use the image-only RETFound-LoRA model trained on controls as the primary RAG generator.

Reasons:

- It avoids circular biomarker leakage.
- It uses all available OCT views per rat/eye/day through attention-MIL.
- It supports out-of-fold control calibration.
- It allows biomarker correlations to be interpreted as downstream associations rather than self-fulfilling model inputs.

## Recommended Statistical Framing

Primary analysis:

- Fit age model on controls only.
- Predict age for out-of-fold controls and HLS rats.
- Compute RAG at rat/eye/day.
- Compute DeltaRAG only when day-0 baseline exists.
- Use mixed-effects or clustered models for final manuscript statistics because repeated rat/eye/day observations are not independent.

Recommended model form for final inference:

```text
biomarker ~ RAG_or_DeltaRAG + group + day + RAG_or_DeltaRAG:group + (1 | rat_id)
```

For biomarkers with strong day effects, use day-centered RAG or include day as categorical.

## What Not To Claim Yet

Do not claim:

- DeltaRAG is a strong HLS classifier.
- Day-90 DeltaRAG separates HLS from controls.
- Biomarker-input age prediction proves biomarker discovery.

The current data do not support those claims.

## What Can Be Claimed

Reasonable current claims:

- Control-trained RETFound-LoRA predicts rat chronological age from OCT with about 31-34 day MAE across folds.
- RAG provides a continuous image-derived deviation score.
- Recovery-phase RAG is strongly associated with temporal choroidal thickness in HLS rats.
- DeltaRAG has candidate associations with choroidal/retinal thickness changes, but these need more power.

## Spaceflight-Relevant Use Case

For spaceflight analog studies, RAG can be framed as an integrated retinal stress-recovery index:

1. Establish a pre-flight or pre-HLS baseline from OCT.
2. Monitor RAG during exposure and recovery.
3. Compare RAG trajectory with ocular structure biomarkers such as retinal/choroidal thickness and IOP.
4. Use persistent abnormal RAG during recovery as a flag for incomplete ocular adaptation.

The value is not replacing detailed biomarkers. The value is summarizing distributed OCT information into one learned age-deviation score, then asking which biological systems explain the deviation.

## Next Experiments

Highest-value next experiments:

- Repeat the full analysis with mixed-effects models and rat-level clustered bootstrap confidence intervals.
- Pre-register a small number of biomarkers before testing: temporal choroid, nasal choroid, temporal retina, nasal retina, and IOP.
- Test phase-specific windows separately: stress days 7-90 and recovery days 97-180.
- Compare image-only RAG against explicit biomarker-only age models, but keep biomarker-input models separate from biomarker-discovery correlation claims.
- Add external validation if another cohort or mission analog dataset becomes available.
