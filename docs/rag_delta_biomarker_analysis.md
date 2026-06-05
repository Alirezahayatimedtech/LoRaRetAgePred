# RAG and DeltaRAG Biomarker Analysis

This note documents the RETFound-LoRA retinal age gap (RAG) experiments on the OSD-679 Brown Norway rat OCT data.

## Objective

The working hypothesis was that a retinal age model trained only on control rats could expose a stress/recovery phenotype in hindlimb suspension (HLS) rats. The primary derived scores were:

- `RAG = predicted retinal age - chronological age`
- `DeltaRAG(day) = RAG(day) - RAG(day 0)` within rat/eye when baseline was available
- `cohort_day_centered_RAG`: RAG after subtracting the control mean for the same study day

The main question was whether RAG or DeltaRAG correlates with ocular biomarkers and/or separates HLS from controls across days.

## Final Model Used For The Main Analysis

- Backbone: RETFound NatureOCT checkpoint
- Adaptation: LoRA on the last RETFound blocks
- Input unit: attention-MIL bag per rat/eye/day, using all available OCT views for that rat/eye/day
- Training set: Controls only
- Evaluation: out-of-fold controls plus HLS rats
- Cohorts: 1, 2, and 3
- Days: all available days, not limited to day 0 and day 90
- Cross-validation: 3 folds by rat, so control evaluations are out-of-fold

The result tables are in `results/rag_delta_biomarker/`.

## Age Prediction Performance

Control out-of-fold performance was stable across folds:

| Fold | Control MAE | Control R2 | HLS MAE | HLS R2 |
|---:|---:|---:|---:|---:|
| 0 | 33.59 days | 0.741 | 32.72 days | 0.762 |
| 1 | 31.35 days | 0.729 | 33.37 days | 0.770 |
| 2 | 31.35 days | 0.742 | 34.13 days | 0.730 |

This supports the model as a usable chronological age regressor, but not by itself as a strong HLS classifier.

## HLS-vs-Control DeltaRAG Separation

DeltaRAG did not strongly separate HLS from controls in the current data.

| Follow-up day | Control n | HLS n | HLS minus control DeltaRAG | AUROC, lower DeltaRAG = HLS | Welch p |
|---:|---:|---:|---:|---:|---:|
| 7 | 9 | 10 | -4.88 | 0.544 | 0.511 |
| 14 | 21 | 22 | -10.99 | 0.615 | 0.093 |
| 28 | 10 | 10 | -2.76 | 0.520 | 0.636 |
| 90 | 70 | 76 | -2.35 | 0.507 | 0.783 |
| 97 | 7 | 6 | 5.16 | 0.429 | 0.730 |
| 104 | 20 | 20 | 4.74 | 0.450 | 0.586 |
| 118 | 4 | 4 | -23.49 | 0.750 | 0.337 |
| 180 | 25 | 23 | 9.84 | 0.438 | 0.542 |

Interpretation: DeltaRAG is not currently powerful as a direct HLS-vs-control classifier. The day-14 trend is suggestive, but it is not statistically strong enough to claim group separation.

## Strongest Biomarker Signal

The strongest robust result was not group classification. It was a recovery-phase association between RAG and choroidal thickness.

Top residualized result:

- Scope: recovery days 97, 104, 118, and 180
- Subset: HLS only
- Score: raw or cohort-day-centered RAG
- Biomarker: temporal choroidal thickness
- n = 17
- Residualized Pearson r = -0.849
- p = 1.7e-5
- BH-adjusted q = 0.0036

This remains the best evidence that model-predicted retinal age is capturing a biological ocular phenotype related to post-HLS recovery/remodeling.

## DeltaRAG Biomarker Signal

DeltaRAG correlations were weaker and did not survive multiple-testing correction in the current sample.

The strongest DeltaRAG result was:

- Day 180
- Delta temporal choroidal thickness
- n = 11
- Pearson r = -0.670
- Spearman r = -0.685
- nominal p about 0.02
- BH q not significant

Interpretation: this is a candidate signal, not a validated finding.

## Biomarker-Fusion Model

An exploratory model was added that can feed ocular biomarkers into the RETFound-LoRA MIL head with missing-value masks. The run was stopped after weak early results:

- Fold 0 control MAE worsened to 35.42 days compared with 33.59 days for image-only fold 0.
- Fold 1 was also not clearly improving relative to the image-only benchmark.

This path is documented as exploratory only. It also creates a circularity problem if the same biomarkers are used as model inputs and then tested for correlation with RAG. For biomarker-discovery claims, the image-only RAG model is the cleaner primary analysis.

## Current Conclusion

The defensible claim is:

> RETFound-LoRA RAG is not yet a strong binary HLS classifier, but it behaves like a continuous retinal/ocular stress-recovery phenotype. Its strongest current biological association is with recovery-phase choroidal thickness, especially temporal choroid in HLS rats.

The practical direction is to treat RAG as an imaging-derived biomarker endpoint, not as a standalone HLS detector.
