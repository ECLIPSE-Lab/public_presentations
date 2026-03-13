# Unit 3 Plan — Machine Learning for Characterization and Processing

## Unit title
Signal and Image Preprocessing + Denoising

## Note
This scaffold is stored in `03_experimental_data_quality_and_ml_readiness/` to preserve existing repo structure; Unit-3 focus follows SS26 plan.

## Learning objectives
1. Explain preprocessing goals in terms of preserved/removed physical information.
2. Compare denoising approaches and identify distortion risks.
3. Connect preprocessing decisions to model bias/variance and leakage.
4. Build a minimal, auditable preprocessing pipeline for lab data.
5. Evaluate preprocessing variants with task-relevant metrics.

## 90-minute structure
- 0–10: recap + motivation from failure cases
- 10–30: noise/artifact taxonomy and signal formation reminders
- 30–50: filters, normalization, baseline correction, segmentation prep
- 50–70: denoising tradeoffs and frequency-domain intuition
- 70–85: pipeline comparisons + validation setup
- 85–90: summary + exercise handoff

## Exercise (90 min)
- preprocess raw microscopy/spectral sample with two pipelines
- train one baseline model per pipeline
- compare grouped/temporal split behavior
- document one artifact-induced model error
