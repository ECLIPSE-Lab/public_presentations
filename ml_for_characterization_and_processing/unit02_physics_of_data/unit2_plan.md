# Unit 2 Plan — ML for Characterization and Processing

## Unit title
Image Formation and Physics of Data

## Audience/profile
- 5th semester undergrad
- Depends on MFML Unit 1 (risk/validation/uncertainty)
- Goal: connect measurement physics to ML preprocessing/model choices

## Learning objectives
By the end of Unit 2 students can:
1. Explain how measurement physics shapes image/signal statistics.
2. Identify common artifacts and how they induce ML failure.
3. Connect sampling, resolution, noise, and aliasing to pipeline design.
4. Choose basic preprocessing steps with explicit assumptions.
5. Build a leakage-aware baseline for image/signal tasks.

## 90-min structure
- 0–10: recap + why data formation matters
- 10–35: signal/image formation basics and artifact taxonomy
- 35–55: sampling, resolution, Fourier intuition, denoising rationale
- 55–75: preprocessing-to-model pipeline and failure patterns
- 75–85: case sketch + split/metric discussion
- 85–90: summary + exercise handoff

## Exercise (90 min)
- inspect raw/processed image or spectrum pairs
- compare two preprocessing pipelines
- train one baseline model and compare split strategies
- document one artifact-induced failure and mitigation

## Book mapping (priority)
1. Neuer: explainability and model trust in technical workflows
2. Sandfeld: domain knowledge integration in data science
3. McClarren: practical ML framing for physical systems
4. Murphy/Bishop: overfitting/model-selection language
