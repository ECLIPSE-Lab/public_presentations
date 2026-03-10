# Unit 7 Plan — Materials Genomics

## Unit title
Regression and Generalization in Materials Data

## Unit focus
Train property predictors with rigorous validation, focusing on generalization under chemical and structural shift.

## Book-chapter anchors used for scaffold design
- Neuer 4.2.2; Neuer 4.5.9; McClarren Ch4+6; Bishop 3.1–3.3; Murphy model selection

## Learning objectives
By the end of Unit 7, students can:
1. Explain the main modeling and data concepts behind **Regression and Generalization in Materials Data**.
2. Map these concepts to materials-discovery decisions and failure modes.
3. Apply leakage-aware validation logic in practical workflows.
4. Distinguish what is lecture-essential vs what belongs in exercise implementation.

## 90-minute lecture structure
- 0–10 min: dependency recap + notation alignment
- 10–35 min: concepts and methods (book-backed foundations)
- 35–60 min: materials-domain translation and modeling choices
- 60–80 min: validation, uncertainty, and failure analysis
- 80–90 min: summary + exercise handoff

## Exercise (90 min)
- implement a minimal reproducible pipeline for the unit topic
- compare two methodological choices under identical split protocol
- perform one structured failure analysis and mitigation proposal
- produce a short report with claims, evidence, and limitations

## Required chapter files
- Neuer:
  - `neuer-machine-learning-for-engineers/markdown/04-supervised-learning.qmd` (4.2.2, 4.2.3, 4.5.9)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/06-part-iii-classical-machine-learning.qmd` (12-13)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/02-linear-models-for-regression-and-classification.qmd`
  - `mcclarren-machine-learning-for-engineers/markdown/03-decision-trees-and-random-forests-for-regression-and-classification.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/07-linear-models-for-regression.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/14-linear-regression.qmd`

## Cross-book summary target
- Use Neuer to define regression tasks, train/test logic, and overfitting control.
- Use Sandfeld, Bishop, and Murphy for regression geometry, feature matrices, basis functions, and the bias-variance trade-off.
- Use McClarren for baseline model families that work well in engineering settings.
- Keep the unit centered on fair comparison of materials-property predictors under grouped chemistry-aware validation.
- Exclude extended statistical proofs and Bayesian derivations; keep the discussion operational.

## 50-slide strategy
- Slides 1-10: target selection, regression framing, baseline metrics.
- Slides 11-22: linear and regularized models, basis expansions, interpretability.
- Slides 23-34: tree/ensemble baselines, grouped splits, cross-validation, residual analysis.
- Slides 35-44: bias-variance, overfitting signatures, OOD behavior in chemical space.
- Slides 45-50: baseline-comparison exercise and summary.

## Website summary update
- Heading: `#### Week 8 – Regression and generalization in materials data (02.06.2026)`
- Add a summary covering:
  - regression as empirical-risk minimization for materials targets,
  - grouped validation and generalization gaps,
  - why chemistry-aware splits matter more than raw accuracy.
