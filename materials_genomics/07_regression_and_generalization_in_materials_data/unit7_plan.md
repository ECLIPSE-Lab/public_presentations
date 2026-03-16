# Unit 7 Plan — Materials Genomics

## Unit title
Regression and Generalization in Materials Data

## Unit focus
Train property predictors with rigorous validation, focusing on generalization under chemical and structural shift.

## Book-chapter anchors used for scaffold design
- Neuer 4.2.2; Neuer 4.5.9; McClarren Ch4+6; Bishop 3.1–3.3; Murphy model selection

## Learning objectives
By the end of Unit 7, students can:
1. Frame materials-property prediction as a regression and generalization problem rather than a leaderboard problem.
2. Compare linear, regularized, tree-based, and simple neural baselines under the same validation protocol.
3. Explain why grouped chemistry-aware or prototype-aware splits are more honest than naive random splits.
4. Diagnose failure modes using residual analysis, learning curves, and OOD-style evaluation.
5. Design a reproducible baseline benchmark for one materials property.

## 90-minute lecture structure
- 0–10 min: dependency recap + notation alignment
- 10–35 min: concepts and methods (book-backed foundations)
- 35–60 min: materials-domain translation and modeling choices
- 60–80 min: validation, uncertainty, and failure analysis
- 80–90 min: summary + exercise handoff

## Exercise (90 min)
- build one reproducible regression benchmark for a materials property
- compare at least two baseline model families on identical grouped splits
- analyze residuals by chemistry or structure family
- report one generalization failure mode and one mitigation idea

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
- Exclude extended statistical proofs and Bayesian derivations; keep the discussion operational and benchmark-oriented.

## 50-slide strategy
- Slides 1-10: target selection, regression framing, baseline metrics.
- Slides 11-22: linear and regularized models, basis expansions, interpretability.
- Slides 23-34: tree/ensemble baselines, grouped splits, cross-validation, residual analysis.
- Slides 35-44: split design, learning curves, OOD behavior, deployment trust checks.
- Slides 45-50: baseline-comparison exercise and summary.

## Website summary update
- Heading: `#### Week 8 – Regression and generalization in materials data (02.06.2026)`
- Add a summary covering:
  - regression as empirical-risk minimization for materials targets,
  - grouped validation and generalization gaps,
  - why chemistry-aware splits matter more than raw accuracy.
