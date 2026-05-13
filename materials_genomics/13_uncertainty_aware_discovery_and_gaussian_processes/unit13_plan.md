# Unit 12 Plan — Materials Genomics

## Unit title
Uncertainty-Aware Discovery and Gaussian Processes

## Unit focus
Use predictive uncertainty to guide materials screening and exploration–exploitation decisions.

## Book-chapter anchors used for scaffold design
- Neuer 2.2; Neuer 6.4; McClarren Ch3; Murphy Ch15; Bishop Bayesian view

## Learning objectives
By the end of Unit 12, students can:
1. Explain the main modeling and data concepts behind **Uncertainty-Aware Discovery and Gaussian Processes**.
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
  - `neuer-machine-learning-for-engineers/markdown/06-physics-informed-learning.qmd` (6.4)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/05-part-ii-a-primer-on-probabilities-distributions-and-statistics.qmd` (probability, variance, covariance, sampling)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/01-the-landscape-of-machine-learning-supervised-and-unsupervised-learning-optimization-and-other-topics.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/06-probability-distributions.qmd`
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/10-kernel-methods.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/22-gaussian-processes.qmd`

## Cross-book summary target
- Use Neuer as the main source for uncertainty integration into learning workflows and for distinguishing data uncertainty from model uncertainty.
- Use Sandfeld and Bishop to refresh the probability and covariance language needed for Gaussian-process intuition.
- Use Murphy as the primary source for Gaussian-process regression, kernels, posterior mean/variance, and small-data advantages.
- Keep the materials focus on screening decisions, calibrated uncertainty, and exploration-versus-exploitation trade-offs.
- Exclude exact GP derivations and matrix-algebra detail beyond what is needed to interpret mean and variance outputs.

## 50-slide strategy
- Slides 1-10: why point predictions are insufficient for discovery.
- Slides 11-22: aleatoric vs epistemic uncertainty, calibration, reliability, covariance intuition.
- Slides 23-34: Gaussian-process prior/posterior intuition, kernels, posterior mean and variance.
- Slides 35-44: acquisition logic, GP scaling limits, ensembles and MC-dropout as alternatives.
- Slides 45-50: screening exercise and summary.

## Website summary update
- Heading: `#### Week 13 – Uncertainty-aware discovery and Gaussian Processes (07.07.2026)`
- Add a summary covering:
  - Gaussian Processes as small-data uncertainty-aware surrogates,
  - exploration-versus-exploitation logic,
  - the role of calibrated uncertainty in candidate ranking.
