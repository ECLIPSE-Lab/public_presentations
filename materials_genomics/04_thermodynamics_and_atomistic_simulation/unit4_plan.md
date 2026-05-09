# Unit 4 Plan — Materials Genomics

## Unit title
From Classical Descriptors to Learned Representations

## Unit focus
Understand descriptor design choices, limits of hand-crafted features, and criteria for when learned representations are justified.

## Book-chapter anchors used for scaffold design
- Sandfeld 2.1–2.2; Neuer 5.5; McClarren Ch5; Murphy Ch1; Bishop 12.1–12.3

## Learning objectives
By the end of Unit 4, students can:
1. Explain the main modeling and data concepts behind **From Classical Descriptors to Learned Representations**.
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
  - `neuer-machine-learning-for-engineers/markdown/05-unsupervised-learning.qmd` (5.5)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/04-part-i-introduction-and-foundations.qmd` (2.2, 2.4)
  - `sandfeld-materials-data-science/markdown/06-part-iii-classical-machine-learning.qmd` (feature matrices, regression context)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/04-finding-structure-within-a-data-set-data-reduction-and-clustering.qmd`
  - `mcclarren-machine-learning-for-engineers/markdown/08-unsupervised-learning-with-neural-networks-autoencoders.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/16-continuous-latent-variables.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/19-latent-linear-models.qmd`

## Cross-book summary target
- Use Sandfeld to motivate descriptor engineering through domain knowledge, feature matrices, and curse-of-dimensionality effects.
- Use Neuer plus McClarren to explain why autoencoder-style learned representations become attractive when hand-crafted features saturate.
- Use Bishop and Murphy only for latent-variable intuition, not for deep theory.
- Keep the materials focus on descriptor families such as Magpie and matminer, invariance requirements, and failure modes from multicollinearity or missing nonlocal physics.
- Exclude architecture-specific training detail; that belongs in later neural-network units.

## 50-slide strategy
- Slides 1-10: descriptor purpose, chemistry and structure feature families, invariance requirements.
- Slides 11-22: Magpie/matminer examples, scaling, normalization, correlation, and sparsity.
- Slides 23-34: where classical descriptors succeed, where they fail, and why.
- Slides 35-44: transition to learned representations, latent-variable intuition, transferability.
- Slides 45-50: descriptor-vs-learned-representation exercise and summary.

## Website summary update
- Heading: `#### Week 4 – Continuum simulations, thermodynamics, and stability (05.05.2026)`
- Add a summary that links stability concepts to representation choices:
  - descriptors as compressed carriers of chemistry and structure,
  - interpretable but limited hand-crafted features,
  - motivation for moving toward learned representations in later weeks.
