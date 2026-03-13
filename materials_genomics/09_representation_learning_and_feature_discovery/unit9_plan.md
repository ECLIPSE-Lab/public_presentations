# Unit 9 Plan — Materials Genomics

## Unit title
Representation Learning and Feature Discovery

## Unit focus
Move from engineered features to learned embeddings and evaluate what is truly gained in transfer and discovery performance.

## Book-chapter anchors used for scaffold design
- Neuer 5.5; McClarren Ch5; Bishop 12.3; Sandfeld 2.2; Murphy latent variable intuition

## Learning objectives
By the end of Unit 9, students can:
1. Explain the main modeling and data concepts behind **Representation Learning and Feature Discovery**.
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
  - `sandfeld-materials-data-science/markdown/07-part-iv-artificial-neural-networks-and-deep-learning.qmd`
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/08-unsupervised-learning-with-neural-networks-autoencoders.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/16-continuous-latent-variables.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/19-latent-linear-models.qmd`
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/35-deep-learning.qmd`

## Cross-book summary target
- Use Neuer and McClarren to explain autoencoders and hidden representations as tools for feature discovery rather than only compression.
- Use Bishop and Murphy to provide latent-variable intuition and transferability language.
- Use Sandfeld to keep the unit grounded in scientific feature interpretation and domain-aware embedding use.
- Emphasize learned embeddings, feature discovery, transfer across chemical systems, and the distinction between useful and merely visually appealing representations.
- Exclude self-supervised objective details beyond concept-level motivation.

## 50-slide strategy
- Slides 1-10: engineered vs learned features and why the distinction matters.
- Slides 11-22: autoencoder and embedding intuition, bottlenecks, reconstruction objectives.
- Slides 23-34: transferability, robustness, downstream regressors, hidden-layer interpretation.
- Slides 35-44: failure cases, source bias, visually clean but useless embeddings.
- Slides 45-50: representation-comparison exercise and recap.

## Website summary update
- Heading: `#### Week 10 – Representation learning and feature discovery (16.06.2026)`
- Add a summary covering:
  - learned embeddings versus engineered descriptors,
  - transferability across chemistry families,
  - how hidden representations can reveal or obscure materials trends.
