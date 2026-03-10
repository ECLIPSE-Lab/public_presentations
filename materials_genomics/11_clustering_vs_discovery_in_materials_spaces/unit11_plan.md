# Unit 11 Plan — Materials Genomics

## Unit title
Clustering vs Discovery in Materials Spaces

## Unit focus
Clarify what clustering can and cannot claim in scientific discovery; connect unsupervised structure to actionable discovery pipelines.

## Book-chapter anchors used for scaffold design
- Neuer 5.3; Sandfeld cluster analysis; Bishop Ch9; Murphy Ch11; McClarren Ch9 context

## Learning objectives
By the end of Unit 11, students can:
1. Explain the main modeling and data concepts behind **Clustering vs Discovery in Materials Spaces**.
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
  - `neuer-machine-learning-for-engineers/markdown/05-unsupervised-learning.qmd` (5.3, 5.4, 5.5.3)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/06-part-iii-classical-machine-learning.qmd` (clustering-related sections)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/04-finding-structure-within-a-data-set-data-reduction-and-clustering.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/13-mixture-models-and-em.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/18-mixture-models-and-the-em-algorithm.qmd`
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/32-clustering.qmd`

## Cross-book summary target
- Use Neuer and McClarren to explain objective-based clustering, low-dimensional embeddings, and outlier logic.
- Use Bishop and Murphy to distinguish hard assignments, soft assignments, and mixture-model interpretations.
- Keep the scientific message explicit: clustering can reveal structure, but discovery requires hypotheses, validation, and uncertainty-aware follow-up.
- Emphasize artifact-driven clusters, source-driven clusters, and the difference between novelty and noise.
- Exclude EM derivations and advanced density-estimation theory.

## 50-slide strategy
- Slides 1-10: clustering objectives versus discovery objectives.
- Slides 11-22: k-means, hierarchical, density-based, and mixture-based clustering intuition.
- Slides 23-34: raw descriptor space vs latent space, validity indices, artifact clusters.
- Slides 35-44: outliers, novelty, candidate identification, discovery loop.
- Slides 45-50: clustering-comparison exercise and summary.

## Website summary update
- Heading: `#### Week 12 – Clustering, uncertainty, and discovery logic`
- Add a summary covering:
  - clustering as an exploratory tool rather than a discovery proof,
  - novelty/outlier logic,
  - the need to connect latent-space structure to validation and uncertainty.
