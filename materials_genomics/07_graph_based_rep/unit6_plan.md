# Unit 5 Plan — Materials Genomics

## Unit title
Graph-Based Crystal Representations

## Unit focus
Model crystals as periodic graphs and understand why graph inductive biases improve structure–property learning.

## Book-chapter anchors used for scaffold design
- Sandfeld 2.2; Neuer 4.5.1–4.5.4; McClarren Ch8; Bishop Ch5; Murphy representation basics

## Learning objectives
By the end of Unit 5, students can:
1. Explain the main modeling and data concepts behind **Graph-Based Crystal Representations**.
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
  - `neuer-machine-learning-for-engineers/markdown/04-supervised-learning.qmd` (4.5.1-4.5.5)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/04-part-i-introduction-and-foundations.qmd` (2.2, 3.3)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/05-feed-forward-neural-networks.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/09-neural-networks.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/35-deep-learning.qmd`

## Cross-book summary target
- Use Neuer, McClarren, Bishop, and Murphy to supply the neural-network language needed to explain message passing as a learned nonlinear update rule.
- Use Sandfeld to keep the materials focus on structure encoding, domain constraints, and why graph objects are natural for crystals.
- Emphasize node, edge, and global attributes; periodic boundary conditions; and readout functions for property prediction.
- Compare graph models to descriptor baselines without getting lost in architecture trivia.
- Exclude advanced GNN math and equivariant formalism beyond intuition.

## 50-slide strategy
- Slides 1-10: why crystals become graphs, atoms/bonds/global state, periodicity.
- Slides 11-22: cutoff construction, edge features, invariance and equivariance intuition.
- Slides 23-34: message passing, CGCNN, MEGNet, SchNet-style intuition.
- Slides 35-44: over-smoothing, readout choices, transferability, shortcut-learning failures.
- Slides 45-50: graph-construction exercise and recap.

## Website summary update
- Heading: `#### Week 6 – Graph-based crystal representations (19.05.2026)`
- Add a summary covering:
  - crystals as periodic graphs,
  - message passing as a structure-property learning mechanism,
  - the role of cutoffs, invariances, and grouped evaluation.
