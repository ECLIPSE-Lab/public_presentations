# Unit 6 Plan — Materials Genomics

## Unit title
Local Atomic Environments

## Unit focus
Use local environment descriptors to bridge interpretable chemistry and modern representation learning.

## Book-chapter anchors used for scaffold design
- Sandfeld 2.2; Neuer 6.2; Neuer 6.3; McClarren Ch4; Bishop kernels/feature maps

## Learning objectives
By the end of Unit 6, students can:
1. Explain the main modeling and data concepts behind **Local Atomic Environments**.
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
  - `neuer-machine-learning-for-engineers/markdown/06-physics-informed-learning.qmd` (6.2, 6.3)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/04-part-i-introduction-and-foundations.qmd` (2.2, 3.3)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/02-linear-models-for-regression-and-classification.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/10-kernel-methods.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/21-kernels.qmd`

## Cross-book summary target
- Use Neuer and Sandfeld to stress that local-environment features are a form of domain-guided data enrichment rather than arbitrary preprocessing.
- Use Bishop and Murphy to motivate kernel and similarity language for SOAP-like descriptors.
- Keep the materials content on coordination numbers, Voronoi views, atom-centered descriptors, and aggregation from local to material-level features.
- Show local descriptors as the bridge between interpretable classical features and learned graph representations.
- Exclude full kernel derivations and spherical-harmonic details.

## 50-slide strategy
- Slides 1-10: local vs global structure, neighbor shells, coordination environments.
- Slides 11-22: bond-length/bond-angle views, Voronoi tessellations, atom-centered features.
- Slides 23-34: SOAP/ACSF intuition, kernel similarity, aggregation to material-level vectors.
- Slides 35-44: defects, noise sensitivity, local-environment failure modes, transfer limits.
- Slides 45-50: descriptor computation exercise and recap.

## Website summary update
- Heading: `#### Week 7 – Local atomic environments (26.05.2026)`
- Add a summary covering:
  - local descriptors as ML-ready fingerprints,
  - Voronoi/SOAP intuition,
  - the bridge from interpretable environments to richer learned representations.
