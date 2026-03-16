# Unit 6 Plan — Materials Genomics

## Unit title
Local Atomic Environments

## Unit focus
Use local environment descriptors to bridge interpretable chemistry and modern representation learning.

## Book-chapter anchors used for scaffold design
- Sandfeld 2.2; Neuer 6.2; Neuer 6.3; McClarren Ch4; Bishop kernels/feature maps

## Learning objectives
By the end of Unit 6, students can:
1. Explain why local atomic environments are a natural compromise between global crystal descriptors and full graph models.
2. Compare coordination features, Voronoi descriptors, ACSF-style fingerprints, and SOAP-like descriptors at a conceptual level.
3. Explain how cutoff choice, periodic images, and aggregation affect the final material representation.
4. Identify failure modes such as parser errors, aliasing across polymorphs, and noise sensitivity in relaxed structures.
5. Build a simple local-environment pipeline suitable for downstream regression.

## 90-minute lecture structure
- 0–10 min: dependency recap + notation alignment
- 10–35 min: concepts and methods (book-backed foundations)
- 35–60 min: materials-domain translation and modeling choices
- 60–80 min: validation, uncertainty, and failure analysis
- 80–90 min: summary + exercise handoff

## Exercise (90 min)
- construct local neighborhoods for a small crystal dataset
- compute coordination statistics and one richer local descriptor
- aggregate local environments to a material-level vector
- compare descriptor behavior for one stable structure and one defective/noisy structure

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
- Exclude full kernel derivations and spherical-harmonic details; keep the treatment conceptual and operational.

## 50-slide strategy
- Slides 1-10: local vs global structure, neighbor shells, coordination environments.
- Slides 11-22: bond-length/bond-angle views, Voronoi tessellations, atom-centered features.
- Slides 23-34: SOAP/ACSF intuition, kernel similarity, aggregation to material-level vectors.
- Slides 35-44: cutoff choice, periodic neighborhoods, defects, noise sensitivity, and transfer limits.
- Slides 45-50: descriptor computation exercise and recap.

## Website summary update
- Heading: `#### Week 7 – Local atomic environments (26.05.2026)`
- Add a summary covering:
  - local descriptors as ML-ready fingerprints,
  - Voronoi/SOAP intuition,
  - the bridge from interpretable environments to richer learned representations.
