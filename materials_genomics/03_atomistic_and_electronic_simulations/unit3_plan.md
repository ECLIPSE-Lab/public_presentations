# Unit 3 Plan — Materials Genomics

## Unit title
Materials Databases and Thermodynamic Quantities

## Unit focus
Build a reliable data foundation for materials ML by understanding databases, schemas, thermodynamic targets, and known biases.

## Book-chapter anchors used for scaffold design
- Sandfeld 2.2; Neuer 4.2.2; Neuer 4.4.1; McClarren Ch4; Bishop 3.1–3.3

## Learning objectives
By the end of Unit 3, students can:
1. Explain the main modeling and data concepts behind **Materials Databases and Thermodynamic Quantities**.
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
  - `neuer-machine-learning-for-engineers/markdown/01-data-as-the-basis-of-models.qmd` (1.3)
  - `neuer-machine-learning-for-engineers/markdown/04-supervised-learning.qmd` (4.2.2, 4.2.3, 4.4.1)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/04-part-i-introduction-and-foundations.qmd` (2.2, 4.5)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/01-the-landscape-of-machine-learning-supervised-and-unsupervised-learning-optimization-and-other-topics.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/07-linear-models-for-regression.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/14-linear-regression.qmd`

## Cross-book summary target
- Use Neuer to frame database targets as supervised-learning objects with explicit train/test logic and careful normalization.
- Use Sandfeld to connect materials datasets to provenance, domain context, and structured records rather than isolated numbers.
- Use Bishop and Murphy selectively for regression language around targets, residuals, and fair comparison.
- Keep the unit centered on real materials records: composition, structure, calculation metadata, formation energy, and energy above hull.
- Exclude extensive regression derivations; the priority is scientific meaning of database fields, convex-hull logic, and reproducible queries.

## 50-slide strategy
- Slides 1-8: database ecosystem and why database choice changes conclusions.
- Slides 9-20: schema elements, identifiers, composition/structure/provenance fields.
- Slides 21-32: formation energy, energy above hull, convex hulls, metastability, synthesis caveats.
- Slides 33-42: confounders from functionals, cutoffs, duplicate structures, normalization choices.
- Slides 43-50: API query workflow, reproducible snapshotting, exercise briefing.

## Website summary update
- Heading: `#### Week 3 – Atomistic and electronic simulations (DFT, MD, MC) (28.04.2026)`
- Add a summary emphasizing:
  - which atomistic simulations populate materials databases,
  - how thermodynamic targets are constructed from those calculations,
  - why method metadata and reference states matter for ML.
