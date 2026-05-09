# Unit 1 Plan — Materials Genomics

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: 2 semesters math, 1 physics, some chemistry
- Assumed: basic undergrad math, SVD familiarity, very basic Python
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 1)
By the end of this unit, students can:
1. Explain the “genomics” analogy for materials discovery without oversimplification.
2. Describe the structure–processing–property-performance graph as a data system.
3. Identify core materials databases and the types of targets they contain.
4. Recognize key sources of bias/incompleteness in materials datasets.
5. Formulate a first ML-ready materials query pipeline.

## Book-aligned content mapping
1. Neuer (2024): model framing, data-based modeling, uncertainty and limits.
2. Sandfeld: materials data science pipeline and domain-specific examples.
3. McClarren (2021): model realism in physical systems.
4. Murphy/Bishop: supervised/unsupervised framing language for later units.

## 90-minute structure
- 0–10 min: Course position in SS26 triad (MFML + ML-PC + MG)
- 10–25 min: What is materials genomics? promises vs reality
- 25–45 min: Data assets: MP/OQMD/AFLOW/NOMAD and core target quantities
- 45–60 min: Representation question: composition/structure/process as ML input
- 60–75 min: Biases, leakage, domain shift, and scientific validity
- 75–85 min: Discovery loop: screening → uncertainty → experiment feedback
- 85–90 min: Unit summary + exercise briefing

## Exercise (90 min)
- Query a materials database (or prepared subset)
- Build first feature table (composition + simple metadata)
- Explore one target (bandgap or formation energy)
- Diagnose one bias artifact and document mitigation

## Assessment alignment
- Written exam: conceptual precision (not coding trivia)
- Students should be able to defend model/data choices scientifically.

## Required chapter files
- Neuer:
  - `neuer-machine-learning-for-engineers/markdown/01-data-as-the-basis-of-models.qmd` (1.1-1.3)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/04-part-i-introduction-and-foundations.qmd` (2.1-2.3)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/01-the-landscape-of-machine-learning-supervised-and-unsupervised-learning-optimization-and-other-topics.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/05-introduction.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/08-introduction.qmd`

## Cross-book summary target
- Start from Neuer's distinction between white-box, grey-box, and black-box models and explain why materials genomics cannot treat screening as a pure black-box exercise.
- Use Sandfeld to define materials data science as a domain-knowledge-guided workflow from data to information to knowledge.
- Use McClarren, Bishop, and Murphy only to stabilize the ML vocabulary: task definition, model selection, validation, and scientific interpretation.
- Keep the focus on databases, discovery loops, and validity criteria rather than on algorithm derivations.
- Exclude detailed probability theory and optimization proofs; they belong to MFML.

## 50-slide strategy
- Slides 1-8: course position, genomics analogy, learning objectives, discovery bottleneck.
- Slides 9-18: design-space framing, PSPP graph, targets, surrogate-model logic.
- Slides 19-30: database landscape, data objects, provenance, bias, and leakage.
- Slides 31-41: regression/classification/ranking tasks, grouped validation, uncertainty-aware decisions.
- Slides 42-50: exercise handoff, reporting checklist, exam-relevant summary statements.

## Website summary update
- Heading: `#### Week 1 – What is Materials Genomics? (14.04.2026)`
- Add a short summary emphasizing:
  - materials genomes as searchable composition-structure spaces,
  - databases plus simulations as the data substrate,
  - the need for validation, uncertainty, and domain knowledge in discovery claims.
