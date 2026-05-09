# Unit 13 Plan — Materials Genomics

## Unit title
Physical Constraints, Trust, and Integration Outlook

## Unit focus
Integrate constraints, explainability, and workflow governance to make materials ML scientifically trustworthy.

## Book-chapter anchors used for scaffold design
- Neuer 6.1–6.3; Neuer Ch7; McClarren Ch11+12; Murphy model selection; Bishop model limits

## Learning objectives
By the end of Unit 13, students can:
1. Explain the main modeling and data concepts behind **Physical Constraints, Trust, and Integration Outlook**.
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
  - `neuer-machine-learning-for-engineers/markdown/06-physics-informed-learning.qmd` (6.1-6.3)
  - `neuer-machine-learning-for-engineers/markdown/07-explainability.qmd`
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/04-part-i-introduction-and-foundations.qmd` (2.1.4, 2.3)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/01-the-landscape-of-machine-learning-supervised-and-unsupervised-learning-optimization-and-other-topics.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/05-introduction.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/08-introduction.qmd`

## Cross-book summary target
- Use Neuer as the primary source for physics-informed learning, embedded analytical expressions, uncertainty, and explainability.
- Use Sandfeld to reconnect model trust to domain knowledge and to the transition from data to knowledge.
- Use McClarren, Bishop, and Murphy only to reinforce scientific-model limitations, parsimony, and responsible interpretation.
- Keep the unit centered on charge neutrality, symmetry, stability bounds, workflow governance, and limits of data-driven discovery.
- Exclude broad AI-ethics abstraction; the discussion should stay on concrete scientific trust and deployment practice.

## 50-slide strategy
- Slides 1-10: why constraints are needed in scientific ML.
- Slides 11-22: hard vs soft constraints, penalty terms, embedded physics, hybrid models.
- Slides 23-34: explainability, sensitivity, uncertainty statements, model cards, reproducibility packages.
- Slides 35-44: association vs mechanism, workflow integration, scientific trust failure modes.
- Slides 45-50: end-of-course synthesis, mini-project briefing, exam checklist.

## Website summary update
- Heading: `#### Week 14 – Physical constraints, limits, and outlook (14.07.2026)`
- Add a summary covering:
  - physical constraints as guardrails for materials ML,
  - explainability and reproducibility as trust requirements,
  - realistic limits of data-driven discovery and integration with experiments.
