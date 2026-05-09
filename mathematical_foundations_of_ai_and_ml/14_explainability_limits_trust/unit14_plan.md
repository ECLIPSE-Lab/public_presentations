# Unit 14 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: 13 prior MFML units covering linear algebra, regression, NNs, backprop, optimization, generalization, probability, representation learning, latent spaces, unsupervised learning, uncertainty, and physics-informed learning
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)
- **This is the FINAL lecture of the course.**

## Learning objectives (Unit 14)
By the end of the unit, students can:
1. Explain why explainability is a scientific and industrial mandate, not an optional add-on.
2. Distinguish semantic ordering structures (synonyms, taxonomies, ontologies) and apply them to technical process descriptions.
3. Map the six levels of explainability (E1–E6) to appropriate audiences and methods.
4. Perform and interpret a perturbation-based sensitivity analysis on a black-box model.
5. Use ontological information for deductive reasoning about model predictions.
6. Distinguish detection from prediction in causal process chains and assess where ML adds value.
7. Articulate data manifold limits and the role of inductive bias in model trustworthiness.
8. Synthesize the full 14-unit arc of the course into a coherent scientific methodology.

## Book-aligned content mapping (priority order)
1. Neuer (2024): Ch. 7 — Explainability (semantic structures, sensitivity analysis, levels of explainability, causality in process chains).
2. McClarren (2021): Ch. 12 — Limitations and Outlook (contextual).
3. Bishop (2006): Ch. 1 §1.1–1.2 — Reflection on model limits (optional depth).

## 90-minute structure
- 0–10 min: The trust mandate — why explainability is non-negotiable in science and industry
- 10–25 min: Semantic structures for digitizing meaning (synonyms, taxonomies, ontologies)
- 25–40 min: Levels of explainability (E1–E6) and audience-appropriate explanations
- 40–55 min: Sensitivity analysis — perturbation theory for probing black-box models
- 55–65 min: Deductive reasoning with ontologies and causality in process chains
- 65–75 min: Data manifold limits, inductive bias, and when models fail
- 75–87 min: Course retrospective — the 14-unit arc from learning to trust
- 87–90 min: Exam preparation guidance + farewell

## Exercise (90 min)
- Mini end-to-end synthesis project:
  - Given a synthetic multi-step process dataset, students build a complete ML pipeline
  - Construct a simple ontology for the process variables
  - Train a model (NN or decision tree) and perform perturbation-based sensitivity analysis
  - Interpret results using ontological context and deductive reasoning
  - Write a short explainability report addressing at least two audience levels (E2 process, E4 model)
  - Reflect on manifold limits and where the model should not be trusted

## Assessment alignment
- This unit ties together all prior exam-relevant concepts.
- Exam-relevant "must-know" statements consolidate the full course.
- The retrospective block explicitly maps each unit to exam topics.
- No new reading assignment (final unit).

## Slide production notes
- Style: follow data_science_for_em revealjs pattern
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
- Include a course retrospective block linking all 14 units
- Final slide: exam preparation checklist and acknowledgments
