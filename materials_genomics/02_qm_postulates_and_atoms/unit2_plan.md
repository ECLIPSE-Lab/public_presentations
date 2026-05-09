# Unit 2 Plan — Materials Genomics

## Unit title
Simulation Methods as Data Generators

## Audience/profile
- 5th semester undergrad
- Depends on MFML Unit 1 language (task/risk/validation)
- Goal: make simulation outputs, scales, and biases concrete before later representation units

## Learning objectives
By the end of Unit 2 students can:
1. Explain why simulation methods are the main generators of modern materials datasets.
2. Compare FEM, MD, MC, and DFT by scale, output type, cost, and dominant bias.
3. Relate simulation assumptions to downstream ML targets and dataset limitations.
4. Identify hidden confounders caused by numerical settings, approximations, and post-processing.
5. Choose a plausible simulation route for a given materials-property question.

## 90-min structure
- 0–10: recap + why simulations dominate materials data generation
- 10–30: scales and outputs of FEM, MD, MC, and DFT
- 30–50: assumptions, approximations, and cost-accuracy tradeoffs
- 50–70: what simulation outputs become ML targets, labels, or descriptors
- 70–85: method-selection case studies and failure modes
- 85–90: summary + exercise handoff

## Exercise (90 min)
- choose suitable simulation methods for several target properties
- inspect example database fields and infer which method produced them
- compare one high-fidelity and one low-fidelity route for the same screening task
- report one bias source and one mitigation strategy

## Book mapping (priority)
1. Sandfeld: materials-data and simulation context
2. Neuer: task framing, targets, and validation logic
3. McClarren: practical ML workflow framing
4. Murphy/Bishop: optional probabilistic depth only when needed

## Required chapter files
- Neuer:
  - `neuer-machine-learning-for-engineers/markdown/01-data-as-the-basis-of-models.qmd` (1.2.3, 1.3)
  - `neuer-machine-learning-for-engineers/markdown/04-supervised-learning.qmd` (4.2.2, 4.4.1)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/04-part-i-introduction-and-foundations.qmd` (2.2, 4.5)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/01-the-landscape-of-machine-learning-supervised-and-unsupervised-learning-optimization-and-other-topics.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/07-linear-models-for-regression.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/14-linear-regression.qmd`

## Cross-book summary target
- Use Sandfeld to explain why simulations are not just computational tools but controlled data generators with scale-specific outputs.
- Use Neuer to frame those outputs as supervised-learning targets and to highlight how assumptions and preprocessing shape generalization.
- Use McClarren, Bishop, and Murphy only to support language about targets, residuals, and validation; do not turn this unit into a generic regression lecture.
- Keep the domain core on FEM, MD, MC, DFT, their outputs, and their systematic biases.
- Exclude crystallographic detail and deep derivations; those belong in later representation and modeling units.

## 50-slide strategy
- Slides 1-10: why simulations generate materials data and what decision problems they support.
- Slides 11-22: FEM, MD, MC, and DFT across scales, outputs, and assumptions.
- Slides 23-34: cost-accuracy tradeoffs, hidden confounders, and metadata requirements.
- Slides 35-44: simulation outputs as labels, descriptors, or constraints in ML pipelines.
- Slides 45-50: method-selection exercise and exam checklist.

## Website summary update
- Heading: `#### Week 2 – Simulation methods as data generators (21.04.2026)`
- Add or revise the summary so Week 2 emphasizes:
  - simulations as controlled data generators,
  - differences between FEM, MD, MC, and DFT outputs,
  - tradeoffs between scale, accuracy, and bias,
  - why metadata matters before any ML model is trained.
