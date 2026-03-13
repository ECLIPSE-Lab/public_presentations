# Unit 8 Plan — Materials Genomics

## Unit title
Neural Networks for Materials Properties

## Unit focus
Use neural surrogates responsibly for structure–property prediction, with attention to training dynamics and robustness.

## Book-chapter anchors used for scaffold design
- Neuer 4.5.1/4.5.3/4.5.4/4.5.5; McClarren Ch8; Bishop 5.1–5.3; Murphy deep model caveats; Sandfeld domain context

## Learning objectives
By the end of Unit 8, students can:
1. Explain the main modeling and data concepts behind **Neural Networks for Materials Properties**.
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
  - `neuer-machine-learning-for-engineers/markdown/04-supervised-learning.qmd` (4.5.1-4.5.5, 4.5.9)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/07-part-iv-artificial-neural-networks-and-deep-learning.qmd`
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/05-feed-forward-neural-networks.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/09-neural-networks.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/35-deep-learning.qmd`

## Cross-book summary target
- Use Neuer as the primary source for neuron models, activations, training, optimization, and overfitting control.
- Use Sandfeld and McClarren to keep the narrative tied to engineering surrogate models rather than generic AI examples.
- Use Bishop and Murphy selectively for backpropagation intuition and depth/capacity trade-offs.
- Keep the materials focus on property prediction under limited data, regularization, and extrapolation risk.
- Exclude advanced architecture variants and theory-heavy universal approximation arguments.

## 50-slide strategy
- Slides 1-10: why nonlinear surrogates are needed for materials properties.
- Slides 11-22: neurons, activations, network depth/width, loss functions.
- Slides 23-34: optimization, batch size, regularization, early stopping, diagnostics.
- Slides 35-44: multitask learning, data scarcity, extrapolation, interpretability limits.
- Slides 45-50: small-network training exercise and summary.

## Website summary update
- Heading: `#### Week 9 – Neural networks for materials properties (09.06.2026)`
- Add a summary covering:
  - neural networks as DFT-surrogate models,
  - training stability and regularization,
  - extrapolation and leakage as the main scientific risks.
