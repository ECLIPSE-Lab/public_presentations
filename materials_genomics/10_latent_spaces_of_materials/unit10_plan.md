# Unit 10 Plan — Materials Genomics

## Unit title
Latent Spaces of Materials

## Unit focus
Build and interpret latent spaces for structure/property organization while avoiding over-interpretation of embeddings.

## Book-chapter anchors used for scaffold design
- Neuer 5.5.1–5.5.3; McClarren Ch5; Bishop 12.3–12.4; Murphy latent models; Sandfeld visualization context

## Learning objectives
By the end of Unit 10, students can:
1. Explain the main modeling and data concepts behind **Latent Spaces of Materials**.
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
  - `neuer-machine-learning-for-engineers/markdown/05-unsupervised-learning.qmd` (5.5.1-5.5.7)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/07-part-iv-artificial-neural-networks-and-deep-learning.qmd`
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/08-unsupervised-learning-with-neural-networks-autoencoders.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/16-continuous-latent-variables.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/19-latent-linear-models.qmd`

## Cross-book summary target
- Use Neuer as the main source for autoencoder topology, latent space, anomaly detection, and uncertainty extensions.
- Use McClarren, Bishop, and Murphy to make latent spaces interpretable as compressed but structured model coordinates.
- Use Sandfeld to keep the discussion tied to materials datasets, visualization, and scientific meaning rather than abstract manifold talk.
- Emphasize traversals, interpolation, neighborhood structure, and anomaly detection for materials families.
- Exclude full variational derivations; keep variational ideas conceptual.

## 50-slide strategy
- Slides 1-10: latent-space motivation, encoder-bottleneck-decoder workflow.
- Slides 11-22: compression, reconstruction loss, bottleneck dimension, regularization.
- Slides 23-34: traversals, interpolation, chemistry/structure trends, anomaly detection.
- Slides 35-44: collapse, source bias, latent-map failure modes, downstream utility.
- Slides 45-50: autoencoder exercise and recap.

## Website summary update
- Heading: `#### Week 11 – Latent spaces of materials (23.06.2026)`
- Add a summary covering:
  - autoencoders and embeddings as compressed materials coordinates,
  - interpretation of latent directions and anomalies,
  - the difference between reconstruction quality and scientific usefulness.
