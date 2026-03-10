# Unit 2 Plan — Materials Genomics

## Unit title
Crystal Structure Fundamentals for Data-Driven Materials Discovery

## Audience/profile
- 5th semester undergrad
- Depends on MFML Unit 1 language (task/risk/validation)
- Goal: make structure representations and symmetry constraints concrete

## Learning objectives
By the end of Unit 2 students can:
1. Describe crystal structures as ML-ready data objects.
2. Explain symmetry, lattices, and Wyckoff-style constraints conceptually.
3. Relate structure representation choices to model performance and validity.
4. Identify leakage/bias risks specific to crystal datasets.
5. Prepare a minimal structure-feature pipeline for exercises.

## 90-min structure
- 0–10: recap + role of structure in MG
- 10–30: lattices, unit cells, symmetry concepts
- 30–50: structure encodings and invariances
- 50–70: data quality + split strategy for crystal datasets
- 70–85: practical pipeline sketch (CIF -> features -> baseline)
- 85–90: summary + exercise handoff

## Exercise (90 min)
- parse CIF files and basic descriptors
- run a first grouped split by composition/structure family
- train simple baseline and inspect one failure mode
- report one data-quality caveat

## Book mapping (priority)
1. Neuer: model trust and explainability framing
2. Sandfeld: materials data science + domain integration
3. McClarren: practical ML task structure
4. Murphy/Bishop: validation and probabilistic interpretation

## Required chapter files
- Neuer:
  - `neuer-machine-learning-for-engineers/markdown/01-data-as-the-basis-of-models.qmd` (1.2.3, 1.2.7)
  - `neuer-machine-learning-for-engineers/markdown/05-unsupervised-learning.qmd` (5.2)
- Sandfeld:
  - `sandfeld-materials-data-science/markdown/04-part-i-introduction-and-foundations.qmd` (3.3, structured/tabular data)
- McClarren:
  - `mcclarren-machine-learning-for-engineers/markdown/04-finding-structure-within-a-data-set-data-reduction-and-clustering.qmd`
- Bishop:
  - `bishop-pattern-recognition-and-machine-learning-2006/markdown/16-continuous-latent-variables.qmd`
- Murphy:
  - `murphy-machine-learning-a-probabilistic-perspective-2012/markdown/19-latent-linear-models.qmd`

## Cross-book summary target
- Use Sandfeld and Neuer to explain how crystal structures must be represented as structured data objects before any ML method can act on them.
- Use McClarren, Murphy, and Bishop only to motivate low-dimensional structure, covariance views, and representation choices, not to teach crystallography itself.
- Keep the domain core on lattice, basis, periodicity, symmetry, and invariance requirements for ML-ready crystal data.
- Explain why representation choices change both model accuracy and leakage risk.
- Exclude formal derivations of PCA and latent-variable models; students already meet the mathematics in MFML.

## 50-slide strategy
- Slides 1-10: structural vocabulary, lattices, bases, unit cells, coordinate systems.
- Slides 11-22: primitive vs conventional cells, symmetry, periodicity, invariances.
- Slides 23-34: CIF/POSCAR-style encodings, tabularization, low-dimensional structure intuition.
- Slides 35-44: data-quality risks, polymorph/prototype leakage, grouped splits.
- Slides 45-50: CIF-to-feature exercise setup and exam checklist.

## Website summary update
- Heading: `#### Week 2 – Simulation methods as data generators (21.04.2026)`
- Add or revise the summary so Week 2 bridges simulation outputs to ML-ready crystal representations:
  - structures as data objects with periodic constraints,
  - symmetry and coordinate choices,
  - low-dimensional organization of crystal data.
