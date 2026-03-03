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
