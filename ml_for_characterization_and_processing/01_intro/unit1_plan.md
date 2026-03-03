# Unit 1 Plan — Machine Learning for Characterization and Processing

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: 2 semesters math, 1 physics, some chemistry
- Assumed: undergrad math, SVD familiarity, very basic Python
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 1)
By the end of this unit, students can:
1. Explain why characterization and process data are ML-hard in practice.
2. Classify typical data modalities (micrographs, spectra, logs, maps) and pitfalls.
3. Use the PSPP graph (processing→structure→property→performance) as modeling scaffold.
4. Define a baseline ML pipeline with proper splits and metrics for experimental data.
5. Recognize where physical priors are necessary to avoid spurious learning.

## Book-aligned content mapping
1. Neuer (2024): explainable/physics-aware engineering ML framing.
2. Sandfeld: materials-data workflow and domain examples.
3. McClarren (2021): physical systems constraints + model trust.
4. Murphy/Bishop: risk, likelihood, and validation language.

## 90-minute structure
- 0–10 min: Course mission and relation to MFML/MG
- 10–30 min: Data modalities in characterization and processing
- 30–45 min: Why naive ML fails (noise, artifacts, drift, leakage)
- 45–65 min: PSPP graph and supervised task definitions
- 65–80 min: Baseline pipeline template for lab data
- 80–88 min: Failure case post-mortem (realistic)
- 88–90 min: Summary + exercise handoff

## Exercise (90 min)
- Build a simple baseline on a small materials dataset
- Compare random split vs grouped split to reveal leakage
- Compute confusion matrix / MAE and explain failure causes
- Add one physics-aware feature and evaluate impact

## Assessment alignment
- Written exam emphasizes modeling judgement, error analysis, and trust.
