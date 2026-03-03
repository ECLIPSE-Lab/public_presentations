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
