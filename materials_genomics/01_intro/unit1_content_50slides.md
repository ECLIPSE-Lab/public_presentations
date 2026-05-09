# Materials Genomics Unit 1 — 50-Slide Content Pack (English)

## Unit theme
**Materials data as a design space: from datasets to discovery loops**

## Book-backed content summary (for this unit)
- Materials genomics treats composition and crystal structure as a searchable design space rather than a fixed list of known compounds.
- Databases and simulations provide candidate structures, target properties, and method metadata that together define the learning problem.
- Model choice in scientific discovery must remain tied to domain knowledge, uncertainty, and explainability rather than pure benchmark accuracy.
- Materials discovery tasks combine regression, ranking, classification, and screening logic within one validation-aware workflow.
- Provenance, dataset bias, and leakage determine whether a discovery claim is scientifically defensible.

## Source anchors used
- Neuer 1.1-1.3
- Sandfeld 2.1-2.3
- McClarren Ch1
- Murphy Ch1
- Bishop Ch1

---

## 50-slide scaffold

### Block A — Why materials genomics? (Slides 1–8)
1. **Title + course role in program**
2. **What “genomics” means (and what it does NOT mean)**
3. **Learning outcomes Unit 1**
4. **Discovery bottleneck in classical materials workflows**
5. **Data-rich turn in materials science**
6. **Where this course connects to MFML and ML-PC**
7. **90-minute roadmap**
8. **Checkpoint prompt: “Where does ML add value here?”**

### Block B — Design space framing (Slides 9–17)
9. **Periodic table + structure space as searchable manifold**
10. **PSPP graph for materials discovery**
11. **Targets: formation energy, stability, bandgap, moduli, etc.**
12. **Direct simulation vs surrogate modeling**
13. **Screening logic: rank then validate**
14. **Why uncertainty is required for candidate prioritization**
15. **Domain knowledge constraints**
16. **First-principles + data-driven hybrid strategy**
17. **Micro-case: when pure data fit fails physically**

### Block C — Data assets and representation basics (Slides 18–30)
18. **Database landscape (MP, OQMD, AFLOW, NOMAD)**
19. **What each database gives / misses**
20. **Data object types: composition, structure, process metadata**
21. **Thermodynamic quantities used in ML datasets**
22. **Representation problem statement**
23. **Classical descriptors vs learned representations (preview)**
24. **Symmetry and invariance constraints**
25. **Data quality dimensions**
26. **Metadata and provenance importance**
27. **Dataset shift across generation pipelines**
28. **Bias map: coverage, publication, synthesis bias**
29. **Leakage map in materials datasets**
30. **Recap: data assumptions that must be explicit**

### Block D — Modeling and validity in discovery (Slides 31–41)
31. **Task formulations in MG**
32. **Regression baseline + error interpretation**
33. **Classification/ranking formulations for screening**
34. **Train/val/test under compositional grouping**
35. **OOD behavior in chemical space**
36. **Uncertainty-aware ranking concept**
37. **Exploration vs exploitation (concept only)**
38. **Decision rule with uncertainty and cost**
39. **Explainability expectations in scientific ML**
40. **Reproducibility checklist for discovery claims**
41. **Common failure post-mortems**

### Block E — Exercise bridge + integration (Slides 42–50)
42. **Exercise objective and dataset**
43. **Step 1: query and clean materials table**
44. **Step 2: feature table v1 (simple descriptors)**
45. **Step 3: baseline model + grouped split**
46. **Step 4: error and bias diagnosis**
47. **Step 5: one uncertainty proxy and discussion**
48. **What to report in notebook (scientific style)**
49. **Unit summary: 10 exam-relevant statements**
50. **References + reading for Unit 2**

---

## Book-derived content snippets to reuse directly in slides
- Data science in materials is interdisciplinary and explicitly tied to **domain knowledge** (Sandfeld Ch. 2.1).
- ML is a method family within a broader AI/data-science ecosystem; avoid buzzword conflation (Sandfeld Ch. 2.1; McClarren Ch. 1).
- Model trust requires explainability and uncertainty framing; “black-box by default” is not acceptable for scientific discovery (Neuer Ch. 1.1.2–1.1.3).

## Lecture vs exercise split
- **Lecture**: design-space thinking, dataset caveats, validity criteria, uncertainty-aware decision logic.
- **Exercise**: practical query/build/evaluate loop with one explicit bias diagnosis.

## Reading assignment after Unit 1
- Sandfeld Ch. 2.1–2.3
- Neuer Ch. 1.1.2–1.1.3
- McClarren Ch. 1.1 + 1.5
