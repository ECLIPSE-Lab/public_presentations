# ML-PC Unit 1 — 50-Slide Content Pack (English)

## Unit theme
**Why characterization and processing data challenge ML — and how to build trustworthy baselines**

## Core source mapping (book-priority aligned)
- **Neuer (2024)**: model categories, explainability, practical process orientation.
- **Sandfeld (2024)**: domain knowledge and workflow coupling in materials data science.
- **McClarren (2021)**: Ch. 1 (supervised task types, Bayesian probability, cross-validation, ML vs simulation).
- **Murphy (2012)**: Ch. 1 (task framing, overfitting, model selection).
- **Bishop (2006)**: Ch. 1 (probabilistic modeling and decision framing).

---

## Slide-by-slide content (target: 50)

### Block A — Context and objectives (Slides 1–8)
1. **Title + course positioning**
2. **Why experimental materials data is uniquely hard**
3. **Unit 1 learning outcomes**
4. **Data modalities in this course**
5. **What students bring from prerequisites**
6. **How this course uses MFML foundations**
7. **90-minute lecture roadmap**
8. **Initial discussion prompt (failure story)**

### Block B — Data modalities and physics of measurement (Slides 9–19)
9. **Micrographs: signal sources + artifacts**
10. **Spectra: resolution/noise/background effects**
11. **Process logs: temporal correlation and drift**
12. **Label quality and annotation variance**
13. **Instrument dependence and domain shift**
14. **Calibration and uncertainty metadata**
15. **Heteroscedastic noise in experiments**
16. **Why IID assumptions often break**
17. **Physics priors as regularization intuition**
18. **Mini-case: same sample, different instrument bias**
19. **Checkpoint question**

### Block C — ML framing for PSPP tasks (Slides 20–31)
20. **PSPP graph recap (processing→structure→property→performance)**
21. **Task mapping onto PSPP edges**
22. **Regression tasks in process modeling**
23. **Classification tasks in defect detection**
24. **Structured outputs (segmentation, spectra decomposition)**
25. **Baseline-before-complexity principle**
26. **Objective functions linked to deployment goals**
27. **Model selection under limited labels**
28. **Cross-validation with grouped/temporal splits**
29. **Leakage examples specific to lab data**
30. **Metrics and cost asymmetry**
31. **Recap of pipeline skeleton**

### Block D — Trustworthy evaluation and uncertainty (Slides 32–43)
32. **Generalization under domain shift**
33. **Overfitting patterns in small lab datasets**
34. **Bias–variance in experimental ML practice**
35. **Aleatoric vs epistemic uncertainty (engineering interpretation)**
36. **Calibration concept for classification**
37. **Uncertainty-aware decision thresholds**
38. **When to reject predictions**
39. **Explainability expectations by stakeholder**
40. **Failure analysis template for lab projects**
41. **Reproducibility checklist (data, code, split, metrics)**
42. **Scientific claim checklist**
43. **Checkpoint MCQ slide**

### Block E — Exercise bridge + execution plan (Slides 44–50)
44. **Exercise objective: baseline on real-ish materials data**
45. **Step 1: clean + split with leakage control**
46. **Step 2: baseline model and metric selection**
47. **Step 3: error analysis by subgroup/source**
48. **Step 4: add one physics-aware feature/constraint**
49. **Unit summary: 10 must-know statements**
50. **References + reading assignment**

---

## Book-derived points to include verbatim-style (adapted)
- ML in engineering should be motivated by **physical-system questions**, not benchmark-only framing (McClarren preface + Ch. 1).
- Data-science success in materials requires **domain knowledge integration**, not isolated algorithm use (Sandfeld Ch. 2.1).
- Explainability and traceability concerns are valid in technical environments and must be addressed explicitly (Neuer Ch. 1.1.2–1.1.3).

## Lecture vs exercise split
- **Lecture**: modality physics, problem framing, evaluation validity, uncertainty and trust.
- **Exercise**: implement baseline, compare split strategies, diagnose failures, add one domain-aware improvement.

## Reading assignment after Unit 1
- McClarren Ch. 1.1–1.5
- Neuer Ch. 1.1 + 1.3 (CRISP-DM perspective)
- Murphy Ch. 1.4 (overfitting/model selection)
