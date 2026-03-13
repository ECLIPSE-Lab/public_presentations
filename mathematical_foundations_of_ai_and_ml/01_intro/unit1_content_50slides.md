# MFML Unit 1 — 50-Slide Content Pack (English)

## Unit theme
**What learning means in engineering and materials**

## Core source mapping (book-priority aligned)
- **Neuer (2024)**: Ch. 1.1.1–1.1.3 (model concept; white/grey/black box; criticism), Ch. 1.3 (data-mining process / CRISP-DM), uncertainty framing.
- **Sandfeld (2024)**: Ch. 2.1 (data science vs ML vs AI; domain knowledge), Ch. 2.2–2.3 (materials context, data→information→knowledge).
- **McClarren (2021)**: Ch. 1 (landscape of ML; regression/classification/time-series; Bayesian probability; cross-validation).
- **Murphy (2012)**: Ch. 1 (supervised vs unsupervised; overfitting/model selection; curse of dimensionality).
- **Bishop (2006)**: Ch. 1 (probability-based model view; model selection; decision view).

---

## Slide-by-slide content (target: 50)

### Block A — Orientation and motivation (Slides 1–8)
1. **Title + positioning in SS26 lecture triad**
   - MFML as formal backbone for MG + ML-PC.
2. **Why this course now?**
   - From tool usage to principled modeling.
3. **Learning outcomes for Unit 1**
   - Definitions, risk minimization, validity, transfer to materials tasks.
4. **What students already know**
   - Calculus, LA/SVD, basic coding → now reframed as learning pipeline.
5. **What students often confuse**
   - AI, ML, DL, statistics, simulation.
6. **Quick map: AI vs ML vs DL vs Data Science**
   - ML as method family; not equal to AI.
7. **Domain knowledge matters**
   - Engineering constraints are not optional add-ons.
8. **Roadmap of today’s 90 min**
   - Conceptual arc + exercise handoff.

### Block B — Model concept and epistemology (Slides 9–17)
9. **What is a model? (Neuer 1.1.1)**
   - Abstraction for prediction + understanding.
10. **First-principles model example**
    - Gravity law as interpretable mapping.
11. **Data-based modeling (top-down)**
    - Dependency assumed to be contained in data.
12. **When first-principles is insufficient**
    - High-dimensional coupled technical processes.
13. **White-box / grey-box / black-box**
    - Precise definitions + engineering implications.
14. **Why black-box criticism appears**
    - Traceability, trust, safety.
15. **Explainability as spectrum, not binary**
    - Model probing and partial interpretability.
16. **Hybrid modeling mindset**
    - Physics core + data-driven correction.
17. **Mini-checkpoint question**
    - “Is linear regression always white-box?”

### Block C — Formal supervised learning core (Slides 18–30)
18. **Data notation and task notation**
    - Dataset, features, target, hypothesis class.
19. **Empirical risk minimization (ERM)**
    - \\(\hat\theta=\arg\min_\theta \frac1N\sum_i \ell(f_\theta(x_i),y_i)\\)
20. **Regularized objective**
    - \\(+\lambda\Omega(\theta)\\) and complexity control.
21. **Population risk vs empirical risk**
    - Why train error is not the goal.
22. **Regression vs classification vs ranking**
    - Output-space semantics.
23. **Loss functions and meaning**
    - MSE, MAE, cross-entropy in practical terms.
24. **Optimization lens**
    - Learning as solving constrained optimization.
25. **Bayesian lens (intro)**
    - Prior, likelihood, posterior (qualitative).
26. **Frequentist vs Bayesian workflow (practical)**
    - What changes in engineering decisions.
27. **Decision layer separate from inference**
    - Predictive probability vs action rule.
28. **No-free-lunch intuition**
    - Inductive bias always present.
29. **Curse of dimensionality (conceptual)**
    - Why data hunger explodes with dimension.
30. **Recap slide: 6 equations/ideas to remember**

### Block D — Validation, generalization, trust (Slides 31–40)
31. **Overfitting explained visually**
    - Training fit vs deployment failure.
32. **Bias–variance intuition**
    - Under/over capacity tradeoff.
33. **Train/val/test splits**
    - Roles and anti-patterns.
34. **Cross-validation**
    - When and why k-fold helps.
35. **Data leakage taxonomy**
    - Temporal leakage, group leakage, preprocessing leakage.
36. **Metrics linked to decisions**
    - MAE vs RMSE vs calibration vs cost-sensitive metrics.
37. **Uncertainty types (Neuer / UQ framing)**
    - Aleatoric vs epistemic (engineering interpretation).
38. **Model confidence vs correctness**
    - Calibration concept.
39. **Trust checklist for engineering ML**
    - Data provenance, assumptions, uncertainty, fail modes.
40. **Checkpoint MCQ slide**
    - Concept quiz for exam-style reasoning.

### Block E — Materials translation + exercise bridge (Slides 41–50)
41. **Materials example 1: process→property regression**
42. **Materials example 2: defect classification from images**
43. **Materials example 3: spectra interpretation task framing**
44. **How MFML links to Materials Genomics**
45. **How MFML links to ML-PC**
46. **Lecture-essential vs exercise content split**
47. **Exercise setup: NumPy linear regression from scratch**
48. **Exercise extension: regularization + split stress test**
49. **Exam-aligned summary: 10 must-know statements**
50. **References + reading assignment for next unit**

---

## Reusable equations (for slide math boxes)
- Empirical risk: \\(R_N(\theta)=\frac1N\sum_i \ell(f_\theta(x_i),y_i)\\)
- Regularized ERM: \\(\hat\theta=\arg\min_\theta R_N(\theta)+\lambda\Omega(\theta)\\)
- Bayesian update: \\(p(\theta|\mathcal D)\propto p(\mathcal D|\theta)p(\theta)\\)
- Expected risk: \\(R(\theta)=\mathbb E_{(x,y)\sim p_{data}}[\ell(f_\theta(x),y)]\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: definitions, conceptual distinctions, objective functions, validation logic, trust framework.
- **Exercise**: implementation, debugging, sensitivity analysis, leakage experiments.

## Reading assignment after Unit 1
- Neuer Ch. 1.1 + 1.3
- McClarren Ch. 1.1 + 1.5
- Murphy Ch. 1.1–1.4 (skim)
- Bishop Ch. 1.1 + 1.3 (conceptual)
