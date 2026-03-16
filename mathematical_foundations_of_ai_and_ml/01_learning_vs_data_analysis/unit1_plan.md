# Unit 1 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: 2 semesters math, 1 physics, some chemistry
- Assumed: basic undergraduate math, SVD familiarity, very basic Python
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 1)
By the end of the unit, students can:
1. Distinguish data analysis, fitting, and machine learning in a scientifically precise way.
2. Write supervised learning problems as optimization of expected risk.
3. Explain model/loss/regularization/generalization in one coherent workflow.
4. Map materials examples to ML task types (regression/classification/representation).
5. Identify what belongs in lecture vs exercise for long-term mastery.

## Book-aligned content mapping (priority order)
1. Neuer (2024): concept of model, data-based modeling, criticism/limits; uncertainty framing.
2. Sandfeld: materials-context framing (where ML enters materials workflows).
3. McClarren (2021): physical systems perspective for model choice and error.
4. Murphy (2012): probabilistic language for supervised learning and risk.
5. Bishop (2006): canonical framing of pattern recognition and model complexity.

## 90-minute structure
- 0–10 min: Why this course exists (for KI-Materialtechnologie)
- 10–25 min: ML as approximation under uncertainty (modeling viewpoint)
- 25–45 min: Formal core: dataset, hypothesis class, loss, empirical risk
- 45–60 min: Regularization, train/val/test, leakage and false confidence
- 60–75 min: Materials case mini-studies (property prediction, defect classification, process modeling)
- 75–85 min: Common failure modes + scientific trust checklist
- 85–90 min: Summary + exercise handoff

## Exercise (90 min)
- Implement linear regression in NumPy from scratch (MSE + gradient descent)
- Compare underfit / overfit with polynomial basis
- Split data into train/val/test correctly and discuss leakage trap
- Bonus: add L2 regularization and observe validation behavior

## Assessment alignment
- Written exam prep starts in Unit 1 via explicit definitions and notation.
- Every unit introduces 3–5 exam-relevant “must-know” statements.

## Slide production notes
- Style: follow data_science_for_em revealjs pattern
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
