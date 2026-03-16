# MFML Unit 14 — 50-Slide Content Pack (English)

## Unit theme
**Explainability, Limits, and Trust — Course Synthesis**

## Core source mapping (book-priority aligned)
- **Neuer (2024)**: Ch. 7 (explainability, semantic structures, sensitivity analysis, levels of explainability, causality).
- **McClarren (2021)**: Ch. 12 (limitations and outlook).
- **Bishop (2006)**: Ch. 1 (model limits, reflection).

---

## Slide-by-slide content (target: 50)

### Block A — The trust mandate (Slides 1-10)
1. **Title + Unit 14 positioning**
   - The final lecture. From physics-informed learning (Unit 13) to the question: can we trust our models?
2. **Learning outcomes for Unit 14**
   - Explainability mandate, semantic structures, sensitivity analysis, causality, data manifold limits, course synthesis.
3. **Why explainability is non-negotiable**
   - Science demands understanding, not just prediction. Industry demands accountability. Regulations demand transparency.
4. **The black-box problem**
   - Deep NNs achieve high accuracy but offer no explanation for individual predictions. This is unacceptable for critical decisions.
5. **Explainability vs interpretability**
   - Interpretability: the model itself is understandable (e.g., linear regression). Explainability: post-hoc methods reveal reasoning of complex models.
6. **Who needs explanations?**
   - Scientists, engineers, regulators, customers, operators — each needs explanations at different levels.
7. **The cost of unexplainability**
   - Rejected by regulators, distrusted by engineers, impossible to debug, liability risk.
8. **Explainability as scientific method**
   - A model that cannot be questioned cannot be falsified — it fails the scientific standard.
9. **Course context**
   - Every unit has built toward this: loss minimization, generalization, uncertainty, physics — all serve trustworthy ML.
10. **Roadmap of today's 90 min**
    - Semantic structures → levels of explainability → sensitivity analysis → causality → data limits → course retrospective.

### Block B — Semantic structures for meaning (Slides 11-18)
11. **Digitizing meaning: the challenge**
    - ML models operate on numbers. Domain knowledge is encoded in language and relationships.
12. **Synonyms and controlled vocabularies**
    - Map different terms for the same concept to a canonical form. "Yield strength" = "elastic limit" = "Re".
13. **Taxonomies: hierarchical classification**
    - Organize concepts in parent-child hierarchies: Material > Metal > Steel > Stainless Steel > 316L.
14. **Ontologies: structured knowledge graphs**
    - Concepts + relationships + constraints: "alloy hasProperty tensileStrength", "tensileStrength measuredIn MPa".
15. **Why ontologies matter for ML**
    - Enable deductive reasoning: if the model violates a known relationship, flag it.
16. **Ontologies for feature engineering**
    - Ontological relationships suggest which features to include and how to transform them.
17. **Materials ontology example**
    - Composition → processing → microstructure → properties. Known causal chain constrains model design.
18. **Checkpoint: semantic structures**
    - "Your model uses 'hardness' and 'HRC' as separate features. What semantic issue exists?" — they're the same concept (synonym).

### Block C — Levels of explainability (Slides 19-26)
19. **The six levels of explainability (E1–E6)**
    - A structured framework for matching explanation depth to audience and purpose.
20. **E1: Data level**
    - "What data was used?" Data provenance, quality, completeness, representativeness.
21. **E2: Process level**
    - "What physical process does this relate to?" Context in the engineering workflow.
22. **E3: Feature level**
    - "Which input features matter most?" Feature importance, sensitivity, selection rationale.
23. **E4: Model level**
    - "How does the model work?" Architecture description, hyperparameter choices, training protocol.
24. **E5: Prediction level**
    - "Why this specific prediction?" Local explanations for individual predictions (LIME, SHAP, perturbation).
25. **E6: Decision level**
    - "What action should be taken?" Mapping predictions to actionable recommendations with confidence.
26. **Matching level to audience**
    - Operator: E2+E6. Data scientist: E3+E4. Regulator: E1+E5. Manager: E6. Scientist: all levels.

### Block D — Sensitivity analysis and causality (Slides 27-38)
27. **Perturbation-based sensitivity analysis**
    - Perturb one input feature by Delta; observe change in output. Sensitivity = |Delta_output / Delta_input|.
28. **Global vs local sensitivity**
    - Global: average sensitivity across the dataset. Local: sensitivity at a specific input point.
29. **Sensitivity analysis in practice**
    - Vary each feature ±10% (or ±1 sigma) while holding others constant; rank features by output sensitivity.
30. **Feature importance from sensitivity**
    - High sensitivity → important feature (changes in it strongly affect predictions).
31. **Sensitivity analysis: limitations**
    - Assumes features are independent. Interactions between features are missed by one-at-a-time perturbation.
32. **Beyond perturbation: SHAP values (brief)**
    - Shapley values from game theory: allocate prediction contribution to each feature.
33. **Causality vs correlation**
    - ML models find correlations. But correlation does not imply causation. Confounders can create spurious patterns.
34. **Causal process chains (Neuer Ch. 7)**
    - In manufacturing: composition → processing → microstructure → properties. The arrow direction matters.
35. **Detection vs prediction**
    - Detection: "this sample has low hardness" (pattern recognition). Prediction: "changing carbon content will increase hardness" (causal claim).
36. **Where ML adds value in causal chains**
    - ML excels at detection and interpolation. Causal prediction requires domain knowledge or experimental design.
37. **Deductive reasoning with ontologies**
    - If the ontology states "grain size affects yield strength" and the model ignores grain size → flag inconsistency.
38. **Checkpoint: causality**
    - "Your model finds that ice cream sales predict drowning rates. What's the issue?" — confounding variable (temperature).

### Block E — Data manifold limits, trust, and course retrospective (Slides 39-50)
39. **Data manifold limits**
    - ML models are only reliable within the data manifold (training distribution). Extrapolation is dangerous.
40. **Detecting extrapolation**
    - Latent space density (Unit 10), reconstruction error (Unit 9), GP uncertainty (Unit 12) — all flag extrapolation.
41. **Inductive bias and trust**
    - Every model has inductive bias (assumptions). Trust requires understanding what the model assumes and where those assumptions fail.
42. **When models should NOT be trusted**
    - Extrapolation beyond training distribution. Confounded features. Insufficient training data. Missing physics.
43. **Building trustworthy ML systems**
    - Uncertainty quantification + explainability + domain validation + human oversight = trustworthy ML.
44. **Course retrospective: the 14-unit arc**
    - A journey from "what is learning?" to "can we trust what the model learned?"
45. **Units 1-4: Foundations**
    - Learning as optimization (Unit 1), linear algebra tools (Unit 2), loss functions (Unit 3), neural architectures (Unit 4).
46. **Units 5-7: Training and generalization**
    - Backpropagation (Unit 5), optimization landscapes (Unit 6), bias-variance-regularization (Unit 7).
47. **Units 8-11: Probabilistic and unsupervised**
    - Probabilistic foundations (Unit 8), representation learning (Unit 9), latent spaces (Unit 10), clustering (Unit 11).
48. **Units 12-14: Uncertainty, physics, and trust**
    - Uncertainty quantification (Unit 12), physics-informed learning (Unit 13), explainability and trust (Unit 14).
49. **Exam-aligned summary: 10 must-know statements (course-wide)**
    1. Learning = minimizing expected risk; empirical risk as a proxy.
    2. The bias-variance tradeoff governs model complexity selection.
    3. Backpropagation enables efficient gradient computation in O(W).
    4. Regularization restricts hypothesis space to improve generalization.
    5. Bayesian inference provides principled uncertainty quantification.
    6. Autoencoders learn compressed representations; linear AE = PCA.
    7. The EM algorithm finds maximum likelihood parameters for mixture models.
    8. GP uncertainty grows away from data — honest epistemic uncertainty.
    9. PINNs embed physics into the loss to reduce data requirements.
    10. Explainability is a scientific and industrial mandate, not an optional add-on.
50. **Final slide: exam preparation and farewell**
    - Exam covers Units 1-14. Focus on derivations (MLE, backprop, bias-variance, EM, GP posterior).
    - Practice: work through all exercise problems. Understand the "10 must-know statements" per unit.
    - Thank you for an excellent semester.

---

## Reusable equations (for slide math boxes)
- Perturbation sensitivity: \\(S_j = \frac{|f(x + \Delta e_j) - f(x)|}{|\Delta|}\\)
- Shapley value (simplified): \\(\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S \cup \{j\}) - f(S)]\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: explainability framework, semantic structures, sensitivity analysis, causality, data limits, course synthesis.
- **Exercise**: end-to-end mini-project with ontology, sensitivity analysis, explainability report.
