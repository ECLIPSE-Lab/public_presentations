# ML-PC Unit 1 — 50-Slide Content Pack (English)

## Unit theme
**What makes materials data special? From History to Modern Workflow**

## Core source mapping (book-priority aligned)
- **Sandfeld (2024)**: Historical roots (Sumerians to Kepler), Definition of Materials Data Science, Curse of Dimensionality, Feature engineering (Ashby maps).
- **Neuer (2024)**: Model categories (White/Grey/Black), CRISP-DM for engineering, data scales (Nominal to Ratio), Correlation vs. Causality.
- **McClarren (2021)**: Ch. 1 (supervised/unsupervised task types, Loss functions, Parsimony/Occam's Razor, Overfitting, R^2, Bayesian probability).

---

## Slide-by-slide content (target: 50)

### Block A — Introduction and History (Slides 1–10)
1. **Title + course positioning**: "Machine Learning in Materials Processing & Characterization"
2. **Learning outcomes**: What will students know by the end of Unit 1?
3. **The "Hype Cycle" vs. Lab Reality**: Moving from "textbook AI" to "lab-ready AI".
4. **Historical roots of data science (I)**: Sumerians (3400 BCE), Cuneiform, and the first "metadata" (symbols for sheep/units).
5. **Historical roots of data science (II)**: Ancient Astronomy. From gods to crystal spheres.
6. **Kepler: The First Data Analyst**: Analyzing 25 years of Mars data to find simple laws. Transition to data-driven discovery.
7. **J. Tobias Mayer**: Moon libration study (1750). The first "modern" use of overdetermined data systems.
8. **Defining Materials Data Science**: The interface of ML, statistics, computer science, and Domain Knowledge.
9. **Domain Knowledge is Key**: Why "off-the-shelf" ML often fails in materials.
10. **The Ashby Map**: A classic example of "feature engineering" without explicit ML.

### Block B — Modeling Frameworks (Slides 11–22)
11. **White-Box vs. Black-Box Models**: Traceability and trust in engineering.
12. **White-Box (First-Principle)**: Bottom-up, physical axioms (Navier-Stokes, Schrödinger).
13. **Black-Box (Data-driven)**: Top-down, extracted from observation (standard neural networks).
14. **Grey-Box (Hybrid)**: The middle ground (Monte Carlo, Physics-Informed ML).
15. **Supervised vs. Unsupervised Learning**: The "Toy Box" analogy.
16. **Supervised Tasks**: Labels as "answers". Regression (numerical) and Classification (categorical).
17. **Regression Goals**: Mapping independent (X) to dependent (Y) variables.
18. **Classification Goals**: Mapping to finite sets (groups/classes).
19. **Unsupervised Tasks**: Finding structure in unlabeled data. Clustering, Embedding, Association rules.
20. **Loss Functions**: How we "score" our model (MSE, Cross-entropy).
21. **Parsimony and Occam's Razor**: Why the simplest explanation is best.
22. **Overfitting**: When accuracy on training data is a trap.

### Block C — What makes materials data special? (Slides 23–34)
23. **The PSPP Paradigm**: Processing–Structure–Property–Performance as a data graph.
24. **Multi-scale Challenge**: From atoms to airplane wings (10 orders of magnitude).
25. **Multi-modal Challenge**: Fusing micrographs, spectra, and process logs.
26. **High Acquisition Cost**: Why materials data is "Small Data".
27. **Measurement Noise and Uncertainty**: Aleatoric vs. Epistemic uncertainty.
28. **The Curse of Dimensionality (I)**: Parameter study case. Why 35 experiments are not enough for 3 parameters.
29. **The Curse of Dimensionality (II)**: Sparse data in high-dimensional space.
30. **Data Scales (I)**: Nominal (names) and Ordinal (ordered names).
31. **Data Scales (II)**: Cardinal/Interval (Celsius) and Ratio (Kelvin/Absolute zero).
32. **Units and Metadata**: Why a number is worthless without its unit (Sandfeld Sumerian connection).
33. **Metadata in Labs**: Recording resolution, drift, and calibration history.
34. **The "Failure Story"**: Learning the serial number instead of the physics (Data Leakage).

### Block D — The Workflow: CRISP-DM for Materials (Slides 35–45)
35. **The CRISP-DM Standard**: Adapting industrial standards for lab discovery.
36. **Phase 1: Business (Scientific) Understanding**: Defining the ROI (Return on Insight).
37. **Phase 2: Data Understanding**: Visualizing raw data, checking for artifacts.
38. **Phase 3: Data Preprocessing**: Cleaning, Scaling, Normalization.
39. **Phase 4: Modeling**: Selecting and training the algorithm.
40. **Phase 5: Evaluation**: Does it generalize? (The R^2 metric and its pitfalls).
41. **Phase 6: Deployment**: Integrating ML into the microscope or furnace control.
42. **Phase 7: Monitoring**: Detecting model drift and out-of-distribution cases.
43. **Correlation vs. Causality (I)**: The "Ice Cream and Crime" case study.
44. **Correlation vs. Causality (II)**: Materials example (Hardness vs. Cooling rate vs. Microstructure).
45. **Scientific Trust**: Why explainability matters for peer review.

### Block E — Exercise bridge + execution plan (Slides 46–50)
46. **Exercise Objective**: Identifying failures and scales in materials data.
47. **Preview: MDS-1 Tensile Test**: A dataset for regression with parameter uncertainty.
48. **Checklist for trustworthy materials ML**: Baseline-before-complexity, Leakage check, Unit consistency.
49. **Unit Summary**: Top 10 takeaways.
50. **References + Reading assignments**: McClarren Ch 1, Neuer Ch 1, Sandfeld Ch 1, 2, 4.

---

## Book-derived points included verbatim-style
- "Materials Data Science is the application of data science approaches to materials science problems, combined with domain knowledge" (Sandfeld).
- "Models are imitations of reality at a lower level of detail" (Neuer).
- "Overfitting: trading accuracy on training data for failure on new data" (McClarren).
- "Data is symbols of objects, events, and their environment. Information is data with context" (Sandfeld).
