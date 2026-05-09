# Unit 1 Plan — Machine Learning for Characterization and Processing

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: 2 semesters math, 1 physics, some chemistry
- Assumed: undergrad math, SVD familiarity, very basic Python
- Lecture: 90 minutes + 90-minute exercise
- Language: English

## Learning objectives (Unit 1)
By the end of this unit, students can:
1. Explain the historical transition from descriptive to data-driven explanatory models in science.
2. Define Materials Data Science at the intersection of ML, statistics, and domain knowledge.
3. Identify unique challenges of materials data: small data, high cost, noise, multi-modality, and the "Curse of Dimensionality".
4. Classify models into White-Box, Grey-Box, and Black-Box categories based on traceability.
5. Apply the CRISP-DM process to structure an experimental materials data project.
6. Differentiate between correlation and causality in materials systems (using the PSPP paradigm).

## Book-aligned content mapping
1. **Sandfeld (2024)**: History of data (Sumerians to Kepler), definition of Materials Data Science, Curse of Dimensionality, feature engineering (Ashby maps).
2. **Neuer (2024)**: Model categories (White/Grey/Black), CRISP-DM for engineering, data scales (Nominal to Ratio), Correlation vs. Causality.
3. **McClarren (2021)**: Supervised/Unsupervised task types, Loss functions (MSE, Cross-entropy), Parsimony/Occam's Razor, Overfitting, R^2.

## 90-minute structure
- **0–15 min: The Long History of Data**
  - From Sumerian cuneiform (metadata) to Kepler (the first data analyst).
  - Transition from physics-only to data-driven discovery.
- **15–30 min: What is Materials Data Science?**
  - Definition and the role of Domain Knowledge.
  - Examples: Ashby maps as early "feature engineering".
- **30–50 min: Challenges: Why Materials ML is Hard**
  - Small data/High cost, Measurement noise, Multi-modality.
  - The Curse of Dimensionality (parameter studies).
- **50–70 min: Modeling Frameworks**
  - Supervised vs. Unsupervised (Toy box analogy).
  - White-Box vs. Black-Box models.
  - Loss functions and Parsimony (Occam's Razor).
- **70–85 min: The Workflow: CRISP-DM adapted for Labs**
  - Business/Scientific Understanding to Deployment.
  - Correlation vs. Causality (Ice cream & Crime).
- **85–90 min: Summary + exercise handoff**

## Exercise (90 min)
- **Data Exploration**: Inspect real materials datasets (e.g., MDS-1 tensile test).
- **Scaling & Normalization**: Practice transforming between Nominal/Ordinal/Ratio scales.
- **Baseline Modeling**: Build a simple regressor, evaluate R^2, and discuss overfitting vs. parsimony.
- **Causality Check**: Analyze the PSPP graph for a given dataset and identify potential spurious correlations.

## Assessment alignment
- Exam questions on: Model classification (White/Grey/Black), Data scales, CRISP-DM phases, and identifying data leakage/overfitting in materials contexts.
