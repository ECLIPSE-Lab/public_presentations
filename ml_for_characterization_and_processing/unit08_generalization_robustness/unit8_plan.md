# Unit 8: Generalization, Robustness, and Process Windows - Lecture Plan

## Objective
The objective of this unit is to teach students how to rigorously evaluate ML models for real-world engineering applications. We move beyond simple accuracy metrics to discuss how models behave under uncertainty, how to optimize them via hyperparameters, and how to define "safe" operational process windows using ML predictions.

## Slide Structure (approx. 50 slides)

### Part 1: Generalization and the Bias-Variance Tradeoff (10 slides)
- **Generalization Error (3 slides):**
  - Training vs. Testing error (Sandfeld 12.2).
  - The goal: Minimizing error on *unseen* data.
- **Overfitting vs. Underfitting (4 slides):**
  - Underfitting: High Bias (Model too simple).
  - Overfitting: High Variance (Model too complex).
  - Visualizing the "U-curve" of generalization error.
- **Parsimony Revisited (3 slides):**
  - Occam's Razor in model selection (Unit 1/4 refresher).
  - Why simpler models are often more robust.

### Part 2: Robustness and Noise (10 slides)
- **Types of Robustness (3 slides):**
  - Robustness to Measurement Noise (Aleatory uncertainty).
  - Robustness to Distribution Shift (Detector drift, different batches).
- **Outliers and Data Cleaning (3 slides):**
  - How single bad data points can "skew" a non-robust model (e.g., LSQ vs. MAE).
- **Adversarial Examples in Materials? (4 slides):**
  - Small perturbations in microstructure images that lead to wrong property predictions.
  - Making CNNs robust via augmentation (Ref. Unit 6).

### Part 3: Cross-Validation and Hyperparameter Tuning (12 slides)
- **Validation Strategies (4 slides):**
  - K-Fold Cross-Validation (Sandfeld 16.2).
  - Stratified and Group-based splits (Ref. Unit 3).
- **Hyperparameters vs. Parameters (3 slides):**
  - Parameters (Weights): Learned via Gradient Descent.
  - Hyperparameters (LR, Layers, Filters): Set by the designer.
- **Optimization Algorithms (5 slides):**
  - Grid Search: The brute force approach.
  - Random Search: Why it's often better than grid search.
  - Bayesian Optimization: Intelligently searching the hyperparameter space.

### Part 4: Defining Process Windows (10 slides)
- **What is a Process Window? (3 slides):**
  - The region in $(P, v, T)$ space where properties are acceptable.
- **Mapping Predictions to Windows (4 slides):**
  - Using ML classifiers to draw the "Operating Boundary."
  - Probability contours: How certain are we about the window edges?
- **Case Study: AM Process Maps (3 slides):**
  - Laser Power vs. Scanning Speed.
  - Defining the "Lack-of-Fusion" vs. "Keyhole" boundaries via ML.

### Part 5: Sensitivity Analysis (8 slides)
- **Feature Importance (4 slides):**
  - Which inputs drive the prediction?
  - Permutation Importance vs. SHAP values (Teaser for Unit 14).
- **Stability Analysis (4 slides):**
  - Does a 1% change in input lead to a 50% change in output?
  - Ensuring the model follows physical continuity.

## Materials Sources
- **Sandfeld (2024):** Ch 12 (Generalization), Ch 16.2 (CV/Tuning).
- **Neuer (2024):** Ch 4 (Supervised learning evaluation).
- **McClarren (2021):** Ch 1 (Overview of robustness).
