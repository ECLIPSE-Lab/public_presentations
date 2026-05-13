# Unit 13 Plan: Physics-Informed and Constrained ML

## Metadata
- **Course:** Machine Learning for Characterization and Processing (ML-PC)
- **Unit:** 13
- **Topic:** Physics-Informed and Constrained ML
- **Duration:** 90 Minutes
- **Key References:** 
    - Sandfeld (2024), Ch 19.6
    - Neuer (2024), Ch 6
    - MFML Unit 13 (Foundations)

## Learning Objectives
- Understand the limitations of purely data-driven models in physical sciences.
- Differentiate between observational, learning, and structural bias.
- Explain the mechanism of Physics-Informed Neural Networks (PINNs).
- Implement basic physical constraints in loss functions using Automatic Differentiation.
- Recognize the potential of Operator Learning (DeepONet) for accelerating simulations.

## Lecture Structure (90 Minutes)

### 1. Introduction: The Limits of Data-Driven ML (15m)
- Why "Black-Box" ML fails in materials science (small data, extrapolation, physical inconsistency).
- The concept of "Scientific Machine Learning" (SciML).
- Integration of domain knowledge: The Bias-Variance-Physics tradeoff.

### 2. Taxonomy of Physics Integration (15m)
- **Observational Bias:** Data enrichment via simulations and data augmentation (symmetries, units).
- **Learning Bias:** Penalty-based methods, modifying the loss function with PDE residuals.
- **Structural Bias:** Designing architectures that satisfy constraints by design (e.g., monotonicity, conservations).

### 3. Physics-Informed Neural Networks (PINNs) (25m)
- The fundamental idea: Networks as mesh-free PDE solvers.
- The Role of Automatic Differentiation (AD): `tf.GradientTape` and `torch.autograd`.
- Case Study: 1D Heat Equation / Diffusion in materials.
- Loss function formulation: $J = J_{data} + \lambda_{pde} J_{pde} + \lambda_{bc} J_{bc} + \lambda_{ic} J_{ic}$.

### 4. Advanced Constraints and Inverse Problems (20m)
- **Inverse Problems:** Discovering material parameters (e.g., diffusivity, elasticity) from noisy experimental data.
- **Constraining Outputs:** Lagaris substitution for hard boundary conditions.
- **Monotonicity and Convexity:** Ensuring physical plausibility in constitutive laws.

### 5. Future Frontiers: Operator Learning (10m)
- From functions to operators: DeepONet.
- Accelerating multi-scale simulations in processing.
- Summary and integration with previous units.

### 6. Summary and Conclusion (5m)

## 50-Slide Strategy
- Slides 1-5: Intro & Motivation
- Slides 6-12: Taxonomy of Bias
- Slides 13-20: Data Enrichment & Observational Bias
- Slides 21-35: PINNs (Theory, AD, Implementation)
- Slides 36-42: Hard Constraints & Lagaris Method
- Slides 43-48: Inverse Problems & Case Studies
- Slides 49-50: Summary & Outlook
