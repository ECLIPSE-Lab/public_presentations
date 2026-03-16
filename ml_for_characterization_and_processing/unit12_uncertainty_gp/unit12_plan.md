# Unit 12 Plan: Uncertainty-Aware Regression & Gaussian Processes

## Metadata
- **Course:** Machine Learning for Characterization and Processing (ML-PC)
- **Unit:** 12
- **Topic:** Uncertainty-Aware Regression & Gaussian Processes
- **Duration:** 90 Minutes
- **Key References:** 
    - Neuer (2024), Ch 6.4
    - Bishop (2006), Ch 3.3, 3.5
    - MFML Unit 12 (Foundations)

## Learning Objectives
- Distinguish between aleatoric and epistemic uncertainty in engineering data.
- Explain the Bayesian approach to regression and its advantages over point estimates.
- Understand the mechanics of Gaussian Processes (GPs) as non-parametric surrogate models.
- Select appropriate kernels for different materials science problems.
- Implement basic uncertainty quantification using MC Dropout or Ensembles.

## Lecture Structure (90 Minutes)

### 1. Introduction: Trust in ML Decisions (15m)
- The cost of being wrong in materials processing (waste, safety, equipment damage).
- Why MSE is not enough: Point estimates hide risk.
- Trust as a function of predictive variance.

### 2. Taxonomy of Uncertainty (15m)
- **Aleatoric Uncertainty:** Measurement noise, stochastic processes (e.g., grain growth fluctuations).
- **Epistemic Uncertainty:** Model uncertainty, lack of data in certain process windows.
- Visualization: How uncertainty "balloons" in data-poor regions.

### 3. Bayesian Linear Regression (20m)
- Refresher on Bayes' Rule for parameters.
- Predictive distribution: The math behind "error bars."
- The role of the prior: Incorporating expert knowledge.

### 4. Gaussian Processes (GPs) (25m)
- From weights to functions: The "Function Space" view.
- Kernels (Covariance functions): RBF, Periodic, Matern.
- GP Regression equations (Mean and Variance).
- Case Study: GPs for property prediction in alloy design.

### 5. Uncertainty in Deep Learning (10m)
- Scaling up: When GPs are too slow ($O(N^3)$).
- MC Dropout: Turning on dropout during testing to sample the model distribution.
- Deep Ensembles: Training multiple models to see where they disagree.

### 6. Summary and Conclusion (5m)

## 50-Slide Strategy
- Slides 1-5: Intro & The Cost of Uncertainty
- Slides 6-12: Aleatoric vs Epistemic (Definitions & Examples)
- Slides 13-20: Bayesian Regression (Prior to Posterior)
- Slides 21-35: Gaussian Processes (Kernels, Math, Surrogates)
- Slides 36-42: Uncertainty in Deep Learning (MC Dropout, Ensembles)
- Slides 43-48: Case Studies (Alloy Design, Process Monitoring)
- Slides 49-50: Summary & Outlook
