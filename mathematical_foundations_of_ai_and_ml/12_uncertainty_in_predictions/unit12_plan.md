# Unit 12 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: Units 1-11 completed (probabilistic view, representation learning, unsupervised learning)
- Assumed: Bayesian update rule, posterior/likelihood/prior, loss minimization, neural network training
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 12)
By the end of the unit, students can:
1. Explain why single-point predictions are insufficient in engineering and distinguish aleatory from epistemic uncertainty formally.
2. Derive the Bayesian predictive distribution by marginalizing over parameter posteriors and interpret the resulting variance decomposition.
3. Describe the evidence framework, marginal likelihood, and the concept of effective number of parameters.
4. Define a Gaussian Process via its mean and kernel function, derive the GP posterior in closed form, and interpret uncertainty bands.
5. Explain how Mixture-Density Networks, MC Dropout, and stochastic enrichment provide practical uncertainty estimates.

## Book-aligned content mapping (priority order)
1. Murphy (2012): Ch. 5 (Bayesian statistics, marginal likelihood, model selection, empirical Bayes), Ch. 15 (GP definition, GP regression, kernel parameters, marginal likelihood for GPs).
2. Neuer (2024): Ch. 6.4 (stochastic enrichment, uncertainty quantification from learned results, MDNs, process corridors).
3. Bishop (2006): Bayesian predictive distribution for regression, evidence framework, effective number of parameters (Ch. 3.5).
4. Rasmussen & Williams (2006): GP prior/posterior, kernel design, uncertainty calibration (reference text for GP details).

## 90-minute structure
- 0-10 min: Why UQ matters in engineering (motivating failures, safety-critical decisions)
- 10-25 min: Bayesian predictive distribution and aleatory vs epistemic decomposition
- 25-40 min: Evidence framework, marginal likelihood, effective number of parameters
- 40-60 min: Gaussian Processes (kernel, prediction, uncertainty bands, marginal likelihood)
- 60-75 min: Practical UQ architectures (MDN, MC Dropout, stochastic enrichment)
- 75-85 min: Physics-informed constraints for uncertainty reduction
- 85-90 min: Summary + exercise handoff

## Exercise (90 min)
- Implement GP regression from scratch (RBF kernel, posterior mean and variance)
- Compare GP uncertainty bands with NN ensemble predictions on a 1D regression task
- Vary kernel hyperparameters and observe effect on uncertainty
- Bonus: implement a simple MDN with 2 Gaussian components in PyTorch

## Assessment alignment
- Written exam prep: derivation of GP posterior, variance decomposition, marginal likelihood formula.
- Every unit introduces 3-5 exam-relevant "must-know" statements.

## Slide production notes
- Style: follow data_science_for_em revealjs pattern
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
