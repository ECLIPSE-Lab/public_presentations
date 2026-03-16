# MFML Unit 12 — 50-Slide Content Pack (English)

## Unit theme
**Uncertainty in Predictions: Bayesian Inference, Gaussian Processes, and Practical UQ**

## Core source mapping (book-priority aligned)
- **Murphy (2012)**: Ch. 5 (Bayesian statistics, marginal likelihood, model selection), Ch. 15 (GP definition, GP regression, kernels).
- **Neuer (2024)**: Ch. 6.4 (stochastic enrichment, MDNs, process corridors, UQ from learned results).
- **Bishop (2006)**: Ch. 3.5 (Bayesian predictive distribution, evidence framework, effective parameters).
- **Rasmussen & Williams (2006)**: GP prior/posterior, kernel design (reference).

---

## Slide-by-slide content (target: 50)

### Block A — Motivation and Bayesian prediction (Slides 1-10)
1. **Title + Unit 12 positioning**
   - From clustering (Unit 11) to quantifying how confident our predictions are.
2. **Learning outcomes for Unit 12**
   - Bayesian predictive distribution, evidence framework, GPs, practical UQ methods.
3. **Why point predictions are not enough**
   - A model predicting 450 MPa tensile strength is useless without knowing if the uncertainty is ±5 or ±100 MPa.
4. **Recall: aleatory vs epistemic uncertainty (Unit 8)**
   - Aleatory: irreducible noise. Epistemic: reducible with more data. Both contribute to prediction uncertainty.
5. **The Bayesian predictive distribution**
   - Integrate over parameter uncertainty: p(y*|x*,D) = integral p(y*|x*,theta) p(theta|D) d theta.
6. **Variance decomposition**
   - Total variance = expected aleatory variance + epistemic variance. Var[y*] = E[sigma^2(theta)] + Var[mu(theta)].
7. **Point estimates vs full distributions**
   - MLE: one theta, one prediction. Bayesian: distribution over theta, distribution over predictions.
8. **When uncertainty matters most**
   - Safety-critical (structural failure), expensive experiments (materials synthesis), extrapolation (new compositions).
9. **Practical UQ: a taxonomy**
   - Exact Bayesian (GPs), approximate Bayesian (MC Dropout, ensembles), direct prediction (MDNs).
10. **Roadmap of today's 90 min**
    - Bayesian prediction → evidence framework → Gaussian Processes → practical UQ methods.

### Block B — Evidence framework and model selection (Slides 11-16)
11. **The marginal likelihood (evidence)**
    - p(D|M) = integral p(D|theta,M) p(theta|M) d theta. Measures how well model M explains the data.
12. **Evidence as automatic Occam's razor**
    - Simple models: high prior mass on few parameters → high evidence if data is simple.
    - Complex models: prior spread thin → lower evidence unless data demands complexity.
13. **Model comparison via evidence**
    - Compare models by their marginal likelihoods: p(M_1|D) / p(M_2|D) = p(D|M_1) / p(D|M_2) (Bayes factor).
14. **Effective number of parameters**
    - gamma = sum lambda_i / (lambda_i + alpha), where lambda_i are eigenvalues of the data precision matrix.
    - gamma << total parameters means the data constrains only a subset of parameters.
15. **Empirical Bayes**
    - Maximize the evidence w.r.t. hyperparameters (prior variance, noise level) instead of fixing them.
16. **Checkpoint: evidence interpretation**
    - "Model A has 100 parameters and log-evidence -500. Model B has 10 parameters and log-evidence -480. Which is preferred?" — B (higher evidence, simpler).

### Block C — Gaussian Processes (Slides 17-32)
17. **What is a Gaussian Process?**
    - A GP is a distribution over functions: f ~ GP(m(x), k(x,x')). Any finite set of function values is jointly Gaussian.
18. **GP as infinite-dimensional Gaussian**
    - Instead of a distribution over parameter vectors, a distribution over entire functions.
19. **Mean function m(x)**
    - The expected function value at each input. Often set to zero (prior mean = 0).
20. **Kernel (covariance) function k(x,x')**
    - Encodes assumptions about function properties: smoothness, periodicity, length scale.
21. **The RBF (squared exponential) kernel**
    - k(x,x') = sigma_f^2 exp(-||x-x'||^2 / (2 l^2)). Length scale l, signal variance sigma_f^2.
22. **Other kernels**
    - Matérn (adjustable smoothness), periodic, linear, composite (sum/product of kernels).
23. **GP prior: sampling functions**
    - Draw function samples from the prior: smooth, random functions with properties set by the kernel.
24. **GP posterior: conditioning on data**
    - After observing D = {(x_i, y_i)}, the posterior is also a GP with updated mean and covariance.
25. **GP posterior: closed-form formulas**
    - mu*(x*) = k*^T (K + sigma_n^2 I)^{-1} y. sigma*^2(x*) = k(x*,x*) - k*^T (K + sigma_n^2 I)^{-1} k*.
26. **GP posterior: interpretation**
    - Mean: best prediction (interpolates data). Variance: uncertainty (small near data, large far away).
27. **GP uncertainty bands**
    - Plot mu(x) ± 2 sigma(x): the 95% credible band. Bands widen away from data — honest uncertainty.
28. **GP hyperparameter learning**
    - Optimize kernel hyperparameters (l, sigma_f, sigma_n) by maximizing the marginal likelihood.
29. **Length scale effect**
    - Short l: wiggly functions, fits noise. Long l: smooth functions, may miss patterns.
30. **GP: computational cost**
    - O(N^3) for training (matrix inversion). Limits GPs to moderate dataset sizes (~10^3-10^4).
31. **GP: strengths and limitations**
    - Strengths: principled UQ, automatic complexity control, interpretable kernels.
    - Limitations: O(N^3) scaling, kernel design requires domain knowledge.
32. **Checkpoint: GP prediction**
    - "A GP is trained on 10 points. You query a point far from all training data. What happens to the uncertainty?" — it grows toward the prior variance.

### Block D — Practical UQ methods (Slides 33-42)
33. **Mixture-Density Networks (MDNs)**
    - Predict parameters of a Gaussian mixture: p(y|x) = sum pi_k(x) N(y|mu_k(x), sigma_k^2(x)).
34. **MDN: capturing multi-modal uncertainty**
    - Unlike a single Gaussian output, MDNs can represent branching predictions (e.g., two possible outcomes).
35. **MC Dropout for uncertainty estimation**
    - Keep dropout active at test time. Run T forward passes → T different predictions. Variance = epistemic uncertainty.
36. **MC Dropout: interpretation**
    - Each forward pass samples a different sub-network → approximate posterior over architectures.
37. **Deep ensembles**
    - Train M independent networks (different initializations). Mean = prediction, variance across members = uncertainty.
38. **Stochastic enrichment (Neuer Ch. 6.4)**
    - Add noise to inputs during prediction to estimate sensitivity. High sensitivity = high uncertainty.
39. **Calibration: are uncertainties trustworthy?**
    - Predicted 90% intervals should contain 90% of test points. Calibration plot: predicted vs observed coverage.
40. **Recalibration methods**
    - Temperature scaling, Platt scaling: post-hoc adjustments to improve calibration.
41. **Comparison of UQ methods**
    - GPs: exact but O(N^3). MC Dropout: cheap but approximate. Ensembles: good but M× cost. MDNs: flexible but need careful training.
42. **Checkpoint: choosing a UQ method**
    - "Small dataset, need exact UQ → GP. Large dataset, need speed → MC Dropout or ensemble."

### Block E — Engineering applications + summary (Slides 43-50)
43. **Materials example: GP for composition-property mapping**
    - GP regression from alloy composition to yield strength. Uncertainty bands guide next experiments.
44. **Materials example: active learning with GP uncertainty**
    - Acquire new training points where GP uncertainty is highest → efficient experimental design.
45. **Materials example: MDN for multi-phase prediction**
    - Some compositions yield two possible phases → MDN captures bimodal property distribution.
46. **Physics-informed uncertainty reduction (preview of Unit 13)**
    - Embedding physical constraints reduces epistemic uncertainty without additional data.
47. **Lecture-essential vs exercise content split**
    - Lecture: Bayesian prediction, evidence framework, GP derivation, practical UQ taxonomy.
    - Exercise: GP from scratch, kernel hyperparameter exploration, ensemble comparison, MDN bonus.
48. **Exercise setup summary**
    - Implement GP regression (RBF kernel, posterior mean and variance) in NumPy.
    - Compare GP bands with NN ensemble predictions on 1D regression.
    - Vary kernel hyperparameters and observe uncertainty behavior.
49. **Exam-aligned summary: 10 must-know statements**
    1. The Bayesian predictive distribution integrates over parameter uncertainty.
    2. Total prediction variance = aleatory variance + epistemic variance.
    3. The marginal likelihood (evidence) measures model fit with automatic complexity penalty.
    4. A GP is a distribution over functions specified by mean and kernel functions.
    5. The GP posterior has closed-form mean and variance (for Gaussian likelihood).
    6. GP uncertainty grows away from training data — honest epistemic uncertainty.
    7. Kernel hyperparameters (length scale, signal variance) control GP behavior.
    8. MC Dropout approximates Bayesian inference by sampling sub-networks at test time.
    9. Deep ensembles provide uncertainty via disagreement among independently trained models.
    10. Calibration plots verify that predicted confidence matches observed accuracy.
50. **References + reading assignment for next unit**
    - Required: Murphy Ch. 5, 15. Neuer Ch. 6.4.
    - Optional: Bishop Ch. 3.5. Rasmussen & Williams (GP reference).
    - Next unit: Physics-Informed Learning — embedding domain knowledge into ML models.

---

## Reusable equations (for slide math boxes)
- Predictive distribution: \\(p(y^*|x^*,\mathcal{D}) = \int p(y^*|x^*,\theta)\,p(\theta|\mathcal{D})\,d\theta\\)
- GP posterior mean: \\(\mu^*(x^*) = \mathbf{k}_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{y}\\)
- GP posterior variance: \\(\sigma^{*2}(x^*) = k(x^*,x^*) - \mathbf{k}_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{k}_*\\)
- RBF kernel: \\(k(x,x') = \sigma_f^2\exp\!\left(-\frac{\|x-x'\|^2}{2\ell^2}\right)\\)
- Marginal likelihood: \\(\log p(\mathbf{y}|X) = -\tfrac{1}{2}\mathbf{y}^\top(K+\sigma_n^2 I)^{-1}\mathbf{y} - \tfrac{1}{2}\log|K+\sigma_n^2 I| - \tfrac{N}{2}\log 2\pi\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: Bayesian prediction, evidence framework, GP derivation, practical UQ taxonomy.
- **Exercise**: GP implementation, hyperparameter exploration, ensemble comparison, MDN bonus.
