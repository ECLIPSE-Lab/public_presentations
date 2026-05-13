# MFML Unit 7 — 50-Slide Content Pack (English)

## Unit theme
**The probabilistic view of learning: uncertainty, likelihood, and Bayesian inference**

## Core source mapping (book-priority aligned)
- **Neuer (2024)**: Ch. 2.2–2.3 (stochastics of data, sampling, Nyquist-Shannon, aleatory vs epistemic uncertainty, moments), Ch. 6.4–6.5 (stochastic enrichment, mixture-density networks, process corridors).
- **Murphy (2012)**: Ch. 2 (frequentist vs Bayesian, Bayes' rule, Gaussian, Student's t), Ch. 4 (multivariate Gaussian, MLE for MVN, maximum entropy).
- **Bishop (2006)**: Ch. 2.1–2.3 (Gaussian properties, MLE derivation, Bayesian inference for Gaussian parameters).

---

## Slide-by-slide content (target: 50)

### Block A — Motivation and uncertainty taxonomy (Slides 1-10)
1. **Title + Unit 7 positioning**
   - From optimization (Units 5-6) and generalization (Unit 7) to probabilistic foundations.
2. **Recap: what risk minimization assumes**
   - Unit 1: minimize expected loss. But expected over what? Over a probability distribution of data.
3. **Learning outcomes for Unit 7**
   - Aleatory vs epistemic uncertainty, Gaussian distribution, MLE, Bayesian inference, engineering applications.
4. **Why probability is the language of learning**
   - Data is inherently noisy. Models are uncertain. Probability quantifies both rigorously.
5. **Aleatory uncertainty — definition**
   - Irreducible randomness in the data-generating process (measurement noise, quantum effects, thermal fluctuations).
6. **Epistemic uncertainty — definition**
   - Uncertainty from limited knowledge — reducible by collecting more data or improving the model.
7. **Why the distinction matters**
   - Aleatory: set error bars. Epistemic: collect more data or improve the model. Confusing the two wastes resources.
8. **The sampling process (Neuer Ch. 2.2)**
   - Data as realizations of a random process; sampling rate, digitization, and reconstruction.
9. **Nyquist-Shannon theorem**
   - Minimum sampling rate f_s >= 2 f_max for perfect reconstruction; aliasing when violated.
10. **Engineering example: sensor data and uncertainty sources**
    - Temperature sensor: aleatory = thermal noise; epistemic = calibration error. Both affect ML model quality.

### Block B — Gaussian distribution and moments (Slides 11-20)
11. **Random variables and probability distributions**
    - Discrete vs continuous; PMF vs PDF; the PDF integrates to 1.
12. **Expected value and variance**
    - E[X] = integral x p(x) dx; Var[X] = E[(X - mu)^2]; standard deviation sigma = sqrt(Var).
13. **Higher moments: skewness and kurtosis**
    - Skewness: asymmetry. Kurtosis: tail heaviness. The Gaussian has skewness 0, kurtosis 3.
14. **The Gaussian distribution (1D)**
    - PDF: p(x|mu, sigma^2) = (1/sqrt(2 pi sigma^2)) exp(-(x-mu)^2 / (2 sigma^2)).
15. **Why the Gaussian is special: maximum entropy**
    - Among all distributions with given mean and variance, the Gaussian has maximum entropy (maximum uncertainty).
16. **Central Limit Theorem connection**
    - Averages of many independent random variables converge to a Gaussian — universal relevance.
17. **Multivariate Gaussian distribution**
    - PDF: p(x|mu, Sigma) = (2pi)^{-d/2} |Sigma|^{-1/2} exp(-1/2 (x-mu)^T Sigma^{-1} (x-mu)).
18. **Covariance matrix: diagonal vs full**
    - Diagonal: independent features. Full: correlated features. Visualization as axis-aligned vs rotated ellipses.
19. **Marginal and conditional Gaussians**
    - Marginals and conditionals of a joint Gaussian are also Gaussian — a key computational convenience.
20. **Checkpoint: interpret the covariance matrix**
    - Given a 2D Gaussian with specific Sigma, sketch the contour ellipse and identify correlated variables.

### Block C — Maximum Likelihood Estimation (Slides 21-30)
21. **The likelihood function**
    - Given data D = {x_1, ..., x_N}, the likelihood is L(theta) = product p(x_i | theta).
22. **Log-likelihood**
    - log L(theta) = sum log p(x_i | theta). Sums are easier to optimize than products.
23. **MLE principle**
    - theta_MLE = argmax_theta log L(theta). Choose parameters that make the observed data most probable.
24. **MLE for Gaussian mean**
    - Derivative of log-likelihood w.r.t. mu, set to zero: mu_MLE = (1/N) sum x_i = sample mean.
25. **MLE for Gaussian variance**
    - sigma^2_MLE = (1/N) sum (x_i - mu_MLE)^2. This is the (biased) sample variance.
26. **MLE and MSE: the connection**
    - For Gaussian noise, maximizing log-likelihood is equivalent to minimizing mean squared error. This justifies MSE as a loss function.
27. **MLE for multivariate Gaussian**
    - mu_MLE = sample mean vector; Sigma_MLE = sample covariance matrix. Direct extension.
28. **MLE: properties and limitations**
    - Consistent (converges to true theta as N -> infinity), efficient, but can overfit with small N.
29. **Robustness: the outlier problem**
    - MLE for Gaussian is sensitive to outliers because the exponential tail decays fast.
30. **Student's t-distribution for robust estimation**
    - Heavier tails than Gaussian; MLE with Student's t downweights outliers automatically.

### Block D — Bayesian inference (Slides 31-42)
31. **Bayes' theorem — statement**
    - p(theta | D) = p(D | theta) p(theta) / p(D). Posterior = likelihood × prior / evidence.
32. **Components of Bayes' theorem**
    - Prior p(theta): what we believe before seeing data. Likelihood p(D|theta): data compatibility. Posterior p(theta|D): updated belief.
33. **The evidence (marginal likelihood)**
    - p(D) = integral p(D|theta) p(theta) d theta. Normalizing constant; often intractable.
34. **Bayesian inference for Gaussian mean (known variance)**
    - Gaussian prior + Gaussian likelihood = Gaussian posterior. Conjugate prior pair.
35. **Posterior update: visual intuition**
    - Prior is wide (uncertain). After N data points, posterior narrows. As N -> infinity, posterior concentrates at MLE.
36. **Bayesian vs frequentist comparison**
    - Frequentist: point estimate + confidence interval (repeated sampling interpretation). Bayesian: full posterior distribution (belief update).
37. **Credible interval vs confidence interval**
    - Bayesian credible interval: "the parameter is in this range with 95% probability." Frequentist confidence interval: "95% of such intervals contain the true parameter."
38. **MAP estimation**
    - Maximum A Posteriori: theta_MAP = argmax_theta p(theta|D). A point estimate from the posterior.
39. **MAP and regularization**
    - Gaussian prior on weights → MAP = Ridge regression. Laplace prior → MAP = Lasso. Bayesian justification for regularization.
40. **When to be Bayesian vs frequentist**
    - Small data, safety-critical: Bayesian (uncertainty quantification). Large data, fast iteration: frequentist/MLE often sufficient.
41. **Predictive distribution**
    - p(x_new | D) = integral p(x_new | theta) p(theta | D) d theta. Integrates over parameter uncertainty.
42. **Checkpoint: update a prior**
    - Given a Gaussian prior mu_0, sigma_0^2 and 5 observations, compute the posterior mean and variance.

### Block E — Engineering applications + summary (Slides 43-50)
43. **Stochastic enrichment of input data (Neuer Ch. 6.4)**
    - Add noise to training inputs to simulate measurement uncertainty; improves robustness.
44. **Mixture-density networks (Neuer Ch. 6.5)**
    - Instead of predicting a single value, predict parameters of a Gaussian mixture. Captures multi-modal uncertainty.
45. **Process corridors via 2D histograms (Neuer Ch. 6.5)**
    - Visualize acceptable process parameter ranges as probability contours; define quality corridors.
46. **Materials example: property prediction with uncertainty**
    - Predict tensile strength with confidence bands. Epistemic uncertainty flags extrapolation regions.
47. **Practical diagnostic: calibration plots**
    - A well-calibrated model's predicted 90% intervals should contain 90% of test points.
48. **Lecture-essential vs exercise content split**
    - Lecture: probability foundations, MLE derivation, Bayesian framework, engineering integration.
    - Exercise: MLE implementation, noise injection, Bayesian updating, calibration assessment.
49. **Exam-aligned summary: 10 must-know statements**
    1. Aleatory uncertainty is irreducible; epistemic uncertainty is reducible with more data.
    2. The Gaussian is the maximum-entropy distribution for given mean and variance.
    3. MLE maximizes the probability of the observed data under the model.
    4. For Gaussian noise, MLE is equivalent to MSE minimization.
    5. Bayes' theorem: posterior ∝ likelihood × prior.
    6. Conjugate priors yield closed-form posteriors (e.g., Gaussian-Gaussian).
    7. MAP estimation with a Gaussian prior is equivalent to Ridge regression.
    8. The predictive distribution integrates over parameter uncertainty.
    9. Student's t-distribution provides robustness to outliers.
    10. Calibration plots assess whether predicted uncertainties match observed frequencies.
50. **References + reading assignment for next unit**
    - Required: Neuer Ch. 2.2–2.3, Murphy Ch. 2.
    - Optional: Bishop Ch. 2.1–2.3, Murphy Ch. 4.
    - Next unit: Representation Learning.

---

## Reusable equations (for slide math boxes)
- Gaussian 1D: \\(p(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\\)
- MVN: \\(p(\mathbf{x}|\boldsymbol{\mu},\boldsymbol{\Sigma}) = (2\pi)^{-d/2}|\boldsymbol{\Sigma}|^{-1/2}\exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)\\)
- Log-likelihood: \\(\ell(\theta) = \sum_{i=1}^N \log p(x_i|\theta)\\)
- MLE mean: \\(\hat{\mu}_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N x_i\\)
- MLE variance: \\(\hat{\sigma}^2_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N (x_i - \hat{\mu})^2\\)
- Bayes' theorem: \\(p(\theta|\mathcal{D}) = \frac{p(\mathcal{D}|\theta)\,p(\theta)}{p(\mathcal{D})}\\)
- Posterior Gaussian mean: \\(\mu_N = \frac{\sigma^2\mu_0 + N\sigma_0^2\bar{x}}{\sigma^2 + N\sigma_0^2}\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: uncertainty taxonomy, Gaussian properties, MLE derivation, Bayesian framework, MAP-regularization connection, engineering examples.
- **Exercise**: noise injection and Nyquist demo, MLE implementation, MLE-MSE equivalence proof, Bayesian update computation, calibration plots.

## Reading assignment after Unit 7
- Neuer Ch. 2.2–2.3
- Murphy Ch. 2 (probability review)
- Bishop Ch. 2.1–2.3 (skim)
- Murphy Ch. 4 (conceptual)
