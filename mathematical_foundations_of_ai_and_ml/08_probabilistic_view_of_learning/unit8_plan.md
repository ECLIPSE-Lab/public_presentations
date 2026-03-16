# Unit 8 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: 2 semesters math, 1 physics, some chemistry; Units 1-7 of MFML completed
- Assumed: probability basics from high school, calculus (integration, differentiation), linear algebra (SVD, eigenvalues), basic Python/NumPy
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 8)
By the end of the unit, students can:
1. Classify sources of uncertainty as aleatory or epistemic and explain why this distinction matters for model design and data collection strategy.
2. Write the Gaussian distribution in 1D and multivariate form, interpret its moments, and explain why it is the maximum-entropy distribution for given mean and variance.
3. Derive the Maximum Likelihood Estimator for the mean and variance of a Gaussian from the log-likelihood and connect the result to MSE minimization.
4. Apply Bayes' theorem to update a prior belief into a posterior distribution and contrast Bayesian inference with frequentist point estimation in terms of uncertainty quantification.
5. Describe how stochastic enrichment, mixture-density networks, and process corridors integrate uncertainty into engineering ML workflows.

## Book-aligned content mapping (priority order)
1. **Neuer (2024) Ch. 2.2**: Stochastics of data — sampling process (Eq. 2.23-2.24), Nyquist-Shannon theorem (Eq. 2.25-2.27), aleatory vs epistemic uncertainty (Sec. 2.2.3), expected value, variance, moments (Sec. 2.3).
2. **Neuer (2024) Ch. 6.4-6.5**: Stochastic enrichment of input/target data, mixture-density networks (Eq. 6.12-6.17), process corridors via 2D histograms (Eq. 6.19-6.20).
3. **Murphy (2012) Ch. 2**: Frequentist vs Bayesian interpretation, Bayes' rule, Gaussian PDF, Student's t-distribution (robustness), mean/variance properties.
4. **Murphy (2012) Ch. 4**: Multivariate Gaussian basics, MLE for MVN (Theorem 4.1.1), maximum entropy derivation.
5. **Bishop (2006) Ch. 2.1-2.3**: Gaussian distribution properties, MLE derivation, Bayesian inference for Gaussian parameters (conjugate priors).

## 90-minute structure
- 0-10 min: Motivation — why probability is the language of learning (link to Unit 1 risk framework)
- 10-25 min: Taxonomy of uncertainty (aleatory vs epistemic) + sampling and Nyquist theorem
- 25-45 min: Gaussian distribution, moments, and Maximum Likelihood Estimation
- 45-60 min: Bayesian inference — prior, likelihood, posterior; frequentist vs Bayesian comparison
- 60-75 min: Robustness (Student's t), stochastic enrichment, mixture-density networks, process corridors
- 75-85 min: Engineering case studies and integration with materials workflows
- 85-90 min: Summary + exercise handoff + reading assignment

## Exercise (90 min)
1. **Noise injection experiment**: Generate a clean sinusoidal signal, add Gaussian noise at three different SNR levels. Sample at various rates and verify the Nyquist criterion visually. Classify which noise is aleatory vs epistemic.
2. **MLE vs MSE comparison**: Implement MLE for Gaussian parameters from scratch in NumPy. Show algebraically and numerically that maximizing the Gaussian log-likelihood is equivalent to minimizing MSE. Compare MLE fits with and without outliers; repeat with Student's t-distribution likelihood to observe robustness.
3. **Bayesian update (bonus)**: Given a Gaussian prior on a parameter, compute the posterior analytically after observing N data points. Plot how the posterior sharpens as N increases. Compare the Bayesian credible interval to the frequentist confidence interval.

## Assessment alignment
- Written exam prep: Bayes' theorem derivation, MLE for Gaussian, uncertainty classification, likelihood-MSE equivalence.
- Every unit introduces 3-5 exam-relevant "must-know" statements.

## Slide production notes
- Style: follow data_science_for_em revealjs pattern (consistent with Unit 1)
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
- Equations in display math blocks for readability on 1920x1080
