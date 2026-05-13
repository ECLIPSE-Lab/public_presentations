# MFML Unit 8 — 50-Slide Content Pack (English)

## Unit theme
**Generalization, Bias-Variance, and Regularization**

## Core source mapping (book-priority aligned)
- **Neuer (2024)**: Ch. 4.5.9 (overfitting detection; cross-validation procedure; k-fold implementation).
- **McClarren (2021)**: Ch. 2.4 (ridge regression; lasso; elastic net; geometric constraint view; variable normalization).
- **Murphy (2012)**: Ch. 6.4.4 (bias-variance tradeoff derivation; MSE decomposition; ridge regression example), Ch. 6.5 (ERM; cross-validation; one-standard-error rule).
- **Bishop (2006)**: Ch. 3.2 (bias-variance decomposition for regression; expected loss = bias^2 + variance + noise).

---

## Slide-by-slide content (target: 50)

### Block A — Motivation and generalization (Slides 1-10)
1. **Title + unit positioning**
   - Unit 8 in the MFML arc: from fitting to generalization.
2. **Learning outcomes for Unit 8**
   - Bias-variance decomposition, regularization, cross-validation, model selection.
3. **Recall: ERM from Unit 1**
   - We minimize empirical risk, but care about population risk.
4. **The generalization gap**
   - Definition: difference between test error and training error.
5. **Why generalization is the central goal**
   - A model that only fits training data is useless in deployment.
6. **Overfitting — definition and visual example**
   - Model captures noise instead of signal; low train error, high test error.
7. **Underfitting — definition and visual example**
   - Model too simple to capture structure; high error on both sets.
8. **The complexity spectrum**
   - Underfitting at low capacity, overfitting at high capacity, sweet spot in between.
9. **Detecting overfitting in practice**
   - Compare training vs validation loss curves over training epochs or model complexity.
10. **Engineering consequence: false confidence**
    - Perfect training fit can mask catastrophic deployment failure.

### Block B — Bias-variance decomposition (Slides 11-22)
11. **Setup: expected prediction error**
    - Consider expected loss over both data D and new test point (x, y).
12. **Decomposing squared error — step 1**
    - Add and subtract the expected prediction: E_D[f_hat(x)].
13. **Decomposing squared error — step 2**
    - Expand the square; cross-term vanishes in expectation.
14. **The three components**
    - Bias^2: (E_D[f_hat(x)] - f_true(x))^2.
    - Variance: E_D[(f_hat(x) - E_D[f_hat(x)])^2].
    - Noise: irreducible error from data noise (Bayes error).
15. **Bias — interpretation**
    - Systematic error from restrictive model assumptions.
16. **Variance — interpretation**
    - Sensitivity of the learned model to the particular training sample.
17. **Intrinsic noise / Bayes error**
    - Lower bound on achievable error; cannot be removed by any model.
18. **The bias-variance tradeoff**
    - Increasing complexity: bias decreases, variance increases.
19. **Visual: U-shaped total error curve**
    - Plotting bias^2, variance, and total error vs complexity.
20. **Example: polynomial regression**
    - Degree 1 = high bias; degree 15 = high variance; degree 3-5 = sweet spot.
21. **Example: ridge regression and the tradeoff (Murphy 6.4.4.2)**
    - High lambda = high bias, low variance; low lambda = low bias, high variance.
22. **Checkpoint: why is the MLE not always best?**
    - Biased estimators (MAP/regularized) can have lower MSE overall.

### Block C — Regularization techniques (Slides 23-34)
23. **Regularization — the key idea**
    - Add penalty to loss function to discourage complexity.
24. **Regularized ERM**
    - Objective: min_theta R_N(theta) + lambda * Omega(theta).
25. **Ridge regression (L2 penalty)**
    - Penalty: lambda * ||w||_2^2; shrinks all coefficients toward zero.
26. **Ridge regression — closed-form solution**
    - w_ridge = (X^T X + lambda I)^{-1} X^T y; always invertible for lambda > 0.
27. **Ridge regression — geometric view**
    - Constraint region is a circle/sphere; solution on ellipse-circle intersection.
28. **Lasso regression (L1 penalty)**
    - Penalty: lambda * ||w||_1; promotes sparsity.
29. **Lasso — geometric view and sparsity**
    - Diamond constraint region; solution at corners sets coefficients to zero.
30. **Ridge vs Lasso — side-by-side comparison**
    - Ridge shrinks uniformly; Lasso performs variable selection.
31. **Elastic net (brief)**
    - Convex combination of L1 and L2; groups of parameters to zero.
32. **Normalization requirement**
    - Features must be scaled before regularization; otherwise penalty is inconsistent.
33. **Dropout as regularization (neural networks)**
    - Randomly zero out neurons during training; prevents co-adaptation.
34. **Choosing lambda — preview of cross-validation**
    - Lambda is a hyperparameter; cannot be learned from training data alone.

### Block D — Cross-validation and model selection (Slides 35-44)
35. **Train / validation / test — the three roles**
    - Train: fit parameters. Validation: tune hyperparameters. Test: final evaluation.
36. **Why we need three sets, not two**
    - Using test for tuning causes optimistic bias in reported performance.
37. **k-fold cross-validation — procedure**
    - Split data into k folds; rotate test fold; average performance.
38. **k-fold cross-validation — variance reduction**
    - Every data point contributes to both training and evaluation.
39. **Leave-one-out CV**
    - k = N; low bias but high variance and computational cost.
40. **Choosing lambda via CV (Murphy 6.5.3.1)**
    - Plot CV error vs log(lambda); select minimum or one-standard-error rule.
41. **The one-standard-error rule**
    - Pick simplest model within one SE of the minimum CV error.
42. **Model selection: complexity vs performance**
    - Compare models of different families and capacities using CV.
43. **Grouped / stratified CV for materials data**
    - When IID assumptions fail: block by composition, batch, or condition.
44. **Checkpoint MCQ slide**
    - Scenario: student uses test set to tune lambda. What goes wrong?

### Block E — Engineering applications + summary (Slides 45-50)
45. **Materials example: overfitting in alloy property prediction**
    - Many compositional features, limited experimental samples; regularization essential.
46. **Materials example: Lasso for identifying governing features**
    - Sparsity selects physically meaningful variables from a large candidate set.
47. **Materials example: polynomial models for process-property curves**
    - High-degree fit captures batch noise; low-degree misses nonlinearity.
48. **Lecture-essential vs exercise content split**
    - Lecture: derivations, interpretations, design principles.
    - Exercise: implementation, sweeps, diagnostics.
49. **Exam-aligned summary: 10 must-know statements**
    1. Generalization gap = test error - train error.
    2. Overfitting: model learns noise, not signal.
    3. MSE = bias^2 + variance + irreducible noise.
    4. Bias decreases and variance increases with model complexity.
    5. Ridge (L2) shrinks all weights; Lasso (L1) sets some to zero.
    6. Regularization strength lambda must be tuned, not learned from training data.
    7. Cross-validation provides an unbiased estimate of generalization error.
    8. Train/val/test roles must never be mixed.
    9. Feature normalization is mandatory before regularization.
    10. Model selection balances complexity against validated performance.
50. **References + reading assignment for next unit**
    - Required reading before Unit 8:
      - Neuer: Ch. 4.5.9 (review overfitting and CV)
      - McClarren: Ch. 2.4 (ridge, lasso, elastic net)
    - Optional depth:
      - Murphy: Ch. 6.4.4, 6.5.3 (bias-variance, CV for lambda)
      - Bishop: Ch. 3.2 (bias-variance decomposition)
    - Next unit: kernel methods and support vector machines.

---

## Reusable equations (for slide math boxes)
- Bias-variance decomposition: \\(\mathbb{E}_{\mathcal{D}}[\ell] = \text{Bias}^2 + \text{Variance} + \sigma^2_{\text{noise}}\\)
- Bias: \\(\text{Bias}[f_\theta(x)] = \mathbb{E}_{\mathcal{D}}[f_\theta(x)] - f_{\text{true}}(x)\\)
- Variance: \\(\text{Var}[f_\theta(x)] = \mathbb{E}_{\mathcal{D}}\big[(f_\theta(x) - \mathbb{E}_{\mathcal{D}}[f_\theta(x)])^2\big]\\)
- Ridge loss: \\(L_{\text{ridge}} = \sum_{i=1}^{N}(\hat{y}_i - y_i)^2 + \lambda \|\mathbf{w}\|_2^2\\)
- Lasso loss: \\(L_{\text{lasso}} = \sum_{i=1}^{N}(\hat{y}_i - y_i)^2 + \lambda \|\mathbf{w}\|_1\\)
- Ridge closed-form: \\(\hat{\mathbf{w}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}\\)
- Elastic net: \\(L_{\text{el}} = \sum_{i=1}^{N}(\hat{y}_i - y_i)^2 + \lambda\big(\alpha\|\mathbf{w}\|_1 + (1-\alpha)\|\mathbf{w}\|_2^2\big)\\)
- k-fold CV error: \\(\text{CV}(k) = \frac{1}{k}\sum_{j=1}^{k} R_{\text{test}}^{(j)}\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: bias-variance derivation, regularization formulation, geometric interpretation, CV design, model selection principles.
- **Exercise**: polynomial overfitting demo, lambda sweeps, Ridge vs Lasso comparison, CV implementation, materials feature selection.

## Reading assignment after Unit 8
- Neuer Ch. 4.5.9
- McClarren Ch. 2.4
- Murphy Ch. 6.4.4 + 6.5.3 (skim)
- Bishop Ch. 3.2 (conceptual)
