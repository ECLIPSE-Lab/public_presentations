# Unit 7 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: 2 semesters math, 1 physics, some chemistry
- Assumed: linear algebra, calculus, probability basics, loss functions, ERM from Units 1-6
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 7)
By the end of the unit, students can:
1. Derive and interpret the bias-variance decomposition of expected prediction error.
2. Diagnose overfitting vs underfitting from training/validation loss curves.
3. Formulate Ridge (L2) and Lasso (L1) regularization as constrained optimization and explain their geometric effects on parameters.
4. Design a k-fold cross-validation procedure for model selection and hyperparameter tuning.
5. Apply regularization and validation strategies to engineering regression problems with awareness of failure modes.

## Book-aligned content mapping (priority order)
1. Neuer (2024): Ch. 4.5.9 (overfitting and cross-validation; detection via train/test error divergence; k-fold procedure).
2. McClarren (2021): Ch. 2.4 (ridge regression, lasso, elastic net; geometric interpretation; variable selection; normalization requirement).
3. Murphy (2012): Ch. 6.4.4 (bias-variance tradeoff derivation; MSE = variance + bias^2; ridge regression example; classification caveat), Ch. 6.5 (empirical risk minimization; cross-validation for lambda selection; one-standard-error rule).
4. Bishop (2006): Ch. 3.2 (bias-variance decomposition for regression; expected loss decomposition; intrinsic noise term).

## 90-minute structure
- 0-10 min: Why generalization is the central goal of ML (connect to Unit 1 ERM)
- 10-25 min: Overfitting vs underfitting — visual examples and detection criteria
- 25-45 min: Bias-variance decomposition — derivation from expected squared error
- 45-60 min: Regularization — Ridge (L2), Lasso (L1), dropout, geometric intuition
- 60-75 min: Cross-validation, train/val/test roles, model selection strategies
- 75-85 min: Engineering examples — overfitting in materials property models
- 85-90 min: Summary + exercise handoff

## Exercise (90 min)
- **Overfitting demo:** Fit polynomial models of degree 1, 5, 15 to noisy sinusoidal data; plot train vs test error as function of degree.
- **Neural network comparison:** Train a small MLP on the same data; observe overfitting with increasing hidden units/epochs.
- **Regularization sweep:** Add L2 penalty to polynomial regression; sweep lambda and plot bias-variance tradeoff empirically.
- **Cross-validation:** Implement 5-fold CV to select optimal polynomial degree and regularization strength.
- **Bonus:** Compare Lasso vs Ridge on a 20-feature materials dataset where only 3 features are relevant; observe sparsity.

## Assessment alignment
- Written exam prep: bias-variance decomposition derivation is a core exam topic.
- Every unit introduces 3-5 exam-relevant "must-know" statements.

## Slide production notes
- Style: follow data_science_for_em revealjs pattern
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
