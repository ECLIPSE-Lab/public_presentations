# Unit 7 Plan — Materials Genomics

## Unit title
Regression and Generalization in Materials Data

## Lecture arc
- Turn structure or composition features into a supervised regression problem for materials targets.
- Establish simple baselines first: linear regression, ridge, lasso, and one nonlinear benchmark.
- Explain what metrics mean and where they can mislead.
- Show why split design is the central scientific issue in materials ML benchmarks.
- Diagnose models through learning curves, residuals, leakage checks, and out-of-distribution tests.
- End with trust criteria for surrogate models and a bridge to neural-network surrogates.

## Timing guide for 90 minutes
- 0-10 min: recap from local descriptors and setup of the supervised prediction problem
- 10-30 min: targets, loss minimization, and linear baselines
- 30-50 min: regularization and nonlinear baselines
- 50-70 min: grouped splits, cross-validation, and leakage
- 70-85 min: learning curves, residual analysis, OOD behavior, and trust
- 85-90 min: summary and transition to Unit 8

## Must-cover concepts
- Regression as empirical risk minimization
- Target choice and target noise in materials datasets
- Linear, ridge, lasso, and one nonlinear baseline
- Metric meaning: MAE, RMSE, `R^2`, and ranking use cases
- Random versus grouped chemistry-aware evaluation
- Residual analysis and learning curves
- Leakage, domain shift, and trust criteria before deployment

## Optional cuts if time runs short
- Shorten the comparison of multiple nonlinear baselines to one tree-based example.
- Keep nested validation conceptual instead of operational.
- Collapse the final deployment checklist into the summary slide.

## Exercise handoff
- Build one reproducible materials-regression benchmark.
- Compare a linear baseline with one nonlinear baseline under the same grouped split.
- Report metrics, learning curves, and residual patterns.
- Identify one failure mode and propose one scientifically motivated mitigation.

## Bridge to Unit 8
Unit 8 keeps the same benchmark discipline but replaces hand-designed baselines with neural surrogates. Students should leave Unit 7 knowing that a more flexible model is only useful if it improves performance under credible validation.
