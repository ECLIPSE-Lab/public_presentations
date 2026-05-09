# Materials Genomics Unit 7 — Regression and Generalization in Materials Data

## Core teaching claim
In materials ML, a low test error is not enough. A useful regression model must be evaluated under a split design that reflects the scientific question: new compositions, new structure families, or new calculation conditions. Unit 7 therefore teaches regression and validation together.

## 1. From representation to prediction

Unit 6 ended with each material mapped to a feature vector. Unit 7 asks what happens next. We choose a target property, fit a predictive model, and then test whether the learned relation survives beyond the narrow domain seen during training.

This is where many materials ML papers fail: representation work is strong, but the validation protocol is too weak to support the scientific claim.

## 2. Regression as empirical risk minimization

Given data `(x_i, y_i)`, regression chooses parameters `theta` by minimizing

`hat(theta) = arg min_theta (1/N) sum_i ell(f_theta(x_i), y_i) + lambda Omega(theta)`

The data term measures prediction error. The regularization term penalizes overly flexible models. In materials science, the point is not only numerical optimization. The point is to build a surrogate that remains credible when chemistry or structure changes.

## 3. The target variable matters

Not all targets are equally learnable.
- band gap can be noisy because DFT approximations and polymorph differences matter
- formation energy or energy above hull may be smoother but still depend strongly on chemistry coverage
- elastic moduli and conductivities may have broad dynamic ranges and heterogeneous uncertainty

Target choice therefore changes the difficulty of the regression problem and the meaning of evaluation metrics.

## 4. Start with a linear baseline

The simplest model is linear regression:

`hat(y) = w^T x + b`

This baseline matters because it tests whether the chosen representation already contains a roughly linear signal. If a simple linear model performs surprisingly well, that is useful scientific information. If it fails badly, that also teaches us something about the representation.

## 5. Least squares geometry

With mean squared error, linear regression chooses `w` so that `Xw` is as close as possible to `y` in Euclidean distance. Geometrically, `Xw` is the projection of `y` onto the column space of the feature matrix `X`.

That interpretation matters because it explains when the model cannot fit the target well: the signal may simply not be represented in the available features.

## 6. Why linear baselines remain important

- they are interpretable
- they are cheap to train
- they reveal whether representation quality or model flexibility is the main bottleneck
- they often remain competitive in small-data regimes

In materials ML, a complicated model that barely beats ridge regression is often less interesting than the paper claims.

## 7. Ridge regularization

Ridge regression adds an `L2` penalty:

`hat(w) = arg min_w ||y - Xw||_2^2 + lambda ||w||_2^2`

Conceptually, ridge shrinks the coefficients and stabilizes the fit when features are correlated. This is common in materials features, where composition descriptors, local-environment summaries, and learned embeddings often overlap in information content.

## 8. Lasso and elastic net

- Lasso uses an `L1` penalty and can drive some coefficients exactly to zero.
- Elastic net combines `L1` and `L2` behavior.

These methods are useful when feature selection matters or when the feature space is large relative to the number of materials. They are not magic, but they often provide a stronger baseline than unregularized least squares.

## 9. Nonlinear baselines

Not every structure-property relation is close to linear. Tree ensembles, kernel models, or shallow neural networks can capture nonlinear interactions between features.

But the important comparison is not "complex model versus no model." It is "complex model versus the strongest simple baseline under the same split."

## 10. Baseline competition must be fair

A fair comparison requires:
- the same train-validation-test split
- the same preprocessing logic
- the same target transformation, if any
- the same reporting metrics

If one model gets a random split and another a grouped split, the benchmark is meaningless.

## 11. The bias-variance tradeoff in materials data

- High-bias models are too simple and miss real structure-property trends.
- High-variance models fit idiosyncrasies of the training set.

Materials datasets are often neither huge nor clean. That means variance control matters a lot. The winning model is often not the most expressive one, but the one whose complexity best matches the available data.

## 12. Overfitting in small and medium datasets

- A model may achieve extremely low training error by memorizing chemistry families or near-duplicate structures.
- This looks impressive until evaluation moves to genuinely new compositions or prototypes.
- Overfitting in materials ML is often hidden by weak split design rather than by obviously bad loss curves.

## 13. What metrics actually measure

- `MAE` gives average absolute error and is easy to interpret in physical units.
- `RMSE` penalizes large mistakes more strongly.
- `R^2` measures explained variance relative to a constant baseline.

None of these tells us automatically whether the model generalizes to new chemistry. Metrics summarize error on a chosen split; they do not validate the split itself.

## 14. Metric choice depends on the scientific task

- If large failures are especially costly, `RMSE` matters.
- If average deviation in physical units is the focus, `MAE` is often clearer.
- If ranking candidate materials matters more than exact calibration, rank-based measures may be relevant.

The right metric follows the decision problem, not benchmarking fashion.

## 15. Random splits are often too optimistic

- Random train-test splits distribute similar compositions and prototypes across both sets.
- The model then benefits from near neighbors in training that are almost duplicates of test examples.
- This usually inflates performance and produces an overly optimistic estimate of discovery power.

This is one of the central methodological problems in materials ML.

## 16. Grouped chemistry-aware splits

- A grouped split holds out entire chemistry families, composition groups, or prototype families.
- This is harder because the model cannot rely on nearly identical examples appearing in training.
- But it is scientifically more honest when the goal is extrapolation to new classes of materials.

The split should mirror the claim. If the claim is generalization to unseen chemistries, the split must enforce that.

## 17. Prototype-aware and OOD evaluation

Two out-of-distribution questions are different:
- can the model handle unseen compositions?
- can it handle unseen structure prototypes?

These are not interchangeable. A model may extrapolate across composition within one prototype family and still fail catastrophically on a new structural motif.

## 18. Validation versus test logic

- Training data fit the model parameters.
- Validation data choose hyperparameters and model class.
- Test data are used once for the final performance estimate.

If the test set influences model selection, the estimate is no longer an honest measure of future performance.

## 19. Nested validation, conceptually

Nested validation separates hyperparameter tuning from final evaluation. It is especially useful when datasets are small and model selection choices are numerous.

The exact procedure can be expensive, but the principle is simple: never let the final evaluation set guide your optimization decisions.

## 20. Leakage is more than copying the target

Feature-target leakage in materials ML can happen subtly:
- duplicate or near-duplicate structures appear in train and test
- post-relaxation quantities leak information about a target derived from the same calculation
- preprocessing is fit on the full dataset instead of training data only

Leakage produces high scores for the wrong reason.

## 21. Learning curves reveal the next bottleneck

- If training error is low but validation error remains high, variance is the issue.
- If both errors are high, the representation or model may be too weak.
- If the validation curve still improves strongly with more data, the problem may be data-limited rather than model-limited.

Learning curves convert vague intuition into a diagnostic tool.

## 22. Residual analysis is not optional

A single scalar metric hides structure in the errors. Residual plots can show:
- systematic underprediction of one chemistry family
- larger errors at one end of the target range
- failures concentrated in unusual prototypes or disordered materials

These patterns often matter more scientifically than a small change in MAE.

## 23. Heteroscedastic noise

Materials targets often do not have constant noise:
- high-energy structures may be less reliable
- some chemistry classes may be harder to model than others
- experimental and computational datasets may mix noise sources

A regression model that ignores this can look accurate on average while being unreliable exactly where decisions matter.

## 24. Worked benchmark: random split versus grouped split

Imagine a band-gap dataset with many related compounds. Under a random split, ridge regression might appear excellent because close composition neighbors appear on both sides of the split. Under a grouped split by chemistry family, the same model can degrade sharply.

The lesson is not that ridge is bad. The lesson is that the split defines the claim.

## 25. Worked benchmark: simple versus complex model

Suppose a random forest beats ridge on the validation set by a small margin. Before concluding that nonlinear modeling is essential, check:
- does the gain persist under a grouped split?
- are the residuals actually improved in hard families?
- is the added complexity justified by the scientific use case?

In small-data materials problems, complexity often wins only on paper.

## 26. Domain shift across databases

- Models trained on one database may fail on another because calculation settings, functional choices, relaxation criteria, or curation standards differ.
- This is domain shift even when the nominal target is the same.
- A benchmark that ignores this can confuse database-specific patterns with general materials physics.

## 27. Trust criteria for surrogate models

Before using a surrogate in screening or optimization, ask:
- does it beat strong simple baselines?
- is the split consistent with the intended deployment scenario?
- are residuals acceptable in the chemistry families that matter?
- does performance degrade gracefully under OOD tests?
- is there evidence against leakage?

A low average error alone is not enough.

## 28. Why interpretability still matters

- Linear or sparse models can reveal which descriptor families matter most.
- Even when the final deployed model is nonlinear, interpretable baselines help detect spurious correlations.
- Interpretability is especially valuable when the dataset is small and domain knowledge should constrain the conclusions.

## 29. The anti-pattern of leaderboard optimization

A common failure mode is to optimize one benchmark score without checking whether the benchmark reflects a real discovery scenario. This leads to:
- excessive hyperparameter tuning on weak splits
- underreporting of baseline models
- inflated claims of extrapolation

Scientifically, this is far less useful than a modest model evaluated honestly.

## 30. Summary

The core message of Unit 7 is:
- prediction quality depends on representation, model class, and regularization
- but in materials ML, evaluation design is often the dominant issue
- grouped splits, learning curves, and residual analysis are central scientific tools
- a surrogate model is useful only when its benchmark matches the intended deployment claim

## 31. Bridge to Unit 8

Unit 8 will introduce neural surrogates for materials properties. The benchmark discipline from this unit carries forward unchanged: a neural model is interesting only if it improves performance under a credible validation protocol.
