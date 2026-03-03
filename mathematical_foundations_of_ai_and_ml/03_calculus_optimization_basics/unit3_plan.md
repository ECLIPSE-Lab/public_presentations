# Unit 3 Plan — Mathematical Foundations of AI & ML

## Unit title
Calculus and Optimization Basics for Machine Learning

## Learning objectives
1. Connect derivatives/gradients to learning objectives and parameter updates.
2. Explain convexity, critical points, and conditioning in optimization behavior.
3. Derive gradient descent updates and interpret hyperparameters geometrically.
4. Distinguish first-order vs second-order intuition for ML optimization.
5. Identify optimization-related failure modes (plateaus, exploding/vanishing gradients, poor scaling).

## 90-minute structure
- 0–10: recap from linear algebra + objective formulation
- 10–30: scalar and multivariate derivatives in ML notation
- 30–50: gradients, Jacobians, Hessians; convexity and curvature intuition
- 50–70: optimization algorithms (GD, SGD, momentum, adaptive methods)
- 70–85: practical pathologies + mitigation (normalization, LR schedules, regularization)
- 85–90: summary + exercise handoff

## Exercise (90 min)
- implement gradient descent for linear/logistic objective
- compare learning-rate schedules
- inspect loss landscape slices
- demonstrate conditioning effects with feature scaling
