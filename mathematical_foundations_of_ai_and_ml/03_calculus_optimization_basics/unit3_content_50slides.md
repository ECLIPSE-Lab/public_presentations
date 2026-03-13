# MFML Unit 3 — 50-Slide Scaffold Pack

## Slide-by-slide scaffold

1. **Title: Calculus & Optimization for ML**
- Unit focus and dependency role.
- Bridge from linear algebra to trainable models.
2. **Learning objectives**
- What students should master today.
- How this unit supports all later ML topics.
3. **Recap from Unit 2**
- Projection and matrix notation refresher.
- Optimization needs geometric intuition from LA.
4. **Objective functions in ML**
- Loss as function of parameters.
- Examples: MSE and logistic loss.
5. **Single-variable derivatives refresher**
- Rate-of-change meaning.
- Gradient direction intuition starts here.
6. **Partial derivatives**
- Multivariate sensitivity interpretation.
- Coordinate-wise update idea.
7. **Total derivative and chain rule**
- Composed models need chain rule.
- Core mechanism behind backprop later.
8. **Gradient vector**
- Direction of steepest ascent.
- Negative gradient for descent.
9. **Directional derivatives**
- Sensitivity along chosen direction.
- Useful for line search reasoning.
10. **Jacobian matrix (conceptual)**
- Vector-output derivatives.
- Needed for transformation pipelines.
11. **Hessian matrix (conceptual)**
- Second-order curvature information.
- Local shape around critical points.
12. **Critical points**
- Gradient equals zero.
- Minima/maxima/saddle distinction.
13. **Convex sets and convex functions**
- Why convexity simplifies optimization.
- Global optimum guarantees in convex problems.
14. **Strong convexity intuition**
- Curvature lower bound.
- Faster convergence implications.
15. **Lipschitz gradient intuition**
- Smoothness upper bound.
- Step-size stability relevance.
16. **Conditioning and curvature anisotropy**
- Elongated valleys slow descent.
- Feature scaling improves behavior.
17. **Taylor expansion viewpoint**
- Local approximation of loss.
- Connects first and second-order methods.
18. **Gradient descent update rule**
- w_{t+1}=w_t-eta*grad L.
- Interpret eta as step size.
19. **Learning rate sensitivity**
- Too high diverges, too low stalls.
- Need principled tuning/schedules.
20. **Batch gradient descent**
- Uses full dataset gradients.
- Stable but computationally heavy.
21. **Stochastic gradient descent**
- Uses sample/minibatch gradients.
- Noisy updates can aid escape from saddles.
22. **Mini-batch tradeoff**
- Variance vs compute efficiency.
- Practical default in ML workflows.
23. **Momentum concept**
- Velocity smooths updates.
- Accelerates along consistent directions.
24. **Nesterov intuition**
- Lookahead gradient correction.
- Often better practical convergence.
25. **Adaptive methods overview**
- Adagrad/RMSProp/Adam concept.
- Coordinate-wise step adaptation.
26. **Why Adam works and fails**
- Fast initial progress.
- Generalization caveats in some settings.
27. **Learning-rate schedules**
- Step/cosine/exponential decay.
- Schedule can dominate convergence quality.
28. **Warmup and cooldown ideas**
- Stabilize early optimization.
- Refine late-stage convergence.
29. **Gradient clipping**
- Control exploding gradients.
- Common in deep sequence models.
30. **Normalization and scaling**
- Feature/target scaling helps optimization.
- Improves conditioning and update balance.
31. **Regularization as optimization shaping**
- Adds penalty curvature.
- Controls complexity and improves generalization.
32. **L2 penalty geometry**
- Shrinks parameter magnitude smoothly.
- Equivalent to ridge in linear models.
33. **L1 penalty geometry**
- Encourages sparse solutions.
- Non-differentiability handled by subgradients/prox.
34. **Constrained optimization preview**
- Feasible sets and constraints.
- Lagrangian idea (high-level only).
35. **Loss landscapes in practice**
- Non-convex surfaces common in NN.
- Multiple minima and saddle regions.
36. **Plateaus and saddle points**
- Slow progress in flat regions.
- Momentum/initialization mitigation.
37. **Initialization effects**
- Starting point impacts path and speed.
- Symmetry breaking importance.
38. **Early stopping as regularizer**
- Stop before overfitting.
- Validation curve driven criterion.
39. **Optimization vs generalization gap**
- Low train loss not enough.
- Need robust validation protocol.
40. **Diagnostics toolkit**
- Loss curves and gradient norms.
- Parameter histograms and learning-rate sweeps.
41. **Case sketch: linear regression training**
- Observe convergence under different eta.
- Relate to condition number.
42. **Case sketch: logistic objective**
- Classification loss dynamics.
- Decision-boundary stabilization.
43. **Case sketch: scaled vs unscaled features**
- Same model, different convergence speed.
- Demonstrates conditioning effect.
44. **Common pitfalls checklist**
- Data leakage, wrong objective, poor scaling.
- Misread metrics and unstable updates.
45. **Mini concept quiz**
- Select correct mitigation for failure mode.
- Exam-style reasoning drill.
46. **Lecture vs exercise split**
- Conceptual theorems in lecture.
- Algorithm implementation in exercise.
47. **Exercise task 1 scaffold**
- Implement GD for MSE objective.
- Compare fixed vs decayed learning rate.
48. **Exercise task 2 scaffold**
- Analyze conditioning with synthetic data.
- Apply feature scaling and compare.
49. **Exercise task 3 scaffold**
- Logistic-loss optimization experiment.
- Inspect calibration vs loss improvements.
50. **Summary + references + Unit 4 bridge**
- Key takeaways on calculus+optimization.
- Transition to probability foundations.
