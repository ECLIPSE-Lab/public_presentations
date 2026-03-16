# MFML Unit 6 — 50-Slide Content Pack (English)

## Unit theme
**Loss Landscapes and Optimization Behavior**

## Core source mapping (book-priority aligned)
- **Neuer (2024)**: Ch. 4.4.6 (hyperparameters), Ch. 4.5.5 (optimization as hyperparameter: SGD, ADAM, AdaGrad, RMSProp).
- **McClarren (2021)**: Ch. 5.2.2 (gradient descent issues, momentum, learning rate sensitivity, SGD and batching).
- **Bishop (2006)**: Ch. 3.4 (Laplace approximation / Hessian role), Ch. 3.5 (Bayesian model comparison and evidence).
- **Goodfellow et al. (2016)**: Ch. 8 (optimization for training deep models — ill-conditioning, saddle points, adaptive methods).
- **Murphy (2012)**: Ch. 8.3–8.5 (gradient-based optimization, second-order methods, practical tips).

---

## Slide-by-slide content (target: 50)

### Block A — Recap and landscape geometry (Slides 1–10)
1. **Title + Unit 6 positioning**
   - Loss landscapes connect gradient computation (Unit 5) to generalization (Unit 7).
2. **Recap: gradient descent as parameter update**
   - Update rule, learning rate, convergence requirement.
3. **Learning outcomes for Unit 6**
   - Hessian interpretation, optimizer comparison, landscape-generalization link.
4. **The loss landscape metaphor**
   - Cost function as high-dimensional surface over parameter space.
5. **Visualizing 1D and 2D loss surfaces**
   - Slices through high-dimensional surfaces; local minima, saddle points, plateaus.
6. **Critical points: gradient equals zero**
   - Minima, maxima, and saddle points classified by second-order information.
7. **The Hessian matrix: definition**
   - Matrix of second-order partial derivatives of the loss.
8. **Eigenvalues of the Hessian**
   - Large eigenvalues = steep curvature; small eigenvalues = flat directions.
9. **Conditioning and the condition number**
   - Ratio of largest to smallest eigenvalue; ill-conditioning causes oscillation.
10. **Geometric interpretation: elliptical contours**
    - Well-conditioned (circular) vs ill-conditioned (elongated) level sets.

### Block B — Saddle points and pathologies (Slides 11–20)
11. **Why local minima are rare in high dimensions**
    - For a critical point to be a local minimum, all eigenvalues must be positive.
12. **Saddle points dominate the landscape**
    - Random matrix theory: probability of all-positive eigenvalues decreases exponentially with dimension.
13. **Saddle point dynamics under gradient descent**
    - Gradient near zero but escape is possible along negative-curvature directions.
14. **Plateaus and vanishing gradients**
    - Flat regions where gradient magnitude is negligible; training stalls.
15. **Loss surface of linear networks (analytical case)**
    - Even simple models exhibit non-convex landscapes with saddle structure.
16. **Empirical observations: loss surfaces of deep networks**
    - Many local minima have similar loss values; sharp vs flat regions.
17. **Role of overparameterization**
    - More parameters than data points create connected low-loss valleys.
18. **Symmetry and mode connectivity**
    - Permutation symmetry of hidden units creates equivalent minima.
19. **Landscape pathologies summary table**
    - Ill-conditioning, saddle points, plateaus, sharp minima — causes and symptoms.
20. **Checkpoint: identify the pathology**
    - Given training curves, diagnose landscape issue (exam-style reasoning).

### Block C — Momentum and accelerated methods (Slides 21–30)
21. **Vanilla GD on ill-conditioned surface**
    - Oscillation along steep direction, slow progress along flat direction.
22. **Momentum: physics analogy**
    - Parameter update as particle with velocity; accumulate speed in consistent directions.
23. **Momentum update rule**
    - Velocity term, momentum coefficient, damping of oscillations.
24. **Momentum on the elongated bowl**
    - Visualization: reduced oscillation, faster convergence along valley floor.
25. **Nesterov accelerated gradient**
    - "Look-ahead" gradient evaluation; improved convergence rate.
26. **Momentum vs Nesterov: comparison**
    - Convergence speed, overshoot behavior, practical defaults.
27. **Learning rate as the most critical hyperparameter**
    - Too large: divergence; too small: slow convergence; just right: fast convergence.
28. **Learning rate sensitivity demonstration**
    - Training curves for different learning rates on the same problem.
29. **Why a single global learning rate is insufficient**
    - Parameters in different layers/directions experience different curvatures.
30. **Recap: what momentum solves and what it does not**
    - Helps with consistent gradients; does not adapt per-parameter.

### Block D — Adaptive optimizers (Slides 31–42)
31. **Per-parameter learning rates: the core idea**
    - Scale each coordinate by its historical gradient magnitude.
32. **AdaGrad: accumulate squared gradients**
    - Update rule, adaptive scaling, effective learning rate decay.
33. **AdaGrad: strengths and weaknesses**
    - Good for sparse gradients; aggressive decay can kill learning.
34. **RMSProp: exponential moving average fix**
    - Decaying average of squared gradients; prevents monotonic shrinkage.
35. **RMSProp update rule**
    - Decay rate, epsilon for numerical stability, effective step size.
36. **ADAM: combining momentum + adaptive scales**
    - First moment (mean) and second moment (variance) estimates.
37. **ADAM update rule (full derivation)**
    - Bias correction for initialization; hyperparameter defaults.
38. **ADAM as landscape normalizer**
    - Effectively rescales coordinates to equalize curvature across directions.
39. **ADAM variants: AdamW, AMSGrad**
    - Weight decay decoupling; guaranteed convergence fixes.
40. **Optimizer comparison on benchmark surfaces**
    - Side-by-side trajectories: SGD, momentum, RMSProp, ADAM.
41. **When ADAM is not enough**
    - Sharp minima preference, generalization gap vs SGD with momentum.
42. **Practical optimizer selection guide**
    - Default: ADAM for exploration; SGD+momentum for final tuning.

### Block E — Generalization, schedules, and exercise bridge (Slides 43–50)
43. **Flat vs sharp minima**
    - Flat minima are robust to parameter perturbation; sharp minima are fragile.
44. **Connection to generalization**
    - Flat minima correspond to solutions less sensitive to distribution shift.
45. **Learning rate schedules**
    - Step decay, exponential decay, cosine annealing, warm-up strategies.
46. **Batch size and gradient noise**
    - Small batches add noise that helps escape sharp minima; large batches converge to sharper minima.
47. **The learning rate–batch size relationship**
    - Linear scaling rule; effective learning rate interpretation.
48. **Exercise setup: optimizer comparison on 2D surfaces**
    - Implement GD, momentum, ADAM; visualize trajectories on ill-conditioned quadratic.
49. **Exam-aligned summary: 10 must-know statements**
    1. The Hessian eigenvalues determine curvature in each direction.
    2. Condition number measures landscape difficulty for gradient descent.
    3. Saddle points outnumber local minima in high dimensions.
    4. Momentum accumulates velocity to dampen oscillations.
    5. AdaGrad adapts per-parameter but decays too aggressively.
    6. RMSProp fixes AdaGrad with exponential moving average.
    7. ADAM combines momentum with adaptive per-parameter scaling.
    8. Flat minima correlate with better generalization.
    9. Learning rate schedules improve convergence and final performance.
    10. Batch size controls the noise level of gradient estimates.
50. **References + reading assignment for next unit**

---

## Reusable equations (for slide math boxes)
- Gradient descent update: \\(\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta J(\theta^{(t)})\\)
- Hessian matrix: \\(H_{ij} = \frac{\partial^2 J}{\partial \theta_i \partial \theta_j}\\)
- Condition number: \\(\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}\\)
- Momentum update: \\(v^{(t+1)} = \alpha v^{(t)} - \eta \nabla J(\theta^{(t)}), \quad \theta^{(t+1)} = \theta^{(t)} + v^{(t+1)}\\)
- ADAM update: \\(m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2\\)
- ADAM bias-corrected: \\(\hat m_t = \frac{m_t}{1-\beta_1^t}, \quad \hat v_t = \frac{v_t}{1-\beta_2^t}, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat v_t}+\epsilon}\hat m_t\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: Hessian interpretation, optimizer derivations, landscape-generalization theory, flat/sharp minima concept.
- **Exercise**: implementation of optimizers, trajectory visualization, learning rate experiments, batch size ablation.

## Reading assignment after Unit 6
- Neuer Ch. 4.5.5 (optimization variants)
- McClarren Ch. 5.2.2 (gradient descent issues and momentum)
- Goodfellow et al. Ch. 8.1–8.5 (optimization for deep learning)
- Bishop Ch. 3.4 (Hessian and Laplace approximation)
