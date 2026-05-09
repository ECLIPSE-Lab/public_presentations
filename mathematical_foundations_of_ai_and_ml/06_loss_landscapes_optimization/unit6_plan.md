# Unit 6 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: 2 semesters math, 1 physics, some chemistry
- Assumed: gradient descent basics from Unit 5, matrix calculus, eigenvalue decomposition
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 6)
By the end of the unit, students can:
1. Describe the loss landscape as a high-dimensional surface and identify critical point types (minima, saddle points, plateaus).
2. Explain how the Hessian matrix and its eigenvalues characterize curvature and conditioning of the landscape.
3. Derive why saddle points dominate over local minima in high-dimensional parameter spaces.
4. Compare momentum, AdaGrad, RMSProp, and ADAM in terms of their mechanisms for handling ill-conditioned landscapes.
5. Relate flat vs sharp minima to generalization and explain the role of learning rate schedules and batch size.

## Book-aligned content mapping (priority order)
1. Neuer (2024): Ch. 4.4.6 (hyperparameters of learning), Ch. 4.5.5 (optimization as hyperparameter: SGD, ADAM, AdaGrad, RMSProp).
2. McClarren (2021): Ch. 5.2.2 (issues with gradient descent, momentum, learning rate schedules, SGD).
3. Bishop (2006): Ch. 3.4 (the Laplace approximation / Hessian), Ch. 3.5 (Bayesian model comparison).
4. Goodfellow, Bengio & Courville (2016): Ch. 8 (optimization for deep learning — saddle points, ill-conditioning, adaptive methods).
5. Murphy (2012): Ch. 8.3–8.5 (gradient-based optimization, second-order methods).

## 90-minute structure
- 0–10 min: Recap of gradient descent + motivation for landscape geometry
- 10–25 min: Loss landscape as surface, Hessian matrix, eigenvalues and curvature
- 25–40 min: Conditioning, ill-conditioned landscapes, saddle points in high dimensions
- 40–55 min: Momentum (physics analogy), Nesterov accelerated gradient
- 55–75 min: Adaptive learning rates — AdaGrad, RMSProp, ADAM derivation and comparison
- 75–85 min: Flat vs sharp minima, learning rate schedules, batch size effects on generalization
- 85–90 min: Summary + exercise handoff

## Exercise (90 min)
- Implement vanilla GD, momentum, and ADAM on a 2D ill-conditioned quadratic (visualize trajectories)
- Experiment with learning rate: too large (divergence), too small (slow convergence), well-tuned
- Compare convergence on a well-conditioned vs ill-conditioned loss surface (eigenvalue ratio experiment)
- Visualize the effect of batch size on gradient noise and trajectory smoothness
- Bonus: implement a cosine annealing learning rate schedule and compare to constant learning rate

## Assessment alignment
- Written exam prep: Hessian eigenvalue interpretation, ADAM update rule derivation, conditioning number.
- Every unit introduces 3–5 exam-relevant "must-know" statements.

## Slide production notes
- Style: follow data_science_for_em revealjs pattern
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
