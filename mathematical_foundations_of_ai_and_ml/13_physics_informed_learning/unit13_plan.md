# Unit 13 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: Units 1--12 completed (optimization, neural networks, regularization, kernels, etc.)
- Assumed: automatic differentiation basics (Unit on NNs), loss function design, ODE/PDE awareness from physics courses
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 13)
By the end of the unit, students can:
1. Explain the philosophy of physics-informed machine learning and why injecting prior knowledge reduces data requirements.
2. Design data enrichment pipelines using known physical transformations (FFT, wavelets, derivatives).
3. Use automatic differentiation to embed analytical constraints into a neural network's loss function.
4. Formulate the composite PINN loss J = J_data + lambda * J_physics and interpret the role of lambda.
5. Apply the Lagaris substitution to enforce initial/boundary conditions in neural ODE/PDE solvers.
6. Relate Occam's razor and information-theoretic model selection to the benefits of physical priors.

## Book-aligned content mapping (priority order)
1. Neuer (2024): Ch. 6.1 (PINN philosophy, motivating example, statistical balance), Ch. 6.2 (data enrichment, expert knowledge objects, automatic optimization of enrichment), Ch. 6.3 (embedding analytical expressions, automatic differentiation, integration, ODE solving, Lagaris substitution).
2. Karniadakis et al. (2021): Nature Reviews Physics survey on PIML taxonomy.
3. Raissi, Perdikaris & Karniadakis (2019): PINN framework for forward/inverse PDE problems.
4. Lagaris, Likas & Fotiadis (1998): foundational NN-based ODE/PDE solvers with boundary condition substitution.

## 90-minute structure
- 0--10 min: Why physics-informed learning? Motivation from data scarcity and explainability
- 10--25 min: Data enrichment strategies (FFT, wavelets, derivatives, histograms, PCA on spectral data)
- 25--40 min: Automatic differentiation for physical constraints (GradientTape / autograd mechanics)
- 40--55 min: PINN loss function design: J_data + lambda * J_physics; balancing data fit and physics residual
- 55--70 min: Boundary conditions via Lagaris substitution; neural integrators and ODE solvers
- 70--80 min: Occam's razor, information-theoretic justification, small-data advantages, explainability
- 80--85 min: Limitations, open challenges, and when PINNs fail
- 85--90 min: Summary + exercise handoff + reading assignment for Unit 14

## Exercise (90 min)
- Part A: Implement a constrained NN regression where a known symmetry or conservation law is enforced via penalty in the loss function.
- Part B: Build a minimal PINN that solves a first-order ODE (e.g., x' = x) using automatic differentiation and the Lagaris substitution; compare convergence to the analytical solution.
- Part C (Bonus): Add a data-fit term with noisy observations and explore the effect of lambda on the physics-data tradeoff.

## Assessment alignment
- Written exam prep: students must be able to write the PINN loss, explain Lagaris substitution, and distinguish data enrichment from loss-based constraint embedding.
- Every unit introduces 3--5 exam-relevant "must-know" statements.

## Slide production notes
- Style: follow data_science_for_em revealjs pattern (consistent with Unit 1)
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
- Use code snippets (TensorFlow/PyTorch style) for auto-diff and PINN loss illustrations
