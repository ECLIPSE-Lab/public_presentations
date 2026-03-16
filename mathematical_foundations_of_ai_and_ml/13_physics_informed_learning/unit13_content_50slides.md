# MFML Unit 13 — 50-Slide Content Pack (English)

## Unit theme
**Physics-Informed Machine Learning: Embedding Domain Knowledge into Models**

## Core source mapping (book-priority aligned)
- **Neuer (2024)**: Ch. 6.1 (PINN philosophy, statistical balance), Ch. 6.2 (data enrichment, expert knowledge objects), Ch. 6.3 (embedding analytical expressions, automatic differentiation, Lagaris substitution).
- **Karniadakis et al. (2021)**: PIML taxonomy (Nature Reviews Physics).
- **Raissi, Perdikaris & Karniadakis (2019)**: PINN framework for PDE problems.
- **Lagaris, Likas & Fotiadis (1998)**: NN-based ODE/PDE solvers with boundary condition substitution.

---

## Slide-by-slide content (target: 50)

### Block A — Motivation and philosophy (Slides 1-10)
1. **Title + Unit 13 positioning**
   - From uncertainty quantification (Unit 12) to reducing uncertainty by embedding physics.
2. **Learning outcomes for Unit 13**
   - PIML philosophy, data enrichment, auto-diff constraints, PINN loss, Lagaris substitution, Occam's razor.
3. **The data scarcity problem**
   - In engineering, labeled data is expensive (destructive testing, slow simulations). Physics provides free knowledge.
4. **What is physics-informed machine learning?**
   - Incorporating known physical laws, symmetries, and constraints into the ML model or training process.
5. **The PIML spectrum**
   - Data-only → data-enriched → physics-constrained loss → physics-embedded architecture → pure physics simulation.
6. **Why physics helps: inductive bias**
   - Physical constraints restrict the hypothesis space → fewer plausible models → less data needed.
7. **Connection to regularization (Unit 7)**
   - Regularization restricts model complexity generally. Physics-informed constraints restrict it specifically and meaningfully.
8. **The statistical balance argument (Neuer)**
   - More information from physics → less information needed from data. The two sources complement each other.
9. **Taxonomy of PIML approaches**
   - (a) Data enrichment, (b) loss-function constraints, (c) architectural constraints, (d) hybrid simulation.
10. **Roadmap of today's 90 min**
    - Data enrichment → auto-diff → PINN loss → Lagaris substitution → Occam's razor → limitations.

### Block B — Data enrichment strategies (Slides 11-18)
11. **Data enrichment: the idea**
    - Transform raw data using known physics to create additional, more informative input features.
12. **FFT-based enrichment**
    - Compute the Fourier spectrum of time-series data → frequency-domain features capture periodicity and resonance.
13. **Wavelet-based enrichment**
    - Wavelet transforms capture time-frequency information simultaneously — useful for transient phenomena.
14. **Derivative-based enrichment**
    - Compute numerical derivatives of input signals → rates of change encode dynamics.
15. **Histogram and statistical enrichment**
    - Compute histograms, moments, or percentiles of input distributions → summarize variability.
16. **PCA on spectral data**
    - Apply PCA to high-dimensional spectra → low-dimensional features that capture dominant variation.
17. **Expert knowledge objects (Neuer)**
    - Domain experts specify transformations: physical units, dimensionless groups, known functional relationships.
18. **Checkpoint: data enrichment**
    - "Your raw features are temperature vs time. What enriched features would you add?" — derivatives (cooling rate), FFT (periodicity), peak temperature.

### Block C — Automatic differentiation and physics constraints (Slides 19-30)
19. **Automatic differentiation recap**
    - Frameworks (PyTorch, JAX) compute exact derivatives of any computation graph — Unit 5 backprop in action.
20. **Using auto-diff for physics constraints**
    - Key insight: we can differentiate the NN output w.r.t. its inputs (not just its parameters).
21. **Example: enforcing smoothness**
    - Penalize large second derivatives: L_smooth = integral (d^2 f / dx^2)^2 dx. Computed via auto-diff.
22. **Example: conservation law**
    - If mass is conserved: d(rho)/dt + div(rho v) = 0. Add this as a penalty to the loss.
23. **The PINN loss function**
    - Composite loss: J = J_data + lambda * J_physics.
24. **J_data: data fidelity**
    - J_data = (1/N) sum ||y_i - f_theta(x_i)||^2. Standard MSE on observed data.
25. **J_physics: physics residual**
    - J_physics = (1/M) sum ||R(f_theta, x_j)||^2 where R is the residual of the governing equation.
26. **Collocation points**
    - Evaluate J_physics at M collocation points sampled across the domain — no labels needed.
27. **The role of lambda**
    - lambda balances data fit vs physics compliance. High lambda → physics dominates. Low lambda → data dominates.
28. **Choosing lambda**
    - Too high: model satisfies physics but ignores data. Too low: model fits data but violates physics.
    - Adaptive strategies: adjust lambda during training based on relative loss magnitudes.
29. **PINN architecture**
    - Standard fully-connected NN. Inputs: spatial/temporal coordinates. Output: physical field (temperature, displacement).
30. **Checkpoint: PINN loss design**
    - "Write the PINN loss for heat diffusion: dT/dt = alpha * d^2T/dx^2 with noisy measurements."

### Block D — Boundary conditions and the Lagaris substitution (Slides 31-38)
31. **Boundary conditions in PDE problems**
    - Physical problems require BCs: f(0) = a (Dirichlet), df/dx(0) = b (Neumann).
32. **Soft BC enforcement**
    - Add BC violation to the loss: J_BC = ||f_theta(x_boundary) - BC_value||^2.
33. **Hard BC enforcement: the Lagaris substitution**
    - Construct the solution form so that BCs are satisfied by construction: f(x) = A(x) + B(x) * NN(x).
34. **Lagaris substitution: example**
    - For f(0) = a: write f(x) = a + x * NN(x). Then f(0) = a automatically, regardless of NN.
35. **Lagaris for ODE: x' = x, x(0) = 1**
    - Trial solution: x(t) = 1 + t * NN(t). The NN only needs to approximate the deviation from the IC.
36. **Advantages of hard enforcement**
    - BCs are exactly satisfied — no penalty term needed, no violation possible.
    - Reduces the search space: the NN only learns the unknown part.
37. **Neural integrators and ODE solvers**
    - Use NN to parameterize the solution of an ODE; train by minimizing the ODE residual.
    - The NN is a continuous, differentiable solution — no time-stepping discretization.
38. **Checkpoint: Lagaris design**
    - "Construct a trial solution for f''(x) = g(x), f(0) = 0, f(1) = 0." — f(x) = x(1-x) * NN(x).

### Block E — Occam's razor, limitations, and summary (Slides 39-50)
39. **Occam's razor and model selection**
    - Prefer the simplest model that explains the data. Physics constraints simplify by restricting the hypothesis space.
40. **Information-theoretic perspective**
    - MDL (Minimum Description Length): physics priors reduce the description length needed for the model.
41. **Small-data advantage of PINNs**
    - With 10 data points, a PINN enforcing the ODE can outperform a standard NN with 1000 points.
42. **Explainability benefit**
    - Physics-constrained models are more interpretable: the model respects known laws by design.
43. **Limitations: when PINNs fail**
    - Incorrect physics: wrong PDE → wrong solution. Multi-scale problems: conflicting length scales.
44. **Limitations: optimization challenges**
    - Balancing J_data and J_physics is hard. Stiff PDEs cause training instability. Spectral bias in NNs.
45. **Limitations: computational cost**
    - Auto-diff through PDE residuals is expensive. Collocation point count scales with domain size.
46. **Beyond PINNs: architectural constraints**
    - Equivariant networks (symmetry built in), Hamiltonian NNs (energy conservation), neural operators.
47. **Lecture-essential vs exercise content split**
    - Lecture: PIML philosophy, data enrichment, PINN loss, Lagaris substitution, Occam's razor.
    - Exercise: constrained NN regression, minimal PINN for ODE, data-physics tradeoff exploration.
48. **Exercise setup summary**
    - Part A: enforce symmetry via loss penalty. Part B: PINN for x' = x with Lagaris. Part C (bonus): noisy data + lambda sweep.
49. **Exam-aligned summary: 10 must-know statements**
    1. Physics-informed ML embeds domain knowledge to reduce data requirements.
    2. Data enrichment adds physics-derived features to raw inputs.
    3. The PINN loss is J = J_data + lambda * J_physics.
    4. J_physics is the PDE/ODE residual evaluated at collocation points.
    5. lambda balances data fidelity against physics compliance.
    6. The Lagaris substitution enforces boundary conditions exactly by construction.
    7. Auto-diff computes exact derivatives of NN outputs w.r.t. inputs for physics residuals.
    8. Physics constraints act as informed regularization, reducing the hypothesis space.
    9. PINNs excel in small-data regimes where physics provides complementary information.
    10. Incorrect physics in PINNs leads to confidently wrong predictions.
50. **References + reading assignment for next unit**
    - Required: Neuer Ch. 6.1–6.3.
    - Optional: Karniadakis et al. (2021), Raissi et al. (2019), Lagaris et al. (1998).
    - Next unit: Explainability, Limits, and Trust — the final lecture.

---

## Reusable equations (for slide math boxes)
- PINN loss: \\(J = J_{\text{data}} + \lambda \cdot J_{\text{physics}}\\)
- Data loss: \\(J_{\text{data}} = \frac{1}{N}\sum_{i=1}^N \|y_i - f_\theta(x_i)\|^2\\)
- Physics loss: \\(J_{\text{physics}} = \frac{1}{M}\sum_{j=1}^M \|\mathcal{R}(f_\theta, x_j)\|^2\\)
- Lagaris substitution: \\(f(x) = A(x) + B(x)\cdot\text{NN}(x)\\), where \\(A\\) satisfies BCs
- Heat equation residual: \\(\mathcal{R} = \frac{\partial T}{\partial t} - \alpha\frac{\partial^2 T}{\partial x^2}\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: PIML philosophy, data enrichment examples, PINN loss derivation, Lagaris substitution, Occam's razor, limitations.
- **Exercise**: constrained NN regression, PINN for first-order ODE, lambda sweep with noisy data.
