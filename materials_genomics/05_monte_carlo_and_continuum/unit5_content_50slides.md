# Unit 5 — 50-slide scaffold

| # | Slide title | One-line content |
|---|---|---|
| 1 | Where We Stand (section title) | Anchor against Units 1–4 |
| 2 | Recap of Units 1–4 | Quantum → potentials → MD; we have $V$ but no equilibrium sampler that handles barriers |
| 3 | What Unit 5 Adds — The Pivot | MC for sampling + continuum for length-scale jump |
| 4 | Lecture Roadmap | Part I MC, Part II continuum, Part III discretisation |
| 5 | A Multiscale View | length/time-scale table mapping DFT / MD / MC / FEM |
| 6 | Part I — Monte Carlo Sampling (divider) | — |
| 7 | The Sampling Problem | Ensemble average as a high-D integral over phase space |
| 8 | Why Brute Force Fails | $n^{3N}$ grid cost = curse of dimensionality |
| 9 | Why Uniform Sampling Fails | Boltzmann mass concentrates in exponentially small region |
| 10 | Importance Sampling — The Idea | Sample $p$, collapse to simple average |
| 11 | The Partition Function Problem | $Z$ intractable; ratios save us |
| 12 | Markov Chains — Setup | Memoryless, transition kernel, ergodicity |
| 13 | Stationary Distributions | Global vs detailed balance |
| 14 | Detailed Balance | Pair-wise flux cancellation |
| 15 | Proposal × Acceptance | Decompose $P=P_q P_a$, symmetric case |
| 16 | Deriving the Metropolis Criterion | Acceptance ratio $\to$ min[1, $e^{-\Delta E/k_BT}$] |
| 17 | The Metropolis Algorithm | Six-step recipe |
| 18 | Why Metropolis Works | Detailed-balance proof, never need $Z$ |
| 19 | Ergodicity | Move-design implications, multi-chain diagnostics |
| 20 | MC Workflow | Burn-in, autocorrelation, effective sample size |
| 21 | Tuning Step Size | 20–50% acceptance sweet spot |
| 22 | Move Catalogue | displacement, swap, volume, insertion, configurational bias, hybrid MC |
| 23 | Ensembles in MC | NVT, NPT, $\mu$VT, Gibbs |
| 24 | MC vs MD | Strengths matrix; complementary, not competitive |
| 25 | Metropolis–Hastings & Beyond | Asymmetric proposals, kinetic MC, modern samplers |
| 26 | Part II — Continuum Mechanics (divider) | — |
| 27 | Why Continuum? | Above ~1 µm, atoms become impractical |
| 28 | 1D Balance Setup | Interval $[x,x+dx]$, flux $q$, source $s$ |
| 29 | Stationary Continuity Equation | $s=\partial q/\partial x$ |
| 30 | Transient Continuity Equation | Add $\partial \rho/\partial t$ |
| 31 | Multi-Dimensional Form | $\partial_t\rho = -\nabla\cdot\mathbf{q} + s$ |
| 32 | Constitutive Laws | Fick, Fourier, Darcy |
| 33 | Isotropy vs Anisotropy | Scalar coefficient vs material tensor $\mathbf{A}$ |
| 34 | Part III — Discretization (divider) | — |
| 35 | Why Discretize? | Real geometries demand algebraic systems |
| 36 | Taylor Expansion | Foundation of all FDM stencils |
| 37 | Forward & Backward Differences | $\mathcal{O}(\delta)$ accurate, useful at boundaries |
| 38 | Central Difference | $\mathcal{O}(\delta^2)$, symmetric |
| 39 | Second-Derivative Stencil | $(\varphi_+ - 2\varphi + \varphi_-)/\delta^2$ |
| 40 | Worked Example: 1D Laplace | 3 nodes, $\delta=2$, $\varphi_2=1/2$ |
| 41 | Boundary Conditions | Dirichlet / Neumann / Robin |
| 42 | Time Stepping & CFL | Explicit / implicit / Crank–Nicolson, stability |
| 43 | Toward FEM: Weak Form | Multiply by test function, integrate by parts |
| 44 | Finite Element Method | Mesh + shape functions + sparse stiffness |
| 45 | Finite Volumes Variant | Local flux balance + divergence theorem |
| 46 | Closing (divider) | — |
| 47 | Where ML Plugs In | MLIPs, neural samplers, neural PDE solvers |
| 48 | Key Takeaways | Six bullets, the scaffolding of the simulation half |
| 49 | Outlook — Unit 6 | Graph-based crystal representations |
| 50 | Continue | Prev/next links |
| 51 | References | Bibliography |
