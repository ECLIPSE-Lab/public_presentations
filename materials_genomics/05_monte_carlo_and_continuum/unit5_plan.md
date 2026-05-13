# Unit 5 Plan — Materials Genomics

## Unit title
Monte Carlo Sampling & Continuum Mechanics

## Unit focus
Close out the simulation half of the course. Cover Monte Carlo as the sampling
counterpart to MD, then move up the length-scale ladder to continuum mechanics
and its discretisation (FDM, FEM, FV). PDF source: Hiemer, *Quantum Mechanics
and Quantum Chemistry* §7.4 and §8.1–8.3.

## Book-chapter anchors used for scaffold design
- Hiemer (2026) §7.4 — Monte Carlo (Boltzmann ratios, detailed balance,
  Metropolis, ergodicity)
- Hiemer (2026) §8.1 — Balance laws (continuity in 1D / multi-D,
  constitutive laws)
- Hiemer (2026) §8.2 — Finite Difference Method (Taylor expansion,
  forward/backward/central differences, 1D Laplace example)
- Hiemer (2026) §8.3 — Excourse: Finite Element / Finite Volume methods
- Frenkel & Smit, *Understanding Molecular Simulation*, Ch. 3–4 (MC)
- Allen & Tildesley, *Computer Simulation of Liquids*, Ch. 4 (MC)
- Strang, *Computational Science and Engineering*, Ch. 6 (FDM / FEM)

## Learning objectives
By the end of Unit 5, students can:
1. Explain why naive integration and uniform sampling both fail in many-body
   phase space, and why importance sampling solves the problem.
2. State and apply detailed balance to design and audit a Metropolis MC move.
3. Diagnose poor MC convergence using equilibration, autocorrelation, and
   acceptance-rate criteria.
4. Derive a 1D continuity equation from a flux balance and identify the
   constitutive law required to close the system.
5. Build a Taylor-expansion-based FDM stencil for first and second
   derivatives and use it to solve a simple BVP.
6. Sketch the weak form behind FEM and explain when FEM, FV, or FDM is the
   right tool for a given geometry / conservation requirement.

## 90-minute lecture structure
- 0–5 min: recap of Units 1–4, position of Unit 5
- 5–45 min: Part I — Monte Carlo sampling (sampling problem, importance
  sampling, Markov chains, detailed balance, Metropolis, ergodicity,
  workflow, move design, ensembles)
- 45–70 min: Part II — Continuum mechanics (balance laws, constitutive
  relations, isotropy/anisotropy)
- 70–85 min: Part III — Discretisation (Taylor stencils, FDM example,
  boundary conditions, time stepping, weak form, FEM, FV)
- 85–90 min: closing — where ML plugs in, takeaways, outlook to Unit 6

## Exercise (90 min)
- Implement a 1D Lennard-Jones Metropolis MC sampler in Python; produce
  $\langle E\rangle$ and the radial distribution function.
- Compare convergence at three step sizes and report acceptance rate and
  autocorrelation time.
- Solve the 1D steady-state diffusion equation with FDM for two boundary
  conditions; verify against the analytic solution.
- Brief write-up: where would each tool fit in a multiscale materials
  workflow?

## Required chapter files
- Hiemer §7.4, §8.1–8.3 (already covered in lecture PDF)
- Frenkel & Smit — Ch. 3 (Metropolis), Ch. 5 (other ensembles)
- Strang Ch. 6 — FDM stencils, BVPs, FEM weak form
- Optional: McClarren ch. on physical-system surrogates for the "where ML
  plugs in" closing slide

## Cross-book summary target
- Use Hiemer as the canonical source for derivations and notation alignment
  with the rest of the realigned QM/QC track (Units 2–4).
- Use Frenkel & Smit for the move-design vocabulary (cluster moves, swap
  moves, $\mu$VT) that the lecture PDF treats only briefly.
- Use Strang only for the FEM / weak-form intuition — no deep functional
  analysis.
- Keep modern context (MACE-style MLIPs as MD/MC engines, neural samplers,
  neural PDE solvers) visible in the closing slides but treat them as
  forward references to MG Units 6–14.

## 50-slide strategy
- Slides 1–5: where we stand, the pivot, the roadmap, the multiscale map.
- Slides 6–25: Monte Carlo (sampling problem → importance sampling →
  Markov chains → detailed balance → Metropolis → ergodicity → workflow
  → moves → ensembles → kMC).
- Slides 26–34: continuum balance laws and constitutive relations.
- Slides 35–46: discretisation — FDM, weak form, FEM, FV.
- Slides 47–51: ML connection, takeaways, outlook, continue / references.

## Website summary update
- Heading: `#### Week 5 – Monte Carlo sampling and continuum mechanics (12.05.2026)`
- Add a summary that:
  - Closes the simulation half of MG (electronic → finite-T → multiscale).
  - Justifies why MC is the natural complement to MD, especially for
    equilibrium properties and rare-event sampling.
  - Motivates continuum mechanics as the rung above MD on the length
    scale ladder, and frames FDM/FEM as the discretisation toolkit.
  - Points forward to representations and graph-based methods in Unit 6.
