# MG Realignment — 2026-05-09

Replaced Units 02–04 with QM/QC/thermodynamics/classical-atomistic content
because the SS26 cohort has insufficient physics background for the
originally planned topics. Source PDF: Hiemer, *Quantum Mechanics and
Quantum Chemistry* (2026-05-05). Unit 1 already covers PDF §2 (history),
§3.1 (hydrogen), and partial §3.2 (postulates 1–4); Units 2–4 split the
remaining ~46 pages.

## New unit titles + content

| Unit | Slug | Title | PDF coverage |
|---|---|---|---|
| 02 | `02_qm_postulates_and_atoms` | QM Postulates, Solvable Systems, Multi-Electron Atoms | §3.2 (rest), §3.3, §3.4, §3.5, §4 |
| 03 | `03_quantum_chemistry_methods` | Quantum Chemistry Methods (HF, MP, CC, DFT) | §5 |
| 04 | `04_thermodynamics_and_atomistic_simulation` | Thermodynamics, Statistical Mechanics & Classical Atomistic Simulation | §6, §7 |

## Where the displaced topics go

The previous Units 02–04 were skeleton-only (50-slide stubs, no
real content) on three topics. Disposition:

- **Simulation methods as data generators** (old U2): mostly absorbed
  by the new U4 (which covers MD/MC properly). The "FEM/continuum"
  framing and "ML-pipeline boilerplate" (provenance, FAIR, leakage,
  metrics, model cards) are dropped — covered by MFML and ML-PC.
- **Materials databases & thermodynamic targets** (old U3): keep the
  databases (MP/OQMD/AFLOW/NOMAD), formation energy, energy-above-hull,
  and convex-hull material — fold into MG Unit 12 (Uncertainty-Aware
  Discovery) where they are needed for the discovery loop. Drop
  CIF/POSCAR file-format details.
- **Classical descriptors → learned representations** (old U4):
  fold the descriptor families (Magpie, matminer, RDF/coordination
  stats) into MG Unit 6 (Local Atomic Environments). The "learned
  representations" half is already covered by MG Unit 9 and MFML
  Units 5, 9.

## Triad coordination

- **MFML** already covers the ML-pipeline foundations (loss functions,
  CV, leakage, regularization, bias-variance, latent spaces, GPs).
- **ML-PC** already covers experimental data quality, leakage,
  process windows, transfer learning, time-series, etc.
- **MG** is therefore now free to focus on the materials-specific
  physics, simulation methods, structure representations, and
  discovery loop without re-teaching pipeline mechanics.
