# Unit 12 Plan — Materials Genomics

## Unit title
Generative Models & Inverse Design

## Unit focus
Treat generative modelling as the engine of modern materials discovery.
Cover crystal-aware diffusion (CDVAE, DiffCSP, MatterGen), flow matching
(FlowMM), autoregressive / LLM-style models (CrystaLLM), conditioning
and constraints, and the downstream filtering funnel that turns
generated candidates into validated discoveries (MLIP relax → DFT →
UQ → autonomous synthesis).

## Replan note
This unit replaces the original "Clustering vs Discovery in Materials
Spaces" content following the 2026-05-13 realignment. Clustering as a
standalone discovery story has been folded into the active-learning
section of Unit 13. Generative models are the headline 2023–2025
development that was missing from the original MG curriculum.

## Anchors used for scaffold design
- Xie et al. 2022 — CDVAE (ICLR)
- Jiao et al. 2023 — DiffCSP
- Jiao et al. 2024 — DiffCSP++
- Zeni et al. 2024 — MatterGen (Microsoft Research)
- Lipman et al. 2023 — Flow Matching
- Miller et al. 2024 — FlowMM
- Antunes et al. 2024 — CrystaLLM
- Merchant et al. 2023 — GNoME (Nature)
- Materials Project / OQMD / Alexandria / ICSD / GNoME data releases
- Sandfeld ch. 2.2 (materials data science framing)

## Learning objectives
By the end of Unit 12, students can:
1. State the difference between forward prediction and inverse design,
   and explain why inverse design needs a generative model.
2. Describe the structural variables of a crystal that any generator
   must produce (composition, lattice, fractional coordinates, optional
   symmetry).
3. Sketch the forward / reverse process of a diffusion model and the
   score-based reformulation.
4. Compare diffusion, flow matching, and autoregressive families on
   sample cost, conditioning flexibility, and current SUN rates.
5. Apply S.U.N. (stable / unique / novel) plus the broader
   validity-novelty-uniqueness-stability-fidelity criteria when
   reading a generative-models paper.
6. Walk through the candidate-filtering funnel (generate → MLIP
   relax → DFT → UQ → synthesise) and justify why each stage is
   needed.
7. Explain how generative models couple to universal MLIPs (Unit 7)
   and to uncertainty-aware discovery (Unit 13) in modern lab-in-the-loop
   platforms.

## 90-minute lecture structure
- 0–10 min: recap of forward modelling (Units 6–11); pivot to inverse
- 10–25 min: Part I — foundations, evaluation metrics, data landscape
- 25–50 min: Part II — diffusion models for crystals (CDVAE, DiffCSP,
  MatterGen)
- 50–65 min: Part III — flow matching, autoregressive / LLM-style
- 65–75 min: Part IV — conditioning, classifier-free guidance,
  multi-objective
- 75–88 min: Part V — downstream filtering funnel, GNoME, lab loops
- 88–90 min: closing — open challenges, takeaways, link to Unit 13

## Exercise (90 min)
- Sample 1000 candidate structures from a pretrained MatterGen or
  DiffCSP checkpoint conditioned on a target property
  (e.g. bandgap ≈ 2 eV).
- Run the candidates through a MACE-MP-0 relaxation and report S.U.N.
  rates against the Materials Project hull.
- Pick the top 10 candidates by predicted property fit + MLIP energy;
  produce a short proposal-style write-up describing what DFT
  validation you would run next and why.

## Required references
- MatterGen paper + open-source repository
- DiffCSP / DiffCSP++ papers
- CrystaLLM and FlowMM papers
- GNoME (Nature 2023) for the empirical-scale framing
- Materials Project / Alexandria release notes (data caveats)
- Unit 7 deck (MLIPs) and Unit 13 deck (UQ) for cross-links

## 50-slide strategy
- Slides 1–5: where we stand, pivot, roadmap, generative landscape.
- Slides 6–13: Part I — foundations (forward vs inverse, crystal as
  data, evaluation, S.U.N., data sources).
- Slides 14–24: Part II — diffusion models, from CDVAE through
  MatterGen, with limitations.
- Slides 25–31: Part III — flow matching, FlowMM, autoregressive,
  CrystaLLM, legacy VAEs/GANs, comparison table.
- Slides 32–36: Part IV — conditioning and constraints.
- Slides 37–44: Part V — downstream filtering funnel, GNoME, active
  learning, lab automation.
- Slides 45–50: closing — open challenges, takeaways, Unit 13
  forward link, continue, references.

## Website summary update
- Heading: `#### Week 12 – Generative models & inverse design (30.06.2026)`
- Summary should:
  - Position generative modelling as the headline 2023–2025 development
    in materials genomics.
  - Stress that generation is one stage in a longer funnel and that the
    rest of the funnel (MLIPs, DFT, UQ, lab automation) is what
    determines actual discovery rate.
  - Pre-announce the Unit 13 link: generation × UQ closes the loop.

## Folder rename
The folder is being renamed from
`12_clustering_vs_discovery_in_materials_spaces` to
`12_generative_models_and_inverse_design` as part of the 2026-05-13
realignment. Slide URL in the manuscript / website must be updated
accordingly.
