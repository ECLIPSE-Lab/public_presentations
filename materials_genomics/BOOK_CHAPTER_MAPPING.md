# Materials Genomics — Book & Paper Mapping (post 2026-05-13 realignment)

Each MG unit folder now matches its lecture week number (Units 5–14
were renumbered on 2026-05-13). Below is the source mapping used for
slide scaffolds in each post-Unit-4 unit. For QM/QC units (Units 1–4)
see `REALIGNMENT_2026-05-09.md`.

## Unit 5 — Monte Carlo sampling & continuum mechanics (Week 5)
- Hiemer (2026) §7.4 — Monte Carlo (importance sampling, detailed
  balance, Metropolis)
- Hiemer (2026) §8.1–8.3 — Balance laws, FDM, FEM/FV excourse
- Frenkel & Smit, *Understanding Molecular Simulation*, Ch. 3–5 (MC,
  ensembles, move types)
- Allen & Tildesley, *Computer Simulation of Liquids*, Ch. 4 (MC
  algorithms)
- Strang, *Computational Science and Engineering*, Ch. 6 (FDM/FEM
  intuition)

## Unit 6 — Graph-based crystal representations (Week 6)
- Neuer 4.5.1 / 4.5.3 / 4.5.4 (neuron / activations / training)
- McClarren Ch. 8 (neural surrogates for physical systems)
- Sandfeld 2.2 (materials data structures, domain constraints)
- Xie & Grossman 2018 — CGCNN
- Chen *et al.* 2019 — MEGNet
- Choudhary *et al.* 2021 — ALIGNN
- Batatia *et al.* 2022 — MACE
- Brief recap of classical descriptors: Ward *et al.* 2016 (Magpie),
  Ong *et al.* 2013 (matminer)

## Unit 7 — Local atomic environments & universal MLIPs (Week 8)
- Sandfeld 2.2 (materials feature context)
- Neuer 6.2, 6.3 (domain knowledge in preprocessing / embedded
  expressions)
- McClarren Ch. 4 (feature-to-regression pipeline)
- Behler & Parrinello 2007 — ACSF / Behler–Parrinello NN potentials
- Bartók *et al.* 2010, 2013 — SOAP / GAP
- Batatia *et al.* 2023 — MACE-MP-0 (universal MLIP)
- Chen & Ong 2022 — M3GNet
- Deng *et al.* 2023 — CHGNet
- Yang *et al.* 2024 — MatterSim (universal MLIP)
- Neumann *et al.* 2024 — ORB

## Unit 8 — Regression & generalization in materials data (Week 9)
- Neuer 4.2.2 (regression), 4.5.9 (overfitting / CV)
- McClarren Ch. 4 + Ch. 6 (regression + model selection)
- Bishop 3.1–3.3 (bias / variance, linear-model depth)
- Sandfeld 2.4 (split design and chemical leakage)
- Riebesell *et al.* 2024 — Matbench Discovery benchmarks
- Bartel *et al.* 2020 — composition-based leakage warnings

## Unit 9 — Neural networks for materials properties (Week 10)
- Neuer 4.5.1 / 4.5.3–4.5.5 (NN foundations + optimisation)
- McClarren Ch. 8 (NNs for physical systems)
- Bishop 5.1–5.3 (backprop, NN basics)
- Schütt *et al.* 2018 — SchNet
- Choudhary *et al.* 2022 — ALIGNN-FF (force-field variant)
- Foundation-model context: Batatia 2023, Yang 2024 (cf. Unit 7)

## Unit 10 — Representation learning + latent spaces (Week 11, merged)
- Neuer 5.5 (autoencoder, representation motivation)
- Neuer 5.5.1–5.5.3 (topology, latent space, anomaly detection)
- McClarren Ch. 5 (dimension reduction / ROM perspective)
- Bishop 12.1–12.4 (PCA, kernel PCA, nonlinear latent intuition)
- Murphy Ch. 19 — latent linear models
- Pretraining for materials: Choudhary *et al.* 2023 (CrysAtm), Allen
  *et al.* 2024 (MatterAtlas)
- Contrastive: Liu *et al.* 2024 — CrystalCLR
- Folder `11_latent_spaces_of_materials` is retained as supplementary
  reading material rather than a lectured deck.

## Unit 12 — Generative models & inverse design (Week 12)
- Xie *et al.* 2022 — CDVAE (ICLR)
- Jiao *et al.* 2023 — DiffCSP
- Jiao *et al.* 2024 — DiffCSP++ (space-group conditioning)
- Zeni *et al.* 2024 — MatterGen (Microsoft Research)
- Lipman *et al.* 2023 — Flow Matching (foundational)
- Miller *et al.* 2024 — FlowMM
- Antunes *et al.* 2024 — CrystaLLM
- Merchant *et al.* 2023 — GNoME (Nature, DeepMind)
- Materials Project / Alexandria / ICSD / OQMD release notes

## Unit 13 — Uncertainty-aware discovery & GPs (Week 13)
- Neuer 2.2, 6.4 (uncertainty classes + stochastic methods)
- McClarren Ch. 3 (error and uncertainty)
- Murphy Ch. 15 (Gaussian process perspective)
- Rasmussen & Williams, *Gaussian Processes for Machine Learning*, Ch.
  2–3 (regression, kernels)
- Lakshminarayanan *et al.* 2017 — deep ensembles
- Sensoy *et al.* 2018; Amini *et al.* 2020 — evidential learning
- Janet *et al.* 2019 — distance-based UQ for materials
- Brief clustering recap (folded in from former Unit 12): Bishop Ch. 9,
  Murphy Ch. 11

## Unit 14 — Physical constraints, limits, outlook (Week 14)
- Neuer 6.1–6.3 (physics-informed and constrained learning)
- Neuer Ch. 7 (explainability and trust)
- McClarren Ch. 11 + Ch. 12 (physics-informed + limitations / outlook)
- Karniadakis *et al.* 2021 — physics-informed ML survey
- Equivariance / symmetry primer: Geiger & Smidt 2022 (e3nn), Batatia
  *et al.* 2023
- FAIR + reproducibility framing: Wilkinson *et al.* 2016, Sansone &
  Aalberg 2024 (materials FAIR audit)
