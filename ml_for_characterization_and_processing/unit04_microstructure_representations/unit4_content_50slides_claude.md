# Unit 4 — Application-Focused Slide Content (Claude variant, 50 slides)

> **Convention.** Each slide gets a short bullet outline. The actual Quarto deck is in `01_intro_claude.qmd`. Citation keys follow `ref.bib`; entries to add are listed at the bottom of this file.

---

## §0 — Frame (3 slides)

### Slide 01 — Title
- **Unit 4: From Classical Metrics to Learned Representations.**
- ML in Materials Processing & Characterization (ML-PC).
- Subtitle: *"What CNNs already do for us — before we open the CNN box next week."*

### Slide 02 — Where We Are
- **Recap Unit 3:** cleaning, scaling, leakage-safe validation.
- **Today (Unit 4):** turn microstructure into model-ready tensors; survey real CNN/representation case studies in characterization and processing.
- **Next (Unit 5):** CNN architectures themselves.

### Slide 03 — Learning Outcomes
- Quantify information loss when a micrograph becomes one stereological scalar.
- Choose between tabular / S₂ / image / 1-D spectral encodings.
- Recognise published CNN applications across characterization and processing.
- Name failure modes that erase apparent gains.
- Explain why CNNs are the right next step.

---

## §1 — Why Classical Metrics Aren't Enough (7 slides)

### Slide 04 — Stereology in One Slide
- $V_V$, $S_V$, mean intercept; ASTM standards.
- Standards-grade — reproducible, auditable, **lossy** by construction.

### Slide 05 — Hand-Crafted Descriptor Families
- Shape (aspect ratio, circularity, tortuosity).
- Distribution (nearest-neighbour, clustering indices).
- Texture (ODF coefficients, pole figures).
- Each family answers questions you knew to ask.

### Slide 06 — The Information Bottleneck
- Micrograph $\sim 10^6$ pixels $\to$ ASTM grain-size number = **single scalar**.
- Vast compression by design — most spatial detail discarded **by construction**, not by accident.

### Slide 07 — Where ASTM Hits a Wall
- High-entropy alloys, additively manufactured microstructures, hierarchical nanocomposites.
- Multi-phase, multi-scale, spatially varying — **stationarity assumption** broken.
- Setup for the next slide: what does "automated, learned" classification get you?

### Slide 08 — Hero Result: Steel Phase Classification (Azimi 2018)
- Dual-phase steel SEM micrographs.
- **Prior SOTA**: 48.9% classification accuracy.
- **FCNN with max-voting**: **93.94%**.
- Same images, no new physics — change of representation alone.
- Citation: @azimi2018advanced.

### Slide 09 — Where Hand-Crafted Hits a Wall (Cont.)
- Ti, Ni-superalloys, weld HAZ — phases overlap, etching artefacts dominate hand-crafted features.
- Holm et al. 2020 review: across image classification, segmentation, detection, instance segmentation — CV/ML systems beat hand-crafted across the board when data + labels exist.
- Citation: @holm2020overview.

### Slide 10 — The Paradigm Shift
- Two-column table:

| | Classical | Modern (learned) |
|---|---|---|
| Input | Image → metrics | Image / signal → representation |
| Features | Hand-crafted, named | Learned (or correlation-based) |
| Bottleneck | Information loss | Data + compute + validation discipline |

- The ethics (specimen splits, leakage, calibration) carry over unchanged.

---

## §2 — Encoding Microstructure for ML (8 slides)

### Slide 11 — The Encoding Question
- Map microstructure → tensor $\mathbf{X}$ before training.
- **Encoding upper-bounds the physics** the hypothesis class can express.
- Garbage encoding → garbage in, regardless of architecture.

### Slide 12 — Tabular: Composition + Process
- $\mathbf{x} \in \mathbb{R}^{10\text{–}50}$: composition fractions, temperatures, times, cooling rate.
- Natural MLP turf; standardise per **train** fold.
- Watch: composition fractions sum to 1 → collinearity.

### Slide 13 — Two-Point Statistics $S_2$
- $S_2(\mathbf{r}) = P(\text{phase}(\mathbf{x})=\alpha \wedge \text{phase}(\mathbf{x}+\mathbf{r})=\alpha)$.
- Translation-averaged correlation — captures length scales, anisotropy, clustering.
- More than one scalar, far less than full pixels.

### Slide 14 — MKS Pipeline (Materials Knowledge Systems)
- Segment → compute $S_2$ on grid → standardise → MLP/linear.
- Bakes in **translation invariance** before any training.
- Keeps $D \sim$ hundreds; scales to materials sample counts.
- Citation: Kalidindi/MKS group references in @sandfeld_materials_data_science.

### Slide 15 — Eigen-Microstructures
- Stack registered fields → PCA on standardised columns/voxels.
- Dominant modes of structural variation, not brightness/thickness.
- Linear cousin of CNN feature learning.

### Slide 16 — Image as Tensor
- $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ for 2-D micrograph.
- $\mathbf{X} \in \mathbb{R}^{D \times H \times W \times C}$ for 3-D tomography.
- MLP on flattened pixels → parameter explosion (next-week motivation).

### Slide 17 — Spectra as 1-D Signals
- XRD pattern, EELS edge, Raman spectrum: $\mathbf{x} \in \mathbb{R}^{N_{\text{channels}}}$.
- **1-D CNN** is the natural architecture — same convolutional inductive bias, applied to channel index instead of pixels.
- Preview: Park 2017 (slide 28).

### Slide 18 — Encoding Decision Rule

| Input type | Typical $D$ | First-line model |
|---|---|---|
| Composition + process | 10–50 | MLP |
| Morphology scalars | 5–50 | MLP |
| $S_2$ / MKS | $10^2$–$10^3$ | MLP / shallow 1-D conv |
| 1-D spectrum | $10^3$–$10^4$ | 1-D CNN |
| 2-D micrograph | $10^4$–$10^7$ | **CNN** (Unit 5) |
| 3-D volume | $10^6$–$10^9$ | 3-D CNN / U-Net |

- Start with the smallest $\mathbf{X}$ that passes physics + grouped CV.

---

## §3 — Application Gallery: Characterization (12 slides)

### Slide 19 — Gallery Overview (Characterization)
- Six task families: classification, segmentation, defect/feature detection, property prediction, spectroscopy, 3-D tomography.
- Pattern: raw signal → CNN/representation → label or property — **no hand-crafted descriptors**.

### Slide 20 — Case 1: Azimi 2018 — Steel Phase Classification
- Task: classify dual-phase steels (martensite/bainite/pearlite/...).
- Method: Fully Convolutional Neural Network + max-voting on superpixels.
- Result: **93.94%** vs prior best 48.9%.
- Lesson: representation change alone unlocks a 45-pt accuracy jump.
- Citation: @azimi2018advanced.

### Slide 21 — Case 2: UHCS Microstructure Manifold (DeCost & Holm)
- Dataset: 961 SEM micrographs, ultra-high-carbon steel, public on `materialsdata.nist.gov`.
- CNN feature representations vs SIFT bag-of-visual-words.
- Result: CNN embeddings cluster heat-treatment classes naturally; enable retrieval and semi-supervised classification.
- Lesson: pretrained CNN features generalise even on small materials datasets.
- Citations: @decost2017uhcs, @decost2019uhcs.

### Slide 22 — Case 3: U-Net for EBSD Phase Segmentation
- Task: pixel-level ferrite/martensite segmentation in dual-phase steel EBSD maps.
- Method: U-Net with encoder backbone.
- Result: ~95% pixel accuracy on raw orientation data; ~98% with derived parameters.
- Citation: @ostormujof2022deep.

### Slide 23 — Case 4: Complex Microstructure Inference (Durmaz et al. 2021)
- Task: distinguish four steel phases (ferrite/pearlite/bainite/martensite) jointly from LOM + SEM.
- Method: ensemble U-Nets across modalities and magnifications.
- Result: ~0.34 µm mean grain-size error on held-out processing conditions.
- Lesson: multi-modal CNN segmentation matches expert metallographers.
- Citation: @durmaz2021deep.

### Slide 24 — Case 5: TEM Dislocation Segmentation (Govind et al. 2024)
- Task: instance segmentation of dislocations in TEM micrographs.
- Method: simulation-based training; YOLO-style and U-Net variants.
- Insight: bypass the "never enough labels" bottleneck via physics-based simulation.
- Citation: @govind2024deep.

### Slide 25 — Case 6: STEM Defects in Irradiated Steels (Roberts 2019)
- Task: semantic segmentation of voids, loops, and precipitates in STEM images.
- Result: ~85% IoU; matches inter-annotator variability.
- Lesson: CNNs reach human-level performance once the label budget is sufficient.
- Citation: @roberts2019stemdefects.

### Slide 26 — Case 7: 3-D-CNN Composite Stiffness from RVEs
- Task: predict effective stiffness tensor of two-phase composites from voxelated RVEs.
- Method: 3-D CNN trained on FE-homogenised stiffness labels.
- Result: > **40% accuracy improvement** over hand-engineered descriptors at a fraction of the FE cost.
- Lesson: CNN replaces homogenisation FE for inner-loop material design.
- Citation: @yang2018cnn.

### Slide 27 — Case 8: Yield-Surface Prediction from Microstructure
- Task: predict the full yield surface (not just one stiffness scalar) from a microstructure image.
- Method: CNN regression with multi-output head.
- Lesson: representation learning lets one model output **functional** properties, not just scalars.
- Citation: @heidenreich2023yield.

### Slide 28 — Case 9: Lee/Park et al. 2020 — XRD Phase ID with 1-D CNN
- Task: phase identification in multi-phase inorganic mixtures from XRD patterns.
- Method: 1-D CNN trained on simulated patterns from ICSD; tested on real experimental data.
- Result: ~**100% phase identification** on held-out experiments; ~86% three-phase quantification.
- Lesson: CNN ≠ image network — convolution applies wherever there is locality (here, peak shape).
- Citation: @lee2020xrd.

### Slide 29 — Case 10: 3-D U-Net for Li-ion Electrode Tomography
- Task: segment cathode active material / carbon-binder / pore from X-ray nano-CT.
- Method: 3-D U-Net trained partly on **synthetic** electrodes with known ground truth.
- Result: reliable segmentation where contrast-based thresholding fails outright.
- Lesson: simulation-augmented training is now standard for hard 3-D segmentation.
- Citation: @muller2021battery.

### Slide 30 — Characterization Gallery Recap
- Same recipe across ten cases: **raw signal → CNN → output**.
- Tasks: classification, segmentation, regression, retrieval — same architecture family.
- Domains: SEM, EBSD, TEM, STEM, XRD, X-ray CT — all fit the convolutional inductive bias.

---

## §4 — Application Gallery: Processing (8 slides)

### Slide 31 — Gallery Overview (Processing)
- In-situ monitoring, defect inspection, surrogate modelling, end-to-end PSP closure.
- The pattern: process **camera/thermograph/sensor stream** → CNN → quality decision in real time.

### Slide 32 — Case 11: LPBF Powder-Bed Quality (Xception, Fischer et al. 2022)
- Task: classify powder-bed defects (balling, incomplete spreading, groove, ridge, spatters, protruding part, scattered powder, homogeneous) from line-sensor recoater images during LPBF.
- Method: Xception pretrained on ImageNet, fine-tuned on Fraunhofer-ILT dataset (coaxial / dark-field / diffuse lighting).
- Result: **99.15%** accuracy across seven classes (dark-field); per-class F1 between 97.85% and 99.71%.
- Lesson: ImageNet pretraining transfers to grayscale recoater frames — a clean transfer-learning teaser for Unit 6.
- Citation: @fischer2022xception.

### Slide 33 — Case 12: Thermographic Porosity Prediction (Oster et al. 2024)
- Task: predict keyhole / lack-of-fusion porosity in LPBF parts from in-situ short-wave-infrared thermography.
- Method: supervised CNN classifier on multi-layer thermographic feature stacks; labels from post-build CT ground truth.
- Result: accuracy ~0.96; F1 ~0.86 for keyhole porosity in small sub-volumes.
- Lesson: thermal history is a proxy for porosity — CNNs decode it densely, far below the resolution of point pyrometers.
- Citation: @oster2024thermographic.

### Slide 34 — Case 13: Real-Time FSW U-Net at ~25 fps
- Task: surface defect segmentation + weld-width geometry on friction-stir welds.
- Method: U-Net, on-device inference at video rate.
- Result: continuous closed-loop quality monitoring during welding.
- Lesson: CNNs are fast enough for in-line process control.
- Citation: @loganathan2026fsw.

### Slide 35 — Case 14: Radiographic Weld Inspection (CNN-ViT)
- Task: classify weld defects (cracks, porosity, slag inclusions) in industrial X-ray radiographs.
- Method: hybrid CNN feature extractor + Vision Transformer head with explainability.
- Result: >98% accuracy with attention-map evidence per decision.
- Lesson: hybrid CNN+ViT combines local feature extraction with global context.
- Citation: @parmar2026weld.

### Slide 36 — Case 15: CNN as Crystal-Plasticity Surrogate
- Task: predict per-voxel stress field in a polycrystalline RVE under loading.
- Method: 3-D CNN trained on CPFEM simulation data.
- Result: orders of magnitude faster than CPFEM at comparable accuracy on test microstructures.
- Lesson: CNN is now a viable surrogate inside design loops where CPFEM is too slow.
- Citation: @mianroodi2021cp.

### Slide 37 — Case 16: End-to-End Process → Structure → Property
- Closing example: process parameters → CNN-predicted microstructure → CNN-predicted property.
- Two CNNs in series, trained independently then chained.
- Lesson: the PSPP backbone of the course is now **fully learnable** end-to-end.
- Citation: @sandfeld_materials_data_science.

### Slide 38 — Processing Gallery Recap
- Six cases: same convolutional toolkit, deployed on cameras, thermographs, radiographs, RVEs.
- Performance ceilings are usually set by **labels and SOPs**, not architectures.

---

## §5 — The Pattern + Pitfalls (8 slides)

### Slide 39 — The Common Pattern
```
Raw signal  →  Encoding  →  CNN / representation  →  Loss → Label
```
- All sixteen case studies above fit this template.
- The "encoding" step is the difference between a clean and a hopeless project.

### Slide 40 — CNN vs 2-Point Statistics — When CNN Wins
- CNN wins when: spatial features are **task-specific**, you have $\gtrsim 10^3$ specimens, imaging SOPs are tight.
- $S_2$/MKS wins when: $N$ is small, simulation data dominate, interpretability is required.
- Recent hybrids (CNN ⊕ $S_2$) achieve $R^2 > 0.96$ on stiffness regression.
- Citation: @mann2022robust.

### Slide 41 — Specimen Splits Revisited
- IID-split crops from the same coupon → optimistic $R^2 \approx 0.95$.
- Specimen-level split on the same labels → honest $R^2 \approx 0.72$.
- The number that matters is the one that survives **deployment-grade** splitting.

### Slide 42 — Cross-Lab Distribution Shift
- Train on microscope A: $R^2 \approx 0.88$.
- Test on microscope B (same alloy): $R^2 \approx 0.45$.
- CNN may latch onto contrast, vignetting, or detector noise rather than grains.
- Mitigations preview Unit 6 (transfer + domain adaptation).

### Slide 43 — Class Imbalance on Rare Defects
- Defect prevalence 2% → "always predict good" → 98% accuracy, 0% recall.
- Use **precision/recall/F1** or precision-recall AUC, not accuracy.
- Stratified specimen-level splits; cost-sensitive losses; active labelling of hard negatives.

### Slide 44 — Label Noise from Upstream Segmentation
- $\mathbf{x}$ derived from segmentation v1.3; $y$ pristine tensile.
- Segmentation drift → false aleatory scatter → CNN fits artefacts.
- Mitigations: inter-annotator study; ensemble segmentation; uncertainty-aware losses (preview Unit 12).

### Slide 45 — Raw-Pixel MLP Failure
- $1024 \times 1024$ RGB → first dense layer ≈ $10^9$ weights.
- With $N \sim 10^2$ specimens: spurious pixel correlations win.
- The exact motivation for next week: **locality + weight sharing** of CNNs.

### Slide 46 — When *Not* to Use a CNN
- Tiny $N$ ($<$ a few hundred specimens) → linear / GP / hand-crafted + MLP first.
- Regulatory / safety contexts that require coefficient-level audit.
- Tabular composition+process inputs — MLP is enough.
- "Use the simplest model that survives grouped CV."

---

## §6 — Bridge & Wrap (4 slides)

### Slide 47 — MFML Toolkit Applied Here
- Forward pass, activations, training loop **don't change** between tabular and image data.
- What changes: **what is in $\mathbf{X}$**, **how you split**, and **what loss reflects deployment cost**.
- The application gallery is a tour of this single toolkit applied to different encodings.

### Slide 48 — Looking Ahead: Unit 5 (CNNs)
- Convolution = locality + weight sharing.
- Architectures: VGG, ResNet, U-Net, ViT (and where each fits a materials task).
- Carry forward: specimen splits, normalisation, shift awareness — CNNs **do not** remove these obligations.

### Slide 49 — Reading + Exercises
- **Reading:**
  - Sandfeld 2024 Ch. 17 (NN); microstructure / 2-point statistics chapters.
  - McClarren 2021 Ch. 5 (feed-forward nets).
  - Neuer 2024 Ch. 4 (supervised workflow).
  - Goodfellow 2016 Ch. 9 (CNN motivation, optional).
  - Holm 2020 review (one-paper survey of the field).
- **Exercises:**
  1. Reproduce Azimi-style classification on UHCS micrographs (NIST public data) — compare a hand-crafted feature pipeline vs a small CNN.
  2. Compute binned $S_2$ on a binary microstructure set; train an MLP on $S_2$ and a small CNN on the raw images; compare grouped-CV scores.
  3. Repeat (2) with deliberate patch-level (instead of specimen-level) splitting; measure how much $R^2$ inflates.

### Slide 50 — Key Takeaways
- Hand-crafted metrics: interpretable, standardised, **lossy**.
- $S_2$ / MKS / eigen-modes: principled middle ground.
- Modern materials ML: **representation > metric**, when data and SOPs allow.
- Sixteen real applications across characterization and processing — same recipe.
- CNNs are the **right inductive bias** for spatial/spectral signals — Unit 5 next.
- Specimen splits, lab shift, imbalance, segmentation noise: still own them.

---

## Citation map (verified — already in `ref.bib`)

The 18 application-paper entries cited by `01_intro.qmd` have been added to `ref.bib` under the section header `% --- Unit 4 (Claude variant) application case studies ---`. Verified via Crossref DOIs:

| Slide | Cite key | Reference (verified) | DOI |
|---|---|---|---|
| 8, 20 | `azimi2018advanced` | Azimi et al., *Sci. Rep.* 8:2128 (2018) | `10.1038/s41598-018-20037-5` |
| 21 | `decost2017uhcs` | DeCost, Francis, Holm, *Acta Materialia* 133, 30–40 (2017) | `10.1016/j.actamat.2017.05.014` |
| 21 | `decost2019uhcs` | DeCost, Lei, Francis, Holm, *Microscopy and Microanalysis* 25(1), 21–29 (2019) | `10.1017/S1431927618015635` |
| 9 | `holm2020overview` | Holm et al., *Met. Mater. Trans. A* 51(12), 5985–5999 (2020) | `10.1007/s11661-020-06008-4` |
| 22 | `ostormujof2022deep` | Martinez Ostormujof et al., *Mater. Charact.* 184, 111638 (2022) | `10.1016/j.matchar.2021.111638` |
| 23 | `durmaz2021deep` | Durmaz et al., *Nat. Commun.* 12:6272 (2021) | `10.1038/s41467-021-26565-5` |
| 24 | `govind2024deep` | Govind, Oliveros, Dlouhy, Legros, Sandfeld, *Mach. Learn.: Sci. Technol.* 5(1), 015006 (2024) | `10.1088/2632-2153/ad1a4e` |
| 25 | `roberts2019stemdefects` | Roberts et al., *Sci. Rep.* 9:12744 (2019) | `10.1038/s41598-019-49105-0` |
| 26 | `yang2018cnn` | Yang et al., *Comp. Mater. Sci.* 151, 278–287 (2018) | `10.1016/j.commatsci.2018.05.014` |
| 27 | `heidenreich2023yield` | Heidenreich, Gorji, Mohr, *Int. J. Plast.* 163, 103506 (2023) | `10.1016/j.ijplas.2022.103506` |
| 28 | `lee2020xrd` | Lee, Park, Lee, Singh, Sohn, *Nat. Commun.* 11:86 (2020) | `10.1038/s41467-019-13749-3` |
| 29 | `muller2021battery` | Müller et al., *Nat. Commun.* 12:6205 (2021) | `10.1038/s41467-021-26480-9` |
| 40 | `mann2022robust` | Mann & Kalidindi, *Front. Mater.* 9:851085 (2022) | `10.3389/fmats.2022.851085` |
| 32 | `fischer2022xception` | Fischer, Zimmermann, Praetzsch, Knaak, *Mater. & Design* 222, 111029 (2022) | `10.1016/j.matdes.2022.111029` |
| 33 | `oster2024thermographic` | Oster, Breese, Ulbricht, Mohr, Altenburg, *J. Intell. Manuf.* 35(4), 1687–1706 (2024) | `10.1007/s10845-023-02117-0` |
| 34 | `loganathan2026fsw` | Loganathan, Andersson, Patel, *J. Manuf. Syst.* 85, 175–192 (2026) | `10.1016/j.jmsy.2026.01.007` |
| 35 | `parmar2026weld` | Parmar et al., *Sci. Rep.* (2026) | `10.1038/s41598-026-44874-x` |
| 36 | `mianroodi2021cp` | Mianroodi, Siboni, Raabe, *npj Comput. Mater.* 7:99 (2021) | `10.1038/s41524-021-00571-z` |

> **Note on corrections.** During verification, six citation keys from the first draft were replaced with the correct ones: the Nat. Commun. 2021 microstructure-inference paper is by **Durmaz et al.** (not Stoll & Benner); the EBSD U-Net paper is by **Martinez Ostormujof et al.** (not Bachmann); the TEM-dislocation paper is by **Govind et al. 2024** (not Trampert 2023); the yield-surface paper is by **Heidenreich, Gorji, Mohr** (not Frankel); the LPBF Xception paper is **Fischer et al. 2022** on **powder-bed quality** (not melt-pool); the thermographic LPBF paper is by **Oster et al. 2024** (not Gobert). Slides 32–33 wording in `01_intro.qmd` was updated accordingly.
