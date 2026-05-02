# Unit 4 — Application-Focused Plan (Claude variant)

## Objective

Make the move "from classical metrics to learned representations" **concrete** by anchoring it to **real, published CNN/representation case studies** across materials characterization and processing. MFML covers the NN math (perceptron, MLP, activations, backprop) — this deck does **not** re-derive it.

By the end of 90 minutes, students should be able to:

1. Quantify the **information loss** when a micrograph collapses to one stereological scalar.
2. Choose between **tabular**, **S₂/MKS**, **eigen-microstructure**, **image**, and **1-D spectral** encodings — and name the matching architecture.
3. Recognise **published CNN applications** in characterization (microscopy, EBSD, XRD, 3-D tomography) and processing (LPBF, welding, crystal plasticity).
4. Identify **failure modes** that erase apparent CNN gains (specimen splits, lab shift, imbalance, segmentation noise, raw-pixel MLPs).
5. Articulate **why** the next unit is CNNs — locality + weight sharing as inductive bias for spatial data.

## Slide Structure (50 slides)

### §0 Frame (3) — slides 1–3
- Title; recap of Unit 3 + today's roadmap; learning outcomes.

### §1 Why classical metrics aren't enough (7) — slides 4–10
- Stereology in one slide; hand-crafted descriptor families; information bottleneck.
- The **Azimi 2018** hero result (FCNN on dual-phase steel: 48.9% → 93.9%) as the motivating "wow".
- Where ASTM-style scalars hit a wall (HEAs, AM, multi-scale composites).
- Paradigm-shift table: classical descriptor pipeline vs learned representation pipeline.

### §2 Encoding microstructure (8) — slides 11–18
- The encoding question (input upper-bounds learnable physics).
- Tabular composition + process; S₂; MKS pipeline; eigen-microstructures; raw images; 1-D spectra.
- Encoding decision rule (table mapping input type → typical D → first-line architecture).

### §3 Application gallery — characterization (12) — slides 19–30
- One overview slide; one recap slide.
- Ten case studies (one slide each):
  1. Azimi 2018 — FCNN steel phase classification.
  2. DeCost & Holm — UHCS microstructure manifold (CNN features beat SIFT).
  3. U-Net for EBSD phase segmentation (dual-phase steel).
  4. Stoll & Benner 2021 — complex microstructure inference (Nat. Commun.).
  5. TEM dislocation segmentation (FZ Jülich / Trampert).
  6. STEM irradiation defects (Roberts 2019).
  7. 3D-CNN composite stiffness from RVE voxels (Yang 2018).
  8. Yield-surface prediction from microstructure (Frankel 2022).
  9. Park 2017 — 1-D CNN for XRD phase ID (Nat. Commun.).
  10. 3-D U-Net for Li-ion battery electrode tomography (Müller 2021).

### §4 Application gallery — processing (8) — slides 31–38
- Overview + recap.
- Six case studies:
  11. LPBF melt-pool monitoring with Xception (~99% defect classification).
  12. Pore detection from coaxial imaging (precision/recall ≈ 1).
  13. Real-time FSW U-Net segmentation at ~25 fps.
  14. Radiographic weld defect classification with hybrid CNN-ViT.
  15. CNN crystal-plasticity surrogate (Mianroodi 2021).
  16. End-to-end Process → Structure → Property closure.

### §5 The pattern + pitfalls (8) — slides 39–46
- Common pattern across §3+§4 (raw signal → representation → label).
- CNN vs 2-point statistics — when CNN wins.
- Specimen splits revisited; cross-lab/instrument shift; class imbalance on rare defects.
- Label noise from upstream segmentation; raw-pixel MLP parameter blow-up; when **not** to use a CNN.

### §6 Bridge & wrap (4) — slides 47–50
- MFML toolkit applied here (why MLP works on tabular and fails on pixels).
- Looking ahead to Unit 5 (CNN architectures).
- Reading + exercises.
- Key takeaways and references.

## Materials sources

- **Sandfeld (2024)** — Ch. 17 nets; microstructure / stereology chapters; 2-point statistics.
- **Neuer (2024)** — Ch. 4 supervised workflow, losses.
- **McClarren (2021)** — Ch. 5 feed-forward nets, capacity.
- **Goodfellow et al. (2016)** — Deep Learning, Ch. 9 (CNN motivation).
- **Application papers** (added to `ref.bib` — see end of `unit4_content_50slides_claude.md`):
  Azimi 2018; DeCost & Holm 2017; DeCost et al. 2019 (UHCS); Stoll & Benner 2021; Roberts 2019; Trampert et al. (TEM dislocations); Yang et al. 2018; Frankel et al. 2022; Mianroodi et al. 2021; Park et al. 2017 (XRD); Oviedo et al. 2019 (XRD); Kaufmann et al. 2020 (electron diffraction); Müller et al. 2021 (Li-ion CT); Scime & Beuth (LPBF); Holm et al. 2020 (CV/ML for microstructure review).

## Differences vs the original `unit4_plan.md`

- **Removed** the MCP/Perceptron/ADALINE/MLP/activations re-derivation (MFML duplicates avoided).
- **Added** §3+§4 (20 slides) of real published case studies — the original deck had zero case-study slides.
- **Kept** the encoding theory (S₂, MKS, eigen-microstructures) but tightened it from 12 to 8 slides.
- **Strengthened** the bridge to Unit 5 (CNNs) with concrete failure modes that motivate convolution as the right inductive bias.
