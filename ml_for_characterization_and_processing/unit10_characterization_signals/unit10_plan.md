# Unit 10 Plan: ML for Characterization Signals

## Metadata
- **Course:** Machine Learning for Characterization and Processing (ML-PC)
- **Unit:** 10
- **Topic:** ML for Characterization Signals
- **Duration:** 90 Minutes
- **Key References:** 
    - Sandfeld (2024), Ch 15.3, 19.4
    - Neuer (2024), Ch 5
    - McClarren (2021), Ch 8
    - MFML Unit 9 (Representation Learning)

## Learning Objectives
- Identify common 1D characterization signals (XRD, EELS, EDS, XPS) and their properties.
- Apply Principal Component Analysis (PCA) for denoising and dimensionality reduction of spectra.
- Understand the architecture and objective of Autoencoders (AEs) for signal processing.
- Compare linear (PCA) and non-linear (AE) dimensionality reduction for complex spectral data.
- Use latent space representations for phase identification and anomaly detection in signals.

## Lecture Structure (90 Minutes)

### 1. Introduction: 1D Signals in Materials Science (10m)
- From images to spectra: High-dimensional data vectors.
- The nature of characterization signals: Peaks, backgrounds, and noise.
- Goal: Compression, Denoising, and Interpretation.

### 2. Classical Spectral Analysis: PCA (20m)
- (Ref: Sandfeld Ch 15.2, 15.3)
- Eigen-spectra (Principal Components) as basis functions.
- Quantifying variance: Scree plots and intrinsic dimensionality.
- Denoising via reconstruction: Keeping only the significant components.

### 3. Deep Unsupervised Learning: Autoencoders (25m)
- (Ref: Sandfeld Ch 19.4; McClarren Ch 8)
- The Bottleneck Principle: Forcing the model to learn a compressed representation.
- Encoder: Mapping raw spectra to latent codes.
- Decoder: Reconstructing the signal from the latent space.
- Reconstruction Error as a measure of "novelty."

### 4. Denoising and Signal Enhancement (15m)
- (Ref: McClarren Ch 8.3.2)
- Denoising Autoencoders: Training with corrupted inputs to learn the data manifold.
- Applications in low-count experimental data (e.g., fast EDS mapping).

### 5. Case Study: Latent Space & Phase Mapping (15m)
- (Ref: Sandfeld Ch 15.5, 15.6)
- Clustering in latent space to identify material phases.
- Visualizing high-dimensional spectral datasets (t-SNE/UMAP).
- Example: Mapping chemical variations in a multi-phase alloy.

### 6. Summary and Conclusion (5m)

## 50-Slide Strategy
- Slides 1-5: Intro & Spectral Data Examples
- Slides 6-15: PCA for Spectra (Theory, Eigen-spectra, Denoising)
- Slides 16-30: Autoencoders (Architecture, Loss, Non-linearity)
- Slides 31-40: Denoising & Feature Extraction
- Slides 41-48: Case Studies (XRD/EELS Phase Identification)
- Slides 49-50: Summary & Outlook
