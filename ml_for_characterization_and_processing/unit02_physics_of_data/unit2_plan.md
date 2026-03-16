# Unit 2: Physics of Data Formation - Lecture Plan

## Objective
Synthesize the connection between physical measurements (image/signal formation) and mathematical representations (data matrix, PCA/SVD) as the foundation for Machine Learning in materials science.

## Slide Structure (approx. 50 slides)

### Part 1: Signal & Image Formation (12 slides)
- **Introduction (2 slides):** Bridging measurement and ML.
- **Physical Sensing (2 slides):** Sensors as transducers ($\xi(t) \to x$).
- **Spatial/Temporal Sampling (3 slides):**
  - Continuous to discrete mapping.
  - Nyquist-Shannon Theorem ($\nu_S \ge 2\nu_{max}$).
  - Aliasing: When resolution fails (Moiré patterns in microscopy).
- **Physical Priors (3 slides):**
  - Jitter ($\delta t$) and Resolution limits.
  - Finite Rate of Innovation: Sampling below Nyquist with prior info.
- **Noise (2 slides):** Sensors as stochastic processes ($x_i = \xi_i + u_x$).

### Part 2: Mathematical Representation of Data (10 slides)
- **Notation (2 slides):** Scalars, Vectors, Matrices (Sandfeld conventions).
- **The Data Matrix (3 slides):**
  - Features (columns) vs. Observations (rows).
  - Feature matrix $X$, Target matrix $Y$.
- **Python Implementation (2 slides):** Numpy/Pandas structures.
- **Statistical Moments (3 slides):**
  - Mean, Variance, Skewness, Kurtosis as noise characterization.
  - Covariance and Correlation matrices (similarity).

### Part 3: Uncertainty & Stochasticity (8 slides)
- **Aleatory vs. Epistemic Uncertainty (3 slides):**
  - Inherent randomness (thermal noise) vs. lack of knowledge (model bias).
- **Physical Noise Models (3 slides):**
  - Gaussian (thermal/electronic noise).
  - Poisson (photon/shot noise in EM).
  - Weibull (failure/defect statistics).
- **Bayes' Theorem in Sensing (2 slides):** $P(\text{Physical State} | \text{Sensor Output})$.

### Part 4: Dimensionality Reduction (15 slides)
- **The Curse of Dimensionality (2 slides):** High-dim experimental data (pixels/spectra).
- **SVD Theory (3 slides):**
  - $X = USV^T$ decomposition.
  - Rank and Independent variables.
- **PCA as Eigendecomposition (3 slides):**
  - Maximizing variance along principal directions.
  - Connecting Covariance Matrix to SVD.
- **Variance Explained (2 slides):** Scree plots and "Fraction of Variation Explained".
- **Case Studies (5 slides):**
  - Time series reduction (McClarren Hohlraum case).
  - Hyperspectral Foliage analysis (McClarren).
  - Eigenfaces/Eigenmicrostructures (Sandfeld MNIST).

### Part 5: Clustering & Visualization (5 slides)
- **K-means Clustering (2 slides):** Finding natural groupings in PC space.
- **t-SNE (3 slides):**
  - Visualizing high-dim clusters (Normal vs. Cauchy tails).
  - Fashion MNIST example.

## Materials Sources
- **Neuer (2024):** Ch 2 (Sampling, Nyquist, Uncertainty types, Moments, Bayes).
- **Sandfeld (2024):** Ch 3 (Data Matrix, Notation), Ch 15 (PCA, Eigenvectors, Images).
- **McClarren (2021):** Ch 4 (SVD, K-means, t-SNE, Case studies).
