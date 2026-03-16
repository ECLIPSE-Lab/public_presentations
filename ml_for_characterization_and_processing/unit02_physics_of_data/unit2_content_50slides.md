# Unit 2: Physics of Data Formation - Slide Content (50 Slides)

## Part 1: Signal & Image Formation (12 slides)

**Slide 1: Title Slide**
- **Unit 2: Physics of Data Formation**
- Machine Learning in Materials Processing & Characterization (ML-PC)
- Topic: From Physical Sensing to Mathematical Representation

**Slide 2: Bridging Measurement and ML**
- How does a material property become a data point?
- Transition from physical process $\xi(t)$ to digital value $x_i$.
- Understanding the physics of sensing is crucial for selecting appropriate ML models (priors).

**Slide 3: Sensors as Transducers**
- Sensors convert physical stimuli (e.g., photon intensity, electron scattering, temperature) into electrical/digital signals.
- The mapping $f: \text{Physical State} \to \text{Digital Representation}$.

**Slide 4: Continuous to Discrete Mapping**
- Sampling: Measuring at discrete points $\tau_i$.
- Neuer's Sampling formula: $x_i = \xi(\tau_i + \delta t) + u_x$
  - $\xi$: Underlying physical truth.
  - $\delta t$: Timing jitter/uncertainty.
  - $u_x$: Amplitude noise/uncertainty.

**Slide 5: Temporal and Spatial Sampling**
- Sampling Rate $\nu_S = 1/\Delta t$.
- In Characterization: Pixel pitch (spatial), frame rate (temporal), energy resolution (spectral).

**Slide 6: The Nyquist-Shannon Theorem**
- To fully capture a signal with maximum frequency $\nu_{max}$, we must sample at least twice as fast: $\nu_S \ge 2\nu_{max}$.
- Nyquist Frequency: $\nu_{Nyquist} = \frac{1}{2} \nu_S$.

**Slide 7: Aliasing - When Resolution Fails**
- If $\nu_S < 2\nu_{max}$, high-frequency components are "folded" into lower frequencies.
- Example: Moiré patterns in TEM when grid resolution and crystal lattice interfere.

**Slide 8: Physical Resolution Limits**
- Optical/Electron diffraction limits.
- Point Spread Function (PSF): The response of an imaging system to a point source.
- Blurring as a physical prior for convolutional models.

**Slide 9: Jitter and Temporal Resolution**
- Jitter ($\delta t$) in our "clock" during sampling.
- Leads to phase noise and uncertainty in transient measurements (e.g., pump-probe experiments).

**Slide 10: Finite Rate of Innovation (FRI)**
- Vetterli et al. (2002): If we have *prior knowledge* of the signal structure (e.g., sum of $K$ diracs), we can sample below Nyquist.
- "Sparsity" in the physical process enables compressed sensing.

**Slide 11: Noise as a Physical Process**
- Sensors as stochastic processes.
- Sources: Thermal noise (Johnson-Nyquist), Shot noise (counting statistics).

**Slide 12: Summary: Part 1**
- Understanding sampling is critical to avoid artifacts.
- Resolution and noise are physical priors for ML.

---

## Part 2: Mathematical Representation (10 slides)

**Slide 13: Introduction to Data Representation**
- How to store the physical observations?
- Sandfeld Conventions: Scalars ($a$), Vectors ($\mathbf{x}$), Matrices ($\mathbf{X}$).

**Slide 14: Notation Standards (Sandfeld Ch 3)**
- **Scalars:** Italic Latin/Greek ($a, \lambda$).
- **Vectors:** Lowercase bold bold ($\mathbf{a}$).
- **Matrices:** Uppercase bold sans-serif ($\mathbf{A}$).
- **Indices:** Python starts at 0; Mathematics often starts at 1.

**Slide 15: The Data Point P**
- A compound $P = (\mathbf{x}, \mathbf{y})$.
- $\mathbf{x}$: Input variables (Features).
- $\mathbf{y}$: Output variables (Targets).

**Slide 16: The Feature Matrix X (Design Matrix)**
- $m$ rows (Observations/Instances).
- $n$ columns (Features).
- $x_{i,j}$: The $j$-th feature of the $i$-th observation.

**Slide 17: Tabular Data (Sandfeld Fig 3.2)**
- [Visual representation of X and Y matrices]
- Features are column vectors $X_j$.
- Observations are row vectors $\mathbf{x}_i$.

**Slide 18: Python Implementation (Numpy)**
- Creating matrices: `np.array`, `np.zeros`, `np.hstack`.
- Accessing data: `X[i, j]` for element, `X[:, j]` for feature.

**Slide 19: Characterizing Data: Expected Value**
- $\mu = E(x) = \sum x_i p(x_i)$.
- Arithmetic mean as realization with uniform distribution.

**Slide 20: Characterizing Data: Variance**
- $\sigma^2 = \text{Var}(\mathbf{x}) = \langle (\mathbf{x} - \langle \mathbf{x} \rangle)^2 \rangle$.
- Measure of fluctuation/uncertainty.

**Slide 21: Higher Moments: Skewness & Kurtosis**
- **Skewness:** Tiling/Asymmetry of distribution.
- **Kurtosis:** Tail weight/Peakiness (Flatness).
- Gaussian reference: Skewness = 0, Kurtosis = 3.

**Slide 22: Covariance and Correlation**
- **Covariance Matrix C:** Similarity between different features.
- **Correlation Matrix:** Standardized covariance ($[-1, 1]$).
- Exploratory data analysis (EDA) tool.

---

## Part 3: Uncertainty & Stochasticity (8 slides)

**Slide 23: Aleatory vs. Epistemic Uncertainty**
- Core distinction in Neuer (Ch 2.2.3).

**Slide 24: Aleatory Uncertainty (Statistical)**
- Inherent randomness (lat. *alea* = dice).
- Example: Shot noise, radioactive decay, thermal fluctuations.
- Cannot be reduced by more data.

**Slide 25: Epistemic Uncertainty (Systematic/Model)**
- Uncertainty from "lack of knowledge."
- Example: Small dataset, model bias, limited precision.
- *Can* be reduced by better models or more data.

**Slide 26: Physical Noise Models: Gaussian**
- Normal distribution $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$.
- Universal model for sum of random disturbances (Central Limit Theorem).

**Slide 27: Physical Noise Models: Poisson**
- Counting statistics for rare events.
- $p(x) = \frac{\lambda^x}{x!} e^{-\lambda}$.
- Mean = Variance = $\lambda$.

**Slide 28: Physical Noise Models: Weibull**
- Failure and defect statistics in materials.
- Models skewed failure distributions.

**Slide 29: Bayes' Theorem in Characterization**
- $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$.
- $P(\text{Physical State} | \text{Sensor Output}) \propto P(\text{Sensor Output} | \text{State}) \times P(\text{Prior})$.
- Physics provides the *Prior*.

**Slide 30: Summary: Part 3**
- Noise is not just "error"; it contains information about the process.
- Distinguishing uncertainty types helps in model selection.

---

## Part 4: Dimensionality Reduction (15 slides)

**Slide 31: The Curse of Dimensionality**
- Why reduce? Experimental data is high-dim (e.g., 1MP image = 10^6 features).
- Sparsity in high-dim space makes learning difficult.

**Slide 32: Singular Value Decomposition (SVD)**
- Matrix factorization: $\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^T$.
- $\mathbf{U}$: Left singular vectors (cases).
- $\mathbf{S}$: Singular values (diagonal, decreasing order).
- $\mathbf{V}$: Right singular vectors (feature combinations).

**Slide 33: Rank and Structure**
- Rank of matrix = Number of non-zero singular values.
- Reveals the number of *truly* independent variables.

**Slide 34: PCA Theory (Sandfeld Ch 15)**
- PCA maximizes variance along new coordinate axes.
- Principal components = Eigenvectors of the Covariance Matrix.

**Slide 35: Maximizing Variance**
- Why maximize variance? Spreading data out maximizes preserved information.
- First PC is the direction of greatest variability.

**Slide 36: SVD vs. PCA**
- SVD is the numerical algorithm.
- PCA is the statistical concept (centered data SVD).

**Slide 37: Scree Plots & Cumulative Variance**
- Fraction of variation explained $m_\ell = \frac{\sum_{n=1}^\ell s_n}{\sum_{n=1}^J s_n}$. (Note: some use squared singular values for variance ratio).
- Decision tool: How many components to keep?

**Slide 38: Case Study 1: Time Series (McClarren)**
- Hohlraum laser simulation.
- Reducing 30 time points to 2 principal components.
- Interpretation: PC1 = Scale (overall height), PC2 = Plateau timing.

**Slide 39: Case Study 1: Visualization**
- [Show reconstruction of profiles using 2 PCs]
- Capturing 83% of variance with only 2 variables.

**Slide 40: Case Study 2: Hyperspectral Imaging (McClarren)**
- 2051 wavelengths reduced to 4 principal components.
- Identifying Distressed vs. Healthy leaves beyond the visible spectrum.

**Slide 41: Case Study 2: Color Reduction**
- Converting spectra to RGB.
- PCA uncovers structure that visible light alone misses.

**Slide 42: PCA on Images (Sandfeld)**
- Flattening 2D images into vectors.
- "Eigenfaces" / "Eigenmicrostructures".
- Most images are "rare" in the high-dim space; they occupy a small manifold.

**Slide 43: PCA for Noise Removal**
- Reconstructing $\mathbf{X} \approx \mathbf{U}_k \mathbf{S}_k \mathbf{V}_k^T$.
- Discarding small singular values (noise/higher modes).

**Slide 44: Standardizing Data before PCA**
- Importance of centering and scaling.
- PCA is sensitive to the scale of features.

**Slide 45: Summary: Part 4**
- Dimensionality reduction extracts low-dimensional "physics" from high-dimensional "data".

---

## Part 5: Clustering & Visualization (5 slides)

**Slide 46: K-means Clustering**
- Unsupervised learning to find natural groupings.
- Minimizing Loss $L = \sum ||\mathbf{x}_i - \boldsymbol{\mu}_k||^2$.

**Slide 47: Elbow Method for K**
- Choosing the number of clusters $K$ using $L \times K$ vs $K$.

**Slide 48: t-SNE: Visualizing High-Dim Data**
- t-Distributed Stochastic Neighbor Embedding.
- Maps high-dim clusters to 2D/3D.
- Preserves local structures (neighbors).

**Slide 49: Why the "t"? (McClarren Ch 4.4)**
- Normal distribution in high-dim space.
- Cauchy/t-distribution (long tails) in low-dim space.
- Long tails "push" clusters apart for better visualization.

**Slide 50: Example: Fashion MNIST**
- [Show t-SNE archipelago of clothing types]
- Uncovering structure based solely on pixel values (e.g., light vs dark shoes).
