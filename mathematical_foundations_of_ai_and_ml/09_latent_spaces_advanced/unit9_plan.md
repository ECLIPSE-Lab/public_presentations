# Unit 9 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: 2 semesters math, 1 physics, some chemistry; Units 1-8 completed (including SVD/PCA from Unit 2, neural networks from Units 6-7)
- Assumed: familiarity with PCA as linear dimensionality reduction, feedforward neural networks, loss functions, gradient-based optimization
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 9)
By the end of the unit, students can:
1. Explain the manifold hypothesis and why real-world high-dimensional data admits low-dimensional representations.
2. Contrast handcrafted feature engineering with learned representations and articulate when each approach is appropriate.
3. Derive the autoencoder objective as identity-mapping through a bottleneck and relate it mathematically to PCA as a special (linear) case.
4. Describe convolutional autoencoder architectures (strided convolutions, transposed convolutions) and denoising autoencoders, including their loss formulations.
5. Apply autoencoders to engineering tasks: data compression, anomaly detection (reconstruction error and latent-space outliers), and feature extraction for downstream supervised learning.

## Book-aligned content mapping (priority order)
1. **Neuer (2024) Ch. 5.5** (primary): autoencoder topology, encoder/decoder split, latent space definition, nonlinear compression vs PCA, anomaly detection (reconstruction quality + latent-space outliers), stacking and combination with downstream methods, uncertainty behavior.
2. **McClarren (2021) Ch. 8** (primary): fully connected autoencoders, hourglass structure, codes/latent variables, convolutional autoencoders (strided convolutions, transposed convolutions), noise reduction, case study (leaf spectra compression, physics simulation data reduction, LSTM with AE inputs).
3. **Murphy (2012) Ch. 12** (supplementary): factor analysis as latent linear model, PPCA as probabilistic PCA, connection between linear latent models and autoencoders, unidentifiability and rotation ambiguity.
4. **Bishop (2006) Ch. 12** (supplementary): continuous latent variables, PCA as maximum-variance projection, probabilistic PCA framework.

## 90-minute structure
- 0-10 min: Motivation — why features matter more than algorithms; manifold hypothesis
- 10-25 min: Handcrafted features vs learned representations; historical context and limitations
- 25-45 min: Autoencoder formalism — encoder, decoder, bottleneck, loss; connection to PCA
- 45-60 min: Convolutional autoencoders — strided convolutions, transposed convolutions; denoising AE
- 60-75 min: Applications — data compression, anomaly detection (two strategies), feature extraction for downstream tasks
- 75-85 min: Engineering case studies — leaf spectra, physics simulations, industrial anomaly detection
- 85-90 min: Summary + exercise handoff + reading assignment

## Exercise (90 min)
- **Part 1 (30 min):** Build a fully connected autoencoder in PyTorch for synthetic 1D signals (Gaussian pulses with varying width/position, following Neuer 5.5). Train on normal cases, visualize latent space, confirm reconstruction quality.
- **Part 2 (30 min):** Reduce bottleneck dimension systematically (e.g., from 10 to 2) and plot reconstruction error vs latent dimension. Compare to PCA reconstruction with same number of components. Observe where nonlinear AE outperforms linear PCA.
- **Part 3 (20 min):** Inject an anomaly (unseen signal shape) and detect it via (a) reconstruction error threshold and (b) latent-space outlier detection. Discuss which method is more robust.
- **Bonus (10 min):** Add noise to test inputs and observe denoising behavior of the trained AE. Optionally retrain with noisy inputs and clean targets (denoising AE) and compare.

## Assessment alignment
- Written exam prep: autoencoder loss formulation, encoder/decoder decomposition, latent space dimensionality argument, PCA-AE relationship.
- Every unit introduces 3-5 exam-relevant "must-know" statements.

## Slide production notes
- Style: follow revealjs pattern from Unit 1 (01_intro.qmd)
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
- Diagrams needed: hourglass topology, encoder/decoder split, manifold illustration, strided convolution, reconstruction error for anomaly detection
