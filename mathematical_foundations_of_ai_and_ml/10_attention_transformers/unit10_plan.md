# Unit 10 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: Units 1-9 completed (linear algebra, optimization, neural networks, representation learning)
- Assumed: PCA/SVD from Unit 2, neural network training from Units 4-5, probabilistic framing from Unit 8
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 10)
By the end of the unit, students can:
1. Define latent spaces formally and distinguish them from raw feature spaces and PCA eigenspaces.
2. Explain how autoencoders learn nonlinear compression and how the bottleneck layer defines a latent manifold.
3. Derive the t-SNE objective (Gaussian high-dim, Student-t low-dim, KL-divergence) and explain the crowding problem.
4. Compare t-SNE, UMAP, and kernel PCA as tools for nonlinear dimensionality reduction and visualization.
5. Use latent-space distributions for anomaly detection and interpret conditional latent probabilities for trust assessment.

## Book-aligned content mapping (priority order)
1. **Neuer (2024)**: Ch. 5.5.1-5.5.3 (autoencoder topology, latent space definition, anomaly detection), Ch. 5.5.5 (classification in latent space), Ch. 5.5.7 (conditional probability in latent space) -- primary source.
2. **McClarren (2021)**: Ch. 5 / Ch. 4.4 (t-SNE computation, Gaussian vs Cauchy distributions, Fashion MNIST case study, interpreting t-SNE archipelagos).
3. **Murphy (2012)**: Ch. 12.1-12.3 (factor analysis as latent linear model, probabilistic PCA, choosing latent dimensions, unidentifiability).
4. **Bishop (2006)**: Ch. 12 (kernel PCA, kernel trick for implicit feature spaces).

## 90-minute structure
- 0-10 min: Motivation -- why compress, what is a latent space, recap of PCA from Unit 2
- 10-25 min: Autoencoder topology, encoder/decoder split, nonlinear compression vs linear PCA
- 25-40 min: Latent space geometry -- embeddings, manifold structure, interpretability of latent neurons
- 40-55 min: t-SNE -- Gaussian neighborhoods, Student-t in low-dim, KL divergence, crowding problem
- 55-65 min: UMAP overview and kernel PCA -- the kernel trick for nonlinear structure
- 65-80 min: Anomaly detection in latent space, conditional latent probabilities, trust quantification
- 80-88 min: Summary -- latent spaces as bridge between unsupervised discovery and supervised prediction
- 88-90 min: Exercise handoff + reading assignment for Unit 11

## Exercise (90 min)
- Build a simple autoencoder in PyTorch (or Keras) on synthetic Gaussian pulse data (following Neuer's setup)
- Visualize the latent space activations and identify cluster structure without labels
- Inject an unknown anomaly and detect it via reconstruction error and latent-space position
- Apply t-SNE (scikit-learn) to Fashion MNIST subset and interpret the resulting archipelago
- Bonus: compare t-SNE vs UMAP on the same dataset and discuss preservation of global vs local structure

## Assessment alignment
- Written exam prep: formal definitions of latent space, encoder/decoder, t-SNE objective
- Every unit introduces 3-5 exam-relevant "must-know" statements
- Exercise connects directly to exam questions on anomaly detection logic and dimensionality reduction comparison

## Slide production notes
- Style: follow data_science_for_em revealjs pattern
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
