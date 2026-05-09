# MFML Unit 10 — 50-Slide Content Pack (English)

## Unit theme
**Latent Spaces, Embeddings, and Nonlinear Dimensionality Reduction**

## Core source mapping (book-priority aligned)
- **Neuer (2024)**: Ch. 5.5.1–5.5.3 (autoencoder latent space, anomaly detection), Ch. 5.5.5 (classification in latent space), Ch. 5.5.7 (conditional probability in latent space).
- **McClarren (2021)**: Ch. 4.4 / Ch. 5 (t-SNE, Gaussian vs Cauchy, Fashion MNIST case study).
- **Murphy (2012)**: Ch. 12.1–12.3 (factor analysis, probabilistic PCA, latent linear models).
- **Bishop (2006)**: Ch. 12 (kernel PCA, kernel trick).

---

## Slide-by-slide content (target: 50)

### Block A — Latent spaces: concept and motivation (Slides 1-10)
1. **Title + Unit 10 positioning**
   - Unit 9: autoencoders learn representations. Unit 10: what is the structure of the latent space?
2. **Learning outcomes for Unit 10**
   - Latent space formalism, t-SNE derivation, UMAP/kernel PCA comparison, anomaly detection in latent space.
3. **What is a latent space?**
   - A low-dimensional space that captures the essential variation of high-dimensional data; "latent" = hidden.
4. **Latent space vs feature space vs PCA eigenspace**
   - Feature space: raw measurements. PCA eigenspace: linear projection. Latent space: potentially nonlinear embedding.
5. **Why latent spaces matter**
   - Compression, visualization, generation, anomaly detection, downstream prediction — all operate in latent space.
6. **Recap: autoencoder as latent space constructor**
   - Encoder maps x → z; bottleneck dimension defines the latent space dimensionality.
7. **Embeddings: points in latent space**
   - Each data point x_i gets an embedding z_i = f(x_i). The collection {z_i} forms a point cloud in latent space.
8. **What makes a good latent space?**
   - Similar inputs → nearby embeddings. Different inputs → distant embeddings. Smooth interpolation between points.
9. **Latent space geometry: clusters and manifolds**
   - Data classes often form clusters in latent space; continuous variation creates manifold structure.
10. **Roadmap of today's 90 min**
    - AE latent space geometry → t-SNE → UMAP and kernel PCA → anomaly detection and trust.

### Block B — Autoencoder latent space structure (Slides 11-18)
11. **Autoencoder latent neurons: what do they encode?**
    - Each latent dimension captures a learned factor of variation (e.g., signal width, position, amplitude).
12. **Visualizing 2D latent spaces**
    - When d_z = 2, plot embeddings directly; color by class or property to reveal structure.
13. **Interpreting latent space clusters**
    - Distinct clusters indicate separable data modes; overlap indicates ambiguity.
14. **Latent space interpolation**
    - Linearly interpolate between two latent codes: z_mix = alpha*z_1 + (1-alpha)*z_2. Decode to see smooth transitions.
15. **Latent space arithmetic**
    - Concept vectors: z_target = z_source + (z_concept_A - z_concept_B). Analogy-based reasoning in latent space.
16. **The issue of latent space structure**
    - Standard AEs produce unstructured latent spaces — "holes" where decoding produces nonsense.
17. **Preview: variational autoencoders (VAEs)**
    - VAEs impose a prior (Gaussian) on the latent space → continuous, structured, generative.
18. **Checkpoint: latent space quality**
    - "Your 2D latent space has a large empty region between clusters. What does this mean?"

### Block C — t-SNE: theory and application (Slides 19-30)
19. **t-SNE: purpose and overview**
    - t-Distributed Stochastic Neighbor Embedding: a nonlinear method for 2D/3D visualization of high-dimensional data.
20. **Step 1: Gaussian similarities in high dimensions**
    - For each pair (i,j), compute conditional probability p_{j|i} based on Gaussian distance.
21. **Perplexity parameter**
    - Perplexity controls the effective number of neighbors; adapts the Gaussian bandwidth per point.
22. **Step 2: Student-t similarities in low dimensions**
    - In the embedding, use a Student-t distribution (heavy tails) instead of Gaussian.
23. **The crowding problem**
    - In low dimensions, moderate-distance points crowd together. Student-t tails allow distant points to spread out.
24. **Step 3: KL divergence minimization**
    - Minimize KL(P || Q) where P = high-dim similarities, Q = low-dim similarities. Gradient descent on embedding coordinates.
25. **t-SNE: what it preserves**
    - Local neighborhood structure: nearby points stay nearby. Global distances are NOT preserved.
26. **t-SNE: common misinterpretations**
    - Cluster sizes have no meaning. Distances between clusters have no meaning. Only within-cluster neighborhoods are reliable.
27. **t-SNE hyperparameters**
    - Perplexity (5-50 typical), learning rate, number of iterations. Results can vary — run multiple times.
28. **t-SNE on Fashion MNIST: archipelago interpretation**
    - Different clothing categories form islands; similar categories (shirt/pullover) overlap. Archipelago = successful separation.
29. **t-SNE: strengths and weaknesses**
    - Strengths: excellent local structure visualization. Weaknesses: non-parametric, slow (O(N^2)), no new-point mapping.
30. **Checkpoint: t-SNE interpretation**
    - "Two clusters appear far apart in a t-SNE plot. Does this mean they are far apart in the original space?"

### Block D — UMAP, kernel PCA, and comparison (Slides 31-38)
31. **UMAP: Uniform Manifold Approximation and Projection**
    - Like t-SNE but based on topological data analysis; faster, better preserves global structure.
32. **UMAP vs t-SNE**
    - UMAP: faster (approximate nearest neighbors), preserves more global structure, supports transform of new points.
33. **UMAP hyperparameters**
    - n_neighbors (local vs global), min_dist (tightness of clusters), n_components (embedding dimension).
34. **Kernel PCA: the kernel trick for nonlinear structure**
    - Replace dot products in PCA with kernel evaluations: k(x_i, x_j) = phi(x_i)^T phi(x_j).
35. **Common kernels**
    - RBF (Gaussian), polynomial, sigmoid. The kernel implicitly maps data to a high-dimensional feature space.
36. **Kernel PCA: strengths and limitations**
    - Principled, eigenvalue-based. Limited: does not scale well to large N; no density estimation.
37. **Comparison: PCA, kernel PCA, t-SNE, UMAP**
    - PCA: linear, fast, global. kPCA: nonlinear, moderate cost. t-SNE: nonlinear, local focus. UMAP: nonlinear, balanced.
38. **When to use which**
    - Exploration/visualization: t-SNE or UMAP. Feature extraction: PCA or kPCA. Downstream ML: autoencoder latent space.

### Block E — Anomaly detection, trust, and summary (Slides 39-50)
39. **Anomaly detection in latent space (Neuer)**
    - Normal data clusters in latent space; anomalies map to low-density regions.
40. **Strategy 1: reconstruction error revisited**
    - High reconstruction error → input differs from learned patterns → potential anomaly.
41. **Strategy 2: latent density estimation**
    - Fit a density model (Gaussian, GMM, KDE) to normal latent embeddings; low-density points are anomalous.
42. **Combining both strategies**
    - Flag a sample if reconstruction error is high OR latent density is low. Reduces false negatives.
43. **Conditional latent probabilities and trust (Neuer Ch. 5.5.7)**
    - For a new input, compute p(class | z). Low maximum probability → uncertain classification → low trust.
44. **Trust quantification in deployment**
    - Epistemic uncertainty in latent space flags inputs outside the training distribution.
45. **Materials example: spectral classification with trust scoring**
    - Classify material type from spectrum. Latent probability indicates confidence. Low-confidence samples sent to expert.
46. **Materials example: process monitoring via latent drift**
    - Track latent embeddings over time. Drift from normal cluster → process degradation.
47. **Lecture-essential vs exercise content split**
    - Lecture: latent space formalism, t-SNE/UMAP theory, anomaly detection, trust.
    - Exercise: AE latent visualization, t-SNE on Fashion MNIST, UMAP comparison, anomaly injection.
48. **Exercise setup summary**
    - Build AE on synthetic data, visualize latent space, run t-SNE and UMAP on Fashion MNIST subset, inject anomalies.
49. **Exam-aligned summary: 10 must-know statements**
    1. A latent space is a low-dimensional space capturing essential data variation.
    2. Autoencoder embeddings reveal cluster and manifold structure without labels.
    3. t-SNE preserves local neighborhoods but not global distances.
    4. The crowding problem motivates using Student-t distributions in low dimensions.
    5. UMAP is faster than t-SNE and better preserves global structure.
    6. Kernel PCA applies the kernel trick to perform nonlinear PCA.
    7. Anomaly detection combines reconstruction error and latent density.
    8. Conditional latent probabilities quantify classification trust.
    9. Latent space interpolation tests smoothness of learned representations.
    10. Visualization methods (t-SNE, UMAP) require careful interpretation — cluster sizes and inter-cluster distances are not reliable.
50. **References + reading assignment for next unit**
    - Required: Neuer Ch. 5.5, McClarren Ch. 4.4/5.
    - Optional: Murphy Ch. 12, Bishop Ch. 12.
    - Next unit: Unsupervised Learning — clustering, mixture models, and density estimation.

---

## Reusable equations (for slide math boxes)
- t-SNE conditional probability: \\(p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}\\)
- t-SNE low-dim similarity: \\(q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}\\)
- KL divergence: \\(\text{KL}(P\|Q) = \sum_{i \neq j} p_{ij} \log\frac{p_{ij}}{q_{ij}}\\)
- Kernel PCA: \\(K_{ij} = k(x_i, x_j)\\), eigendecompose centered kernel matrix
- RBF kernel: \\(k(x_i, x_j) = \exp(-\gamma\|x_i - x_j\|^2)\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: latent space concept, t-SNE derivation, UMAP overview, kernel PCA, anomaly detection/trust design.
- **Exercise**: AE latent visualization, t-SNE/UMAP on Fashion MNIST, anomaly injection, comparison experiments.

## Reading assignment after Unit 10
- Neuer Ch. 5.5
- McClarren Ch. 4.4, 5
- Murphy Ch. 12 (skim)
- Bishop Ch. 12 (conceptual)
