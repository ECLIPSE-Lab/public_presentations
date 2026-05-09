# MFML Unit 11 — 50-Slide Content Pack (English)

## Unit theme
**Unsupervised Learning: Clustering, Mixture Models, and the EM Algorithm**

## Core source mapping (book-priority aligned)
- **Murphy (2012)**: Ch. 11.1–11.5 (latent variable models, mixture models, EM algorithm, lower bound, model selection).
- **Neuer (2024)**: Ch. 5.3 (K-Means algorithm, cluster variance, batch K-Means).
- **McClarren (2021)**: Ch. 4.3 (K-Means, loss function, choosing K).
- **Bishop (2006)**: Ch. 9.2–9.4 (GMM, EM derivation, lower bound view).

---

## Slide-by-slide content (target: 50)

### Block A — Motivation and the unsupervised paradigm (Slides 1-8)
1. **Title + Unit 11 positioning**
   - From latent spaces (Unit 10) to discovering structure without labels.
2. **Learning outcomes for Unit 11**
   - K-Means, GMM, EM algorithm, lower bound argument, engineering applications.
3. **Supervised vs unsupervised learning**
   - Supervised: (x, y) pairs. Unsupervised: x only — discover hidden structure.
4. **Why unsupervised learning matters in engineering**
   - Labels are expensive (destructive testing), unavailable (sensor streams), or undefined (exploratory analysis).
5. **Types of unsupervised learning**
   - Clustering, density estimation, dimensionality reduction (Unit 9-10), generative models.
6. **Clustering: the core task**
   - Partition data into groups such that within-group similarity is high and between-group similarity is low.
7. **What defines a "good" cluster?**
   - Compact (low within-cluster variance), separated (high between-cluster distance), interpretable.
8. **Roadmap of today's 90 min**
   - K-Means → GMM → EM algorithm → lower bound → variants → applications.

### Block B — K-Means clustering (Slides 9-18)
9. **K-Means objective**
   - Minimize the total within-cluster sum of squared distances (distortion): J = sum_k sum_{i in C_k} ||x_i - mu_k||^2.
10. **K-Means algorithm: two alternating steps**
    - Assignment: assign each x_i to nearest centroid. Update: recompute centroids as cluster means.
11. **K-Means as coordinate descent**
    - Assignment step minimizes J w.r.t. assignments (fixed centroids). Update step minimizes J w.r.t. centroids (fixed assignments).
12. **K-Means convergence**
    - J decreases (or stays constant) at every step. Finite number of assignments → convergence guaranteed.
13. **K-Means: initialization matters**
    - Bad initialization → poor local minimum. K-Means++ initialization: spread initial centroids.
14. **K-Means: choosing K**
    - Elbow method: plot J vs K, look for diminishing returns. Silhouette score: measures cluster quality.
15. **K-Means: limitations**
    - Assumes spherical clusters of equal size. Hard assignments — no uncertainty.
16. **K-Means: worked example**
    - 2D data with 3 Gaussian blobs: initialize → assign → update → converge in 5 iterations.
17. **K-Medoids: robust variant**
    - Use actual data points as cluster centers (medoids); robust to outliers.
18. **Checkpoint: K-Means**
    - "K-Means splits an elongated cluster into two. Why?" — spherical assumption fails for non-spherical clusters.

### Block C — Gaussian Mixture Models and EM (Slides 19-34)
19. **From hard to soft clustering**
    - K-Means: each point belongs to exactly one cluster. GMM: each point has a probability of belonging to each cluster.
20. **Gaussian Mixture Model: definition**
    - p(x) = sum_{k=1}^K pi_k N(x | mu_k, Sigma_k). Mixing coefficients pi_k, means mu_k, covariances Sigma_k.
21. **GMM as a latent variable model**
    - Introduce latent indicator z_i in {1,...,K}: p(z_i = k) = pi_k, p(x_i | z_i = k) = N(x_i | mu_k, Sigma_k).
22. **Responsibilities: soft assignments**
    - r_{ik} = p(z_i = k | x_i) = pi_k N(x_i | mu_k, Sigma_k) / sum_j pi_j N(x_i | mu_j, Sigma_j).
23. **The log-likelihood for GMMs**
    - log L = sum_i log [sum_k pi_k N(x_i | mu_k, Sigma_k)]. The sum inside the log makes direct optimization hard.
24. **Why MLE is hard for mixtures**
    - No closed-form solution because the log of a sum cannot be decomposed.
25. **The EM algorithm: overview**
    - Iterate between: E-step (compute responsibilities) and M-step (update parameters).
26. **E-step: compute responsibilities**
    - r_{ik} = pi_k N(x_i | mu_k, Sigma_k) / sum_j pi_j N(x_i | mu_j, Sigma_j). "How much does cluster k explain point i?"
27. **M-step: update parameters**
    - N_k = sum_i r_{ik}. mu_k = (1/N_k) sum_i r_{ik} x_i. Sigma_k = (1/N_k) sum_i r_{ik} (x_i - mu_k)(x_i - mu_k)^T. pi_k = N_k / N.
28. **EM algorithm: pseudocode**
    - Initialize parameters. Repeat: E-step (responsibilities) → M-step (update mu, Sigma, pi) → check convergence.
29. **EM convergence**
    - The log-likelihood increases (or stays constant) at each iteration. Guaranteed convergence to a local maximum.
30. **The lower bound perspective**
    - EM maximizes a lower bound on log L using Jensen's inequality. Each E-step tightens the bound; each M-step increases it.
31. **Jensen's inequality and the auxiliary function**
    - log E[X] >= E[log X]. The auxiliary function Q(theta | theta_old) is the expected complete-data log-likelihood.
32. **K-Means as a special case of EM**
    - Set all Sigma_k = sigma^2 I and let sigma -> 0: soft assignments become hard, EM reduces to K-Means.
33. **GMM: worked example**
    - 2D data with 2 overlapping Gaussians: initialize → E-step → M-step → iterate → convergence.
34. **Checkpoint: EM understanding**
    - "After the E-step, what do the responsibilities tell you?" — the posterior probability of each cluster given each data point.

### Block D — Variants and model selection (Slides 35-40)
35. **Bernoulli mixture models**
    - For binary data: replace Gaussian with Bernoulli likelihood. EM updates become means of binary features per cluster.
36. **Model selection: choosing K**
    - BIC: log L - (K_params/2) log N. AIC: log L - K_params. Lower is better (penalizes complexity).
37. **BIC vs AIC**
    - BIC has stronger penalty (prefers simpler models for large N). AIC is closer to cross-validation.
38. **Initialization strategies for EM**
    - Random, K-Means initialization (common), multiple restarts with best log-likelihood.
39. **Singular covariance matrices**
    - When a cluster collapses to a single point, Sigma_k becomes singular. Fix: add small diagonal regularization.
40. **Checkpoint: model selection**
    - "BIC selects K=3 but you know there are 5 groups. What might be wrong?" — groups may overlap; BIC prefers parsimony.

### Block E — Engineering applications + summary (Slides 41-50)
41. **Application: image segmentation**
    - Model pixel colors as GMM; each cluster = one material/region. Segment by maximum responsibility.
42. **Application: vector quantization**
    - Replace each data point by its nearest centroid (codebook). Used in signal compression and materials fingerprinting.
43. **Application: regime identification in manufacturing**
    - Cluster sensor data over time; each cluster = one operating regime. Detect regime changes for process control.
44. **Application: anomaly detection via density**
    - Low p(x) under the fitted GMM flags anomalies. Principled threshold from density estimation.
45. **Clustering vs autoencoder embeddings**
    - Clustering operates directly on data; AE learns representations. Combining: cluster in latent space for best of both.
46. **Materials example: alloy phase identification**
    - Cluster X-ray diffraction patterns to identify crystallographic phases without prior labeling.
47. **Lecture-essential vs exercise content split**
    - Lecture: K-Means objective, GMM formulation, EM derivation, lower bound, model selection.
    - Exercise: K-Means implementation, EM for 2-component GMM, hard vs soft comparison.
48. **Exercise setup summary**
    - NumPy K-Means on synthetic clusters; EM for GMM; track log-likelihood; compare hard vs soft assignments.
49. **Exam-aligned summary: 10 must-know statements**
    1. K-Means minimizes within-cluster sum of squared distances.
    2. K-Means alternates between assignment and update steps (coordinate descent).
    3. A GMM models data as a weighted sum of Gaussian components.
    4. Responsibilities are posterior probabilities of cluster membership.
    5. The E-step computes responsibilities; the M-step updates parameters.
    6. EM monotonically increases the log-likelihood (convergence to local max).
    7. EM maximizes a lower bound on the observed-data log-likelihood.
    8. K-Means is a limiting case of EM with isotropic covariances and hard assignments.
    9. BIC/AIC balance model fit against complexity for choosing K.
    10. GMM-based density estimation enables principled anomaly detection.
50. **References + reading assignment for next unit**
    - Required: Murphy Ch. 11.1–11.5, Neuer Ch. 5.3.
    - Optional: McClarren Ch. 4.3, Bishop Ch. 9.2–9.4.
    - Next unit: Uncertainty in Predictions — Bayesian predictive distributions and Gaussian Processes.

---

## Reusable equations (for slide math boxes)
- K-Means objective: \\(J = \sum_{k=1}^K \sum_{i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2\\)
- GMM: \\(p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\\)
- Responsibility: \\(r_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}\\)
- M-step mean: \\(\boldsymbol{\mu}_k = \frac{\sum_i r_{ik}\mathbf{x}_i}{\sum_i r_{ik}}\\)
- BIC: \\(\text{BIC} = \log L - \frac{K_{\text{params}}}{2}\log N\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: K-Means objective and algorithm, GMM formulation, EM derivation, lower bound argument, model selection.
- **Exercise**: K-Means from scratch, EM implementation, log-likelihood tracking, hard vs soft comparison.
