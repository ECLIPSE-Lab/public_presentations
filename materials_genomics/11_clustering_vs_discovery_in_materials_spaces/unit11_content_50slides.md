# Materials Genomics Unit 11 — 50-Slide Content Pack (Clustering vs. Discovery)

## Core teaching claim
Clustering is a powerful exploratory tool for revealing hidden structures in materials data, but it is not a "discovery proof" in itself. In Materials Genomics, we distinguish between finding clusters (unsupervised organization) and discovering materials (actionable scientific claims). True discovery requires connecting cluster structures to physical hypotheses, quantifying uncertainty, and validating candidates through synthesis or simulation.

---

## Part A: Concepts and Objectives (0–20 min)

### 1. Introduction: Clustering in the Discovery Loop
- Why cluster? Materials space is too large to explore exhaustively.
- Clustering as a "coarse-graining" step: reduce 10^5 structures to 10^2 families.
- Role: Organize the unlabeled data backlog before selecting candidates.

### 2. Learning Outcomes for Unit 11
- Explain hard vs. soft assignment logic.
- Implement K-Means, DBSCAN, and GMM on materials descriptors.
- Diagnose artifact clusters (source-bias, scaling issues).
- Connect cluster outliers to discovery novelty.

### 3. Clustering vs. Discovery: The Gap
- Clustering: $x \to \text{cluster label } c$. (Exploratory)
- Discovery: $x \to \text{high performance } y$. (Predictive/Actionable)
- The bridge: Clusters must correlate with physical mechanisms (Unit 4/5).

### 4. Similarity and Dissimilarity: The Choice of Metric (Murphy 25.1.1)
- Euclidean distance: Standard for continuous features (Unit 4).
- Hamming distance: Standard for categorical features (Unit 2).
- Mahalanobis distance: Accounting for feature correlations.

### 5. Why the Metric is the Most Important Choice
- Scaling sensitivity: If density is in g/cm³ and volume is in Å³, which dominates the cluster?
- Feature weighting: Giving more weight to electronegativity vs. atomic radius changes the cluster map.
- "Clustering is only as good as the metric space it lives in."

### 6. K-Means: The "Hard Assignment" Baseline (Bishop 9.1)
- Objective: Minimize the distortion measure $J$ (sum of squared distances to prototypes).
- Assumptions: Clusters are spherical, isotropic, and roughly equal in size.
- Materials risk: Many chemical families are elongated or anisotropic.

### 7. The EM Algorithm for K-Means: E-step
- Assignment phase: Fix cluster centers $\boldsymbol{\mu}_k$, assign each $x_n$ to the nearest one.
- Voronoi partitioning: K-Means effectively tiles the materials space with Voronoi cells.

### 8. The EM Algorithm for K-Means: M-step
- Update phase: Fix assignments $r_{nk}$, update $\boldsymbol{\mu}_k$ to the mean of assigned materials.
- Convergence: Guaranteed to decrease $J$ at every step until a local minimum.

### 9. K-Means Limitations in Materials Space
- Sensitivity to outliers (one "crazy" structure can pull the whole cluster center).
- Difficulty with clusters of different densities (e.g., common oxides vs. rare nitrides).
- The need to pre-specify $K$ (How many materials families are there?).

### 10. Choosing $K$: The Elbow Method and Silhouette Score
- Elbow: Plot $J$ vs. $K$ and look for the "kink."
- Silhouette: Measure how well-separated and cohesive clusters are.
- Materials context: $K$ is often determined by known crystal structure archetypes.

---

## Part B: Methods and Manifolds (20–45 min)

### 11. Mixture Models: The Probabilistic View (Bishop 9.2)
- Gaussian Mixture Model (GMM) assumes data comes from $K$ Gaussian distributions.
- $\pi_k$: Mixing coefficients (prior probability of a cluster).
- $\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$: Mean and covariance of the cluster.

### 12. Soft Assignments and Responsibilities
- Instead of hard labels, we get $\gamma(z_{nk})$: the probability that $x_n$ belongs to cluster $k$.
- Use case: A material that sits between two structure types (e.g., near a phase boundary).

### 13. Hierarchical Clustering: Dendrograms (Murphy 25.1)
- Agglomerative (bottom-up) vs. Divisive (top-down).
- Linkage criteria: Single, complete, and average linkage.
- Benefit: Reveals "taxonomies" of materials, from broad classes to specific motifs.

### 14. Dendrogram Cutting and Discovery Strategy
- Cutting high: Broad material classes (e.g., Halides vs. Chalcogenides).
- Cutting low: Fine-grained structural nuances (e.g., different tilting patterns in perovskites).
- Interactive exploration: Drifting through the tree to find interesting sub-families.

### 15. Density-Based Clustering: DBSCAN (Neuer 5.3)
- Idea: Clusters are continuous high-density regions separated by low-density areas.
- Hyperparameters: $\epsilon$ (epsilon distance) and `min_samples`.
- No need to specify $K$ in advance.

### 16. DBSCAN: Finding "Islands of Stability"
- Dense regions $\approx$ stable chemistry/structure combinations.
- Sparse regions $\approx$ unstable or unexplored chemistry.
- Identifying "valleys" in the chemical manifold.

### 17. SVD and PCA as Preprocessors (McClarren 4.1)
- High-dimensional clustering fails due to "distance concentration" (all points look equidistant).
- SVD/PCA extracts the main axes of variation.
- Clustering in PCA-space (usually top 10-20 dims) is more stable and physically meaningful.

### 18. Variance Explained vs. Clustering Resolution
- Dropping too many PCA components might lose the structural nuance that defines a cluster.
- Balancing dimensionality reduction with clustering "granularity."

### 19. t-SNE: Preserving Local Neighborhoods (Neuer 5.4)
- Mapping to 2D for visualization.
- t-SNE preserves the probability that point $i$ is a neighbor of point $j$.
- KL-divergence minimization: The cost function for visual organization.

### 20. UMAP: Preserving Global Structure
- Uniform Manifold Approximation and Projection (UMAP).
- Generally faster and better at preserving the relative positions of distant clusters.
- The current standard for visualizing large materials databases (Materials Project, OQMD).

---

## Part C: Domain Translation and Modeling (45–65 min)

### 21. Artifact Clusters: The Source Bias Problem
- Case: Clustering a dataset with structures from both VASP and Quantum Espresso.
- Result: Two giant clusters representing the software used, not the chemistry.
- Lesson: Normalization must be done per-source to find true materials trends.

### 22. Scaling Artifacts: The "Unit" Trap
- If atomic radius is in Å (order 1) and formation energy is in eV (order -10), radius will dominate the cluster.
- Feature standardization (Z-score) is mandatory for unbiased clustering.

### 23. Clustering Raw Descriptors (Unit 4) vs. Latent Codes (Unit 10)
- Raw: Interpretable features, but subject to human selection bias.
- Latent: Unwrapped manifold, often captures "deep" structural relationships descriptors miss.
- Recommendation: Perform both and compare the Rand Index.

### 24. Visualization Hallucinations (Neuer 5.4.3)
- t-SNE can show clusters where there is only a continuous trend.
- t-SNE distance is NOT physical distance.
- Visual inspection is the beginning of discovery, not the proof.

### 25. External Validation: Purity and Rand Index (Murphy 25.1.2)
- Purity: Do clusters align with known crystal systems?
- Adjusted Rand Index (ARI): Measuring similarity between unsupervised clusters and known categories.
- High ARI against space groups validates the "structural awareness" of the descriptors.

### 26. Case Study: Clustering the Materials Project (MP)
- Objective: Find new "prototypes" in a database of 150,000 materials.
- Methodology: SOAP descriptors (Unit 6) + UMAP + DBSCAN.
- Result: Discovery of rare coordination environments that were not labeled in the database.

### 27. Case Study: Clustering 2004 Cars Data (Murphy 12.2)
- A non-materials example to build intuition.
- Axes: Price vs. Fuel Efficiency vs. Size.
- Clusters: Economy, Luxury, SUVs.
- Analogy: In MG, we find clusters for "Hard, Stable, Conductive" materials.

### 28. Case Study: Compressing Leaf Spectra (McClarren 4.2)
- Using SVD + K-Means to identify plant species from hyperspectral images.
- The latent factors correspond to chlorophyll levels and water content.
- Bridge: In MG, we cluster XRD or EELS spectra to find structural phases.

### 29. The Discovery Objective: Selecting the "Golden Cluster"
- Once clusters are found, we check property statistics:
- Which cluster has the highest mean bandgap?
- Which cluster is the most thermodynamically stable?
- This moves from "grouping" to "targeting."

### 30. Outlier Detection: The Discovery of the "Black Swan"
- Discovery often happens at the fringe.
- An outlier in a stability cluster might be a metastable phase with extraordinary performance (e.g., diamond vs. graphite).
- Strategy: Exhaustively screen the DBSCAN "noise" points.

---

## Part D: Validation and Failure Analysis (65–80 min)

### 31. Uncertainty in Cluster Assignments
- GMM "responsibilities" as uncertainty proxies.
- If $P(\text{Cluster 1}) = 0.51$ and $P(\text{Cluster 2}) = 0.49$, the discovery claim is weak.
- Mapping "fuzzy" regions of the materials manifold.

### 32. Robustness to Dataset Size
- Do the clusters remain stable if we remove 10% of the data?
- "Bootstrap" clustering: Run the algorithm on different subsets and measure stability.
- Instability signals that the discovery claim is sensitive to sampling noise.

### 33. The Silhouette Score Trap
- A high silhouette score only means clusters are well-separated.
- It does NOT mean the separation is physically meaningful.
- Example: A perfect silhouette score on "Source" clusters (Slide 21).

### 34. Feature Importance for Clustering
- Which descriptor (Unit 4) is driving the cluster separation?
- Using Random Forest "importance" on cluster labels to interpret the discovery drivers.
- Connecting "unsupervised" results back to "supervised" features.

### 35. Over-Clustering: Finding Structure in Noise
- Clustering algorithms *will* find clusters in random noise if forced.
- Permutation test: Run the algorithm on shuffled data. If you still find "clusters," your discovery is an artifact.

### 36. Interpreting Cluster Prototypes
- A prototype (mean) $\boldsymbol{\mu}_k$ might not correspond to a physical material.
- Finding the "Medoid": The actual material in the dataset closest to the cluster center.
- Using the Medoid as the "representative" of the discovered family.

### 37. Bridging to Latent Traversals (Unit 10)
- Once a discovery cluster is found, perform a traversal around its Medoid.
- Explore the "neighborhood" of the cluster to find more candidates.
- Unsupervised clustering sets the "starting point" for local discovery.

### 38. Cluster-Target Correlation
- Calculate the correlation between cluster membership and property $y$.
- If a cluster "explains" property variation, the structural family is a valid discovery lead.

### 39. Handling Categorical Descriptors
- Mixing continuous (atomic radius) and categorical (space group) features.
- Using Gower's distance or Multiple Correspondence Analysis (MCA).
- Preventing categorical features from drowning out structural nuances.

### 40. The Role of Normalization and Whitening
- Why "whitening" (removing correlations) improves clustering stability.
- Preventing redundant features from being double-counted in the distance metric.

### 41. Anomaly Detection vs. Novelty Detection
- Anomaly: Likely a simulation error (Unit 9).
- Novelty: A physically plausible "new" structure type.
- Discovery logic: Filter anomalies, prioritize novelties.

### 42. Domain-Aware Outlier Filtering
- Is the outlier just a material with an extremely large unit cell?
- Normalizing by "Complexity" to ensure outliers are chemically novel, not just computationally complex.

### 43. Failure Mode: Shortcut Learning from Composition
- If composition is included in the descriptors, clustering often just recreates the Periodic Table.
- Discovery goal: Find clusters driven by *structure* (bonding, symmetry), not just stoichiometry.

### 44. Failure Mode: Feature Saturation
- Using too many descriptors (Unit 4) makes every material look unique.
- "The Curse of Dimensionality" leads to a single giant cluster or $N$ single-point clusters.

---

## Part E: Synthesis and Exercise (80–90 min)

### 45. Summary of the Discovery Logic
- Clustering: Organizes structure.
- Prototypes: Define families.
- Outliers: Signal novelty.
- Metrics: Validate physics.

### 46. Next Unit Bridge: Uncertainty-Aware Discovery
- Unit 11 was about "hard" boundaries.
- Unit 12 will replace boundaries with continuous uncertainty maps (Gaussian Processes).
- Rule: Cluster for exploration, model uncertainty for exploitation.

### 47. Exercise Task 1: Spectral Clustering
- Dataset: Leaf spectra from McClarren.
- Task: Compare K-Means vs. DBSCAN on SVD-reduced spectra.
- Deliverable: Silhouette score vs. $K$ plot.

### 48. Exercise Task 2: Structural Clustering
- Dataset: Materials Project formation energy subset.
- Task: Cluster structures using simple composition vectors vs. SOAP embeddings.
- Deliverable: Rand Index against known crystal systems.

### 49. Exercise Task 3: Artifact Detection
- Task: Intentionally mis-scale one feature and observe how the cluster map collapses.
- Deliverable: Before/after UMAP visualization.

### 50. Exam Checklist: Evidence-backed Discovery
- Can you explain why t-SNE distances are not physical?
- How do you diagnose "source bias" in a cluster map?
- What is the difference between a centroid and a medoid?
- When is DBSCAN superior to K-Means for materials discovery?
- Why is validation (Rand Index) necessary for unsupervised results?
