# Materials Genomics Unit 11 — Clustering vs Discovery in Materials Spaces

## Core teaching claim
Clustering is a powerful exploratory tool for revealing hidden structures in materials data, but it is not a "discovery proof" in itself. In Materials Genomics, we distinguish between finding clusters (unsupervised organization) and discovering materials (actionable scientific claims). True discovery requires connecting cluster structures to physical hypotheses, quantifying uncertainty, and validating candidates through synthesis or simulation.

## 1. The Clustering Objective vs. the Discovery Objective
Clustering seeks to group similar objects based on a metric (e.g., Euclidean distance in descriptor space). Discovery seeks to identify new materials with specific performance targets. As Murphy (25.1) notes, clustering is about grouping similar objects, but in materials science, "similar" must be physically meaningful to lead to discovery.

## 2. Similarity and Dissimilarity in Materials Spaces (Murphy 25.1.1)
How we measure distance determines the clusters we find. For real-valued materials features (e.g., lattice constants), we use squared Euclidean distance. For categorical features (e.g., crystal system), we use Hamming distance. The choice of dissimilarity matrix $\mathbf{D}$ is the most critical modeling decision in unsupervised materials discovery.

## 3. K-Means: The "Hard Assignment" Baseline (Bishop 9.1)
K-Means partitions the materials space into $K$ disjoint sets by minimizing the distortion measure $J$:
$$
J = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2
$$
In materials terms, each $\boldsymbol{\mu}_k$ is a **prototype** representing a family of compounds. K-Means is a "hard" assignment: a material belongs to exactly one cluster.

## 4. The Iterative EM Logic of Clustering (Bishop 9.1)
K-Means proceeds in two steps:
1. **E-step**: Assign each material to the nearest prototype $\boldsymbol{\mu}_k$.
2. **M-step**: Re-calculate prototypes as the mean of assigned materials.
This iterative refinement allows us to "discover" the central tendencies of chemical families without knowing their labels in advance.

## 5. Hierarchical Clustering: Materials Taxonomies (Murphy 25.1)
Hierarchical methods create a nested tree (dendrogram) of partitions. This is particularly useful for materials science because it reflects the hierarchical nature of chemistry—from broad classes (metals vs. insulators) to specific structural motifs (perovskites vs. spinels). It is deterministic and doesn't require pre-specifying $K$.

## 6. Density-Based Clustering: Finding the "Islands of Stability"
Algorithms like DBSCAN (Neuer 5.3) find clusters based on local density. In Materials Genomics, these dense regions often correspond to "islands of stability" in the chemical manifold. Points not belonging to any cluster are flagged as **noise** or **outliers**, which may be the most interesting discovery candidates.

## 7. Gaussian Mixture Models (GMM): Soft Assignments (Bishop 9.2)
GMMs provide a probabilistic view of clustering. Instead of hard assignments, each material has a probability of belonging to each cluster. This "soft assignment" is essential for materials discovery because many compounds exist at the boundary between different phases or structure types.

## 8. Principal Component Analysis (PCA) as a Clustering Preprocessor (McClarren 4.1)
SVD and PCA reduce high-dimensional descriptors into uncorrelated variables. We often cluster in the PCA space rather than the raw space to avoid the "curse of dimensionality" and to focus on the axes that explain the most variation in the dataset.

## 9. t-SNE and UMAP: Visualizing Materials Clusters (Neuer 5.4)
t-SNE maps high-dimensional data into 2D or 3D by preserving local neighborhood probabilities. It is the most common tool for "seeing" materials clusters. However, as Neuer warns, t-SNE is slow and not always reproducible. UMAP is often preferred for maintaining more global structure.

## 10. The Trap of "Visual Discovery" (Neuer 5.4.3)
A "pretty" t-SNE plot with clear clusters is not proof of physical discovery. Clusters can be artifacts of the projection, the initialization, or the specific hyperparameters (like perplexity). Discovery claims must be backed by quantitative metrics, not just visual inspection.

## 11. Validating Clusters: Purity and Rand Index (Murphy 25.1.2)
We validate unsupervised clusters against known labels (e.g., crystal system) using Purity and the Rand Index. If the clusters align well with known physics, we gain confidence that the unsupervised model has captured the underlying "Genomic" rules of the material.

## 12. Artifact Clusters: The Source Bias Problem
In Materials Genomics, clusters often form around **dataset sources** (e.g., all MP structures in one cluster, all OQMD in another) or **simulation artifacts** (e.g., different DFT functionals). These are "artifact clusters" that hide the true chemistry. We must use domain-adversarial techniques or normalization to remove these non-physical signals.

## 13. Raw Descriptor Space vs. Latent Space Clustering
Clustering in Unit 4 (Raw) vs. Unit 10 (Latent):
- **Raw**: Clusters based on human-selected features.
- **Latent**: Clusters based on the model's learned representation.
Latent space clustering is often more powerful because it works on the "unwrapped" materials manifold.

## 14. Novelty vs. Anomaly vs. Noise (Neuer 5.5.3)
- **Noise**: Low-density points with no structure (sampling errors).
- **Anomaly**: Points that violate the learned reconstruction rule (potential errors).
- **Novelty**: Points that are far from known clusters but physically plausible (discovery candidates).
Distinguishing these three is the core task of discovery logic.

## 15. The Discovery Loop: From Cluster to Candidate
How a cluster becomes a discovery:
1. Identify a cluster with promising property proxies.
2. Extract the prototype (mean) of that cluster.
3. Perform latent traversal (Unit 10) around the prototype.
4. Validate candidates via high-throughput DFT or experiment.

## 16. Outliers as Discovery Targets
Often, the most "valuable" material is the one that *doesn't* fit in. An outlier in a stability cluster might be a metastable phase with extraordinary properties. Clustering helps us find these "black swans" by defining what "normal" looks like.

## 17. The Role of Domain Knowledge (Sandfeld 2.2)
Unsupervised clustering is "blind." We must use domain knowledge to interpret the results. If a cluster contains both a superconducting cuprate and a common insulator, the representation is likely missing critical physics (e.g., electronic correlations).

## 18. Uncertainty-aware Discovery (Preview of Unit 12)
Every cluster assignment has an associated uncertainty. In regions of the materials space where data is sparse, cluster boundaries are "fuzzy." Discovery logic must account for this epistemic uncertainty before committing to expensive simulations.

## 19. Summary: Clustering as a Hypothesis Generator
- Clustering organizes materials data into prototypes and families.
- It is an exploratory starting point, not a final conclusion.
- Success is measured by scientific utility, not just silhouette score.
- The path to discovery goes through validation and uncertainty quantification.

## 20. Exercise: Clustering Spectral vs. Structural Data
Students will perform K-Means and DBSCAN on both leaf spectra (McClarren) and crystal descriptors. They will compare validity indices, visualize the results with UMAP, and identify "artifact clusters" vs. "physical clusters."
