# Materials Genomics Unit 10 — 50-Slide Content Pack (Latent Spaces of Materials)

## Core teaching claim
Latent spaces are more than just compressed representations; they are searchable materials manifolds where structural and chemical relationships are "unwrapped" into a continuous coordinate system. By interpreting these spaces through traversals and neighborhood analysis, we can identify anomalies, interpolate between known phases, and provide the foundation for generative materials design.

---

## Part A: Foundations and Workflows (0–20 min)

### 1. Introduction: The "Genomic" Coordinate System
- Why do we need a latent space? Raw features are high-dimensional and non-interpretable.
- Goal: Find a low-dimensional space that "explains" the materials data.
- The "Genomic" analogy: mapping structural motifs to computable coordinates.

### 2. Learning Outcomes for Unit 10
- Define the latent space as a low-rank manifold of materials structure.
- Explain the role of the Factor Loading Matrix.
- Distinguish between reconstruction loss and latent novelty.
- Implement traversals to "decode" the physical meaning of latent axes.

### 3. Recap: Representation Learning (Unit 9)
- Review: $\mathcal{E}(x) = z$, $\mathcal{D}(z) = \hat{x}$.
- The Bottleneck Principle: Forcing information through a narrow pipe.
- Connection: Today we focus on the "pipe" itself (the space $\mathbb{L}$).

### 4. Defining the Latent Space (Murphy 12.1)
- The space of variables $z \in \mathbb{R}^L$ that are "hidden" but drive the observed $x$.
- Probabilistic view: We assume $z$ is generated from a simple prior (e.g., Gaussian).
- Mapping the "intrinsic dimensionality" of crystal structures.

### 5. Latent Models as Low-Rank Parameterization (Murphy 12.1.1)
- Factor Analysis (FA) models the data covariance as: $\Sigma = WW^T + \Psi$.
- Reducing $O(D^2)$ parameters to $O(LD)$.
- Why "low-rank" matters: it filters noise and highlights global structural correlations.

### 6. The Factor Loading Matrix ($W$)
- $W$ describes how much each latent factor contributes to each observed feature.
- Interpreting $W$: Which physical features (Unit 4) are "loaded" onto Latent Factor 1?
- Identifying the "drivers" of the latent manifold.

### 7. Probabilistic PCA (PPCA) (Murphy 12.1.1)
- A special case where the observation noise $\Psi$ is isotropic ($\sigma^2 I$).
- Bridge: PPCA connects the linear geometry of PCA to the probabilistic world of neural networks.

### 8. The Autoencoder as a Nonlinear Learner (Sandfeld 19.4)
- Why linear PCA fails: Materials manifolds are curved.
- Nonlinear activations (ReLU, Sigmoid) "unwrap" the manifold.
- "Unsupervised discovery of the nonlinear structure of the materials world."

### 9. Top-down view: Encoder, Decoder, and Bottleneck
- Encoder: The "measuring" function (from atoms to code).
- Decoder: The "generative" function (from code to atoms).
- The Bottleneck: The "distillation" of materials essence.

### 10. Loss Functions for Latent Learning
- Mean Squared Error (MSE): Faithful reconstruction of coordinates.
- Cross-Entropy: For categorical structure types or binary fingerprints.
- Regularization: Preventing the model from "memorizing" individual data points.

---

## Part B: Interpretation and Navigation (20–45 min)

### 11. Latent Scores: The Addresses of Materials
- The latent vector $z$ is the "score" of a specific material.
- Plotting $z_1$ vs. $z_2$: Seeing the "Periodic Table" of the dataset.
- Clusters in scores-space correspond to chemical prototypes.

### 12. The Problem of Unidentifiability (Murphy 12.1.3)
- Rotational invariance: Any orthogonal rotation of $W$ yields the same likelihood.
- Warning: "Axis 1" might not be a single physical property unless we enforce sparsity.
- Subspace vs. Axis interpretation.

### 13. Latent Traversals: Interrogating the Axis
- Move along $z$: $z_{new} = z_{base} + \alpha \vec{v}$.
- Decode the path: $\hat{x} = \mathcal{D}(z_{new})$.
- Visualization: Does the crystal lattice expand? Do bond angles change?
- This "decodes" the machine's learned features.

### 14. Neighborhood Consistency: Similarity Searching
- If $A$ and $B$ are close in $z$, are they chemically similar?
- "Latent-space similarity" as a discovery tool for findng analogues of rare materials.
- Preserving physical topology in the low-dimensional embedding.

### 15. Interpolation: The Path of Least Resistance
- Path between $A$ and $B$: $z(t) = (1-t)z_A + t z_B$.
- Why it's better than raw interpolation: The decoder ensures intermediate points stay on the "physics manifold."
- Identifying "missing" materials between known phases.

### 16. Disentanglement: Finding Independent Drivers
- Goal: Each latent dimension corresponds to exactly one physical factor (e.g., $z_1=$ volume, $z_2=$ symmetry).
- $\beta$-VAE: Using a penalty to force independent (uncorrelated) latent variables.
- Scientific value: Clearer "knobs" for materials design.

### 17. Case Study: 2D Car Projection (Murphy 12.2)
- Biplot analysis: Price vs. Fuel Efficiency axes.
- Analogy: In MG, we find axes for "Stiffness" vs. "Stability."
- Using biplots to guide the interpretation of crystal embeddings.

### 18. Case Study: Compressing Leaf Spectra (McClarren 8.2)
- Reducing 4,000 spectral channels to 4 latent factors.
- Traversal shows that $z_1$ moves the entire spectrum up/down (overall intensity).
- Bridge: Latent spaces for XRD or EELS phase identification.

### 19. Anomaly Detection via Reconstruction Loss (Neuer 5.5.3)
- "I can't draw this": High loss means the model hasn't seen this structure type.
- Filtering out "broken" simulations or unphysical configurations.

### 20. Anomaly Detection in Latent Scores (Neuer 5.5.3)
- "This point is alone": Outliers in the $z$-distribution.
- Identifying novelties that are structurally consistent but chemically rare.

---

## Part C: Advanced Architectures (45–65 min)

### 21. Variational Autoencoders (VAE) (Neuer 5.5.6)
- Learning a distribution $p(z|x)$ instead of a point.
- Bottleneck outputs $\mu$ and $\sigma$ (mean and variance).
- Reparameterization trick: $z = \mu + \epsilon \odot \sigma$.

### 22. VAE Loss: The KL-Divergence
- Loss = Reconstruction + KL-Divergence.
- KL-Divergence forces the latent space to follow a standard normal prior $\mathcal{N}(0, I)$.
- Prevents "holes" in the materials manifold.

### 23. Smoothing the Manifold for Discovery
- Regular autoencoders have "islands" of valid points.
- VAEs force the manifold to be continuous and dense.
- Discovery: Any point sampled from the VAE latent space is likely to be a "valid" material.

### 24. Posterior Collapse: The Decoder Problem
- When the decoder is too strong, it ignores the latent space.
- Identifying and fixing collapse in materials-design tasks.
- Balancing "expressive power" with "latent dependency."

### 25. Conditional VAEs (cVAE)
- Forcing the latent space to be conditioned on a property $y$ (e.g., Bandgap).
- $z$ then only represents the "leftover" structural variation.
- Strategic Goal: Generating materials with a specific $y$.

### 26. Denoising Autoencoders for Structural Cleanliness
- Training to reconstruct "perfect" lattices from "noisy" coordinates.
- Application: Cleaning up experimental TEM or XRD data using a learned physical prior.

### 27. Adversarial Latent Spaces (GANs bridge)
- Using a discriminator to force the latent space to be "source-blind" (Slide 21 in Unit 11).
- Removing non-physical artifacts from the materials design map.

### 28. Information Bottleneck Theory
- $I(z;y) \to \text{max}$, $I(z;x) \to \text{min}$.
- Discarding structural noise while keeping property signal.
- Finding the "minimal sufficient representation" for a materials target.

### 29. Latent Evolution: Mapping Trajectories
- Following a material's state in latent space during an MD simulation.
- Identifying "phase events" as jumps in the latent trajectory.

### 30. Visualizing Latent Uncertainty
- Mapping the "standard deviation" layer of a VAE.
- Where is the model most unsure about its representation?
- Targeting these regions for new simulation data collection.

---

## Part D: Domain Translation and Discovery (65–80 min)

### 31. Case: Latent Trajectories across composition series
- Mapping $Ba_{1-x}Sr_xTiO_3$.
- Does the path break when the symmetry changes?
- Using latent paths to detect subtle structural transitions.

### 32. Case: Embedding Drift in high-throughput data
- How latent coordinates shift between different DFT functional datasets.
- Correcting for "computational drift" in the materials manifold.

### 33. Case: Latent Space of Microstructures (Sandfeld 15.6)
- Convolutional AE for SEM/TEM images.
- Latent factors = Grain size, phase fraction, boundary density.
- Connecting the "pixel world" to the "design world."

### 34. The "Meaning" of Latent Neurons
- Analyzing activation maps in the latent layer.
- Which atoms "activate" Latent Dimension 3? (Attention-like analysis).
- Discovery: Realizing that $z_3$ is a "local octahedral tilting" detector.

### 35. Quantifying Manifold Smoothness
- Measuring the stability of property predictions under latent perturbations.
- A "smooth" manifold is a "trustworthy" discovery map.

### 36. Latent Class Separation
- Do different space groups separate naturally?
- Measuring the "structural awareness" of the learned embedding.

### 37. Probing the Bottleneck
- Linear probes (Unit 11) as a diagnostic for Unit 10 latent quality.
- "If I can't predict Bandgap from $z$, the latent space is missing electronic info."

### 38. Latent Density as a Discovery Proxy
- High density $\approx$ the "safe" zone.
- Low density $\approx$ the "frontier" zone.
- Designing exploration strategies for the frontier.

### 39. Handling Periodic Boundary Conditions in AE
- How to ensure the latent space respects the 3D periodicity of crystals.
- Using invariant descriptors (Unit 6) as the AE input.

### 40. Symmetry-Aware Latent Spaces
- Forcing $z$ to be invariant to rotations and translations.
- Preventing the latent space from becoming redundant.

### 41. Failure Mode: Dimensionality Mis-match
- What happens if $L$ is too small? (Slide 19 in Unit 9).
- What happens if $L$ is too large? (The latent space becomes a hairball).

### 42. Failure Mode: Feature Dominance
- One raw feature (e.g., Atomic Number) drowning out structural detail in the latent manifold.
- Normalization strategies for balanced discovery.

### 43. Failure Mode: Shortcut Learning from indexing
- Memorizing the input order instead of the chemistry.
- Using "random shuffle" validation to detect shortcuts.

### 44. The "Explainability" vs. "Performance" Tradeoff
- Does a 10D latent space outperform a 2D PCA?
- Deciding when the complexity of a nonlinear manifold is justified.

---

## Part E: Synthesis and Exercise (80–90 min)

### 45. Summary: Latent Space as a Design Tool
- Searchable manifolds.
- Interpretable axes.
- Actionable novelties.

### 46. Next Unit Bridge: Clustering (Unit 11)
- Unit 10: Continuous navigation.
- Unit 11: Discrete partitioning.
- Rule: Navigating the manifold (Unit 10) sets the stage for naming the families (Unit 11).

### 47. Exercise Task 1: Building a simple AE
- Dataset: MP formation energy.
- Deliverable: Plot reconstruction error vs. epoch.

### 48. Exercise Task 2: Latent Traversal
- Task: Pick a point $z$, move along $z_1$, and print the decoded composition/volume.
- Deliverable: Description of what $z_1$ "learned."

### 49. Exercise Task 3: Anomaly Scoring
- Task: Use the AE to score the "unusualness" of a nitride family (trained on oxides).
- Deliverable: Reconstruction loss histogram comparison.

### 50. Exam Checklist: Latent Manifolds
- Explain the Factor Loading Matrix.
- Why is latent interpolation better than raw interpolation?
- How does reconstruction error flag an anomaly?
- What is unidentifiability in latent variable models?
- What is the reparameterization trick in VAEs?
