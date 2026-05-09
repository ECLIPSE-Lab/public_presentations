# Materials Genomics Unit 9 — Representation Learning and Feature Discovery

## Core teaching claim
In Materials Genomics, representation learning moves beyond hand-crafted descriptors toward learned embeddings that can discover hidden structure in materials data. While autoencoders provide the mathematical framework for this discovery, their scientific value is determined by whether the resulting latent space is physically interpretable, transferable across tasks, and robust against dataset bias.

## 1. Beyond fixed descriptors: why learn representations?

Until now, we have assumed that the representation of a material—be it a composition vector or a structural fingerprint—is fixed before the model is trained. Unit 9 marks a shift: the representation itself is now part of the learning problem. We no longer just ask "how well can we predict $y$ from $x$?" but "what is the best $z$ to represent $x$ for a range of discovery tasks?" This transition is motivated by the fact that hand-crafted features often saturate in performance or fail to capture the complex, nonlinear interactions inherent in chemical and structural spaces.

## 2. Representation learning as an unsupervised task

Most representation learning in Materials Genomics begins as an unsupervised task. We use the data itself to define the representation without requiring expensive labels. As Neuer (5.5) and McClarren (8.1) explain, this is often operationalized through the identity mapping $f(x) = x$. By forcing the model to reconstruct the input through a bottleneck, we compel it to discover the most salient features of the materials data. This is the essence of feature discovery: the model must "decipher the writings" (West/McClarren) of the materials manifold.

## 3. The Autoencoder: Encoder, Decoder, and Bottleneck

The primary architecture for this task is the autoencoder. It consists of two half-networks: the encoder $\mathcal{E}$, which compresses the input $x$ into a lower-dimensional code $z$, and the decoder $\mathcal{D}$, which reconstructs the input from that code. The "bottleneck" (the middle layer) is the critical design element. Its dimension determines the compression ratio and forces the network to prioritize global structural patterns over local noise. In materials terms, the encoder acts as a "nonlinear featurizer" that replaces manual descriptor engineering.

## 4. Latent Space: The "Genomic" Coordinate System

The values in the bottleneck layer form the latent space $\mathbb{L}$. This space acts as a compressed, continuous coordinate system for materials. If the autoencoder is well-trained, similar materials should lie close together in this space. As Bishop (12.4.2) and Sandfeld (19.4) note, this latent space is a nonlinear extension of the linear eigenspace found in PCA. In Materials Genomics, we treat this space as a searchable manifold where we can interpolate between known structures to discover new candidates.

## 5. Non-linear dimensionality reduction vs. PCA

A key advantage of autoencoders over PCA is their ability to capture nonlinear relationships. PCA is limited to linear combinations of input features (Neuer 5.2). Materials data, however, often lives on complex, curved manifolds—for example, the transition from one crystal symmetry to another is rarely a linear path in coordinate space. Autoencoders can "unwrap" these manifolds into a flat latent space, making downstream tasks like regression or clustering much more efficient.

## 6. Case Study: Compressing Spectral Data (McClarren 8.2)

Spectral data (XRD, Raman, or the leaf spectra in McClarren) is high-dimensional and highly redundant. An autoencoder can reduce a 2000-channel spectrum to just 2 or 4 latent variables while retaining the essential physical signal. McClarren demonstrates that these latent variables often correspond to physical drivers—like the overall intensity or the presence of specific peaks—but in a way that respects the nonlinear nature of the signal. In MG, we use this to handle the massive output of high-throughput characterization.

## 7. Feature Discovery: Interpreting the Latent Dimensions

The "discovery" in feature discovery happens when we analyze what the latent variables $z$ actually represent. By varying one dimension of $z$ while keeping others fixed and observing the decoded output, we can visualize the "features" the model has learned. In materials science, a latent dimension might align with atomic radius, electronegativity, or a specific structural motif like octahedral tilting, even if we never explicitly told the model about these concepts.

## 8. Embedding quality: separability and structure

A "good" embedding is not just one that reconstructs well. We also look for separability—do different chemical families or structural prototypes form distinct clusters? As Sandfeld (2.2) emphasizes, if the latent space is just a "hairball" of points with no clear structure, it may have failed to discover the underlying physics of the dataset. We use visualization tools like t-SNE or UMAP to probe these spaces, but with the caution that visual cleanliness is not a proof of physical discovery.

## 9. Transferability: Embeddings as Pre-trained Featurizers

One of the most powerful applications of representation learning is transfer learning. We can train an autoencoder on a large, unlabeled database (like the Materials Project) to learn a general "materials embedding." This pre-trained encoder can then be used as a fixed featurizer for a small-data task, such as predicting the thermal conductivity of a rare class of perovskites. Because the encoder has already learned the "language" of crystal structures, it often outperforms hand-crafted descriptors on these downstream tasks.

## 10. Probing Embeddings: The Readout Test

To quantify how much "information" is in a learned representation, we use a "readout" or "probe" test. We fix the encoder and train a simple linear regressor on the latent variables $z$ to predict a property $y$. If the linear probe performs well, it means the representation has successfully "linearized" the complex structure-property relationship. This is a standard diagnostic for representation quality in both MFML and MG.

## 11. Self-supervised learning: The broader context

While autoencoders use reconstruction, other self-supervised objectives can also drive representation learning. For example, "masking" (predicting missing atoms in a structure) or "contrastive learning" (recognizing that a rotated crystal is the same as the original) are increasingly common. The goal remains the same: use the inherent structure of the materials world to learn a representation that is more powerful than what we could engineer by hand.

## 12. Failure Mode: Shortcut Learning and Data Leakage

A common pitfall in representation learning is "shortcut learning." An autoencoder might learn to reconstruct a crystal structure by simply memorizing its position in a spreadsheet or a specific artifact of the simulation software, rather than learning the underlying chemistry. This leads to embeddings that look perfect on the training set but fail completely on new data. Rigorous split design (grouped by chemistry or prototype) is as essential here as it was in Unit 7.

## 13. Failure Mode: Over-compression and Information Loss

If the bottleneck is too narrow, the autoencoder will lose critical details—a "lossy" compression that might smooth over the very structural nuances that determine a property. For example, a too-small latent space might map a stable and an unstable polymorph to the same point. Choosing the latent dimension is a tradeoff between capturing the manifold's structure and preserving the target's sensitivity.

## 14. Visualization Pitfalls: t-SNE is not Discovery

t-SNE and UMAP are frequently used to visualize latent spaces. However, as Neuer (5.5) and Sandfeld (2.2) warn, these algorithms can create clusters where none exist or hide distances that are physically meaningful. A "pretty" t-SNE plot is a diagnostic starting point, not a scientific conclusion. Quantitative metrics like reconstruction error and probe accuracy must always accompany the pretty pictures.

## 15. The Latent Space of Microstructures (Sandfeld 15.6)

Representation learning also applies to microstructural images. By training an autoencoder on TEM or SEM micrographs, we can discover latent factors that correspond to grain size, orientation, or phase fraction. This bridges the gap between the "pixel world" of images and the "parameter world" of constitutive models, allowing us to treat microstructure as a computable coordinate in a design space.

## 16. Regularization and Latent Geometry

The geometry of the latent space can be influenced by regularization. For example, Variational Autoencoders (VAEs) force the latent space to follow a specific distribution (usually Gaussian). This makes the space "smoother" and better for interpolation, preventing the "holes" that can occur in plain autoencoders. This is a direct application of the probabilistic view of learning from MFML Unit 8.

## 17. Case: Embedding Drift across Domains

Embeddings learned on one database (e.g., OQMD) may drift when applied to another (e.g., Materials Project). This is because the representation captures not only the chemistry but also the specific biases of the source simulation pipeline. Understanding and mitigating this drift is essential for building universal materials discovery platforms.

## 18. Hybrid Pipelines: Descriptors + Embeddings

In practice, the best results often come from hybrid pipelines. We might combine a few critical hand-crafted descriptors (which we know are physically important) with a set of learned embeddings (which capture the remaining complexity). This "grey-box" approach (Sandfeld 2.2) balances the interpretability of classical methods with the power of modern representation learning.

## 19. Uncertainty in Embedding-based Predictions

Predicting from an embedding adds a layer of uncertainty. Is the model uncertain because the regressor is weak, or because the representation itself is ambiguous for that specific chemistry? Decomposing these uncertainties is a key theme for Unit 12, but it starts here with the realization that every representation choice is a source of epistemic risk.

## 20. Summary and Next Steps

Unit 9 has shown that representations don't have to be static. By learning the "latent language" of materials, we can discover features that human experts might miss. However, these learned features must be held to the same standards of validity and scientific trust as any other model. In Unit 10, we will take these latent spaces and ask: "how can we use them to actually generate new materials?"

## 21. Exercise: Comparing Engineered vs. Learned Features

The exercise for this unit involves taking a materials property task and comparing two pipelines: one using standard `matminer` descriptors and one using a learned embedding from a simple autoencoder. Students will evaluate not only the RMSE but also the transferability of the features to a slightly different chemical subset, diagnosing where the learned representation adds value and where it fails.

## 22. Exam Checklist: Representation Learning
- Can you explain the bottleneck effect in an autoencoder?
- What is the difference between an encoder and a decoder?
- Why is the latent space often called a "nonlinear manifold"?
- What are the risks of using t-SNE for scientific discovery?
- How does transfer learning benefit from pre-trained embeddings?

## 23. References for Unit 9
- Neuer: Section 5.5
- McClarren: Chapter 8
- Bishop: Section 12.4.2
- Sandfeld: Section 2.2 and 19.4
- Murphy: Chapter 35 (Deep Learning intro)
