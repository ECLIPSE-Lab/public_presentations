# MFML Unit 9 — 50-Slide Content Pack (English)

## Unit theme
**Representation Learning: From Handcrafted Features to Autoencoders**

## Core source mapping (book-priority aligned)
- **Neuer (2024)**: Ch. 5.5 (autoencoder topology, latent space, anomaly detection, nonlinear compression vs PCA).
- **McClarren (2021)**: Ch. 8 (fully connected and convolutional autoencoders, noise reduction, engineering case studies).
- **Murphy (2012)**: Ch. 12 (factor analysis, probabilistic PCA, latent linear models).
- **Bishop (2006)**: Ch. 12 (PCA as maximum-variance projection, probabilistic PCA).

---

## Slide-by-slide content (target: 50)

### Block A — Motivation and the manifold hypothesis (Slides 1-10)
1. **Title + Unit 9 positioning**
   - From probabilistic foundations (Unit 8) to learning useful data representations.
2. **Learning outcomes for Unit 9**
   - Manifold hypothesis, handcrafted vs learned features, autoencoder formalism, anomaly detection.
3. **Why features matter more than algorithms**
   - The quality of the representation determines the ceiling for any downstream model.
4. **The curse of dimensionality**
   - High-dimensional spaces are mostly empty; distances become meaningless; models need exponentially more data.
5. **The manifold hypothesis**
   - Real data lives on or near a low-dimensional manifold embedded in the high-dimensional ambient space.
6. **Visual: Swiss roll and manifold structure**
   - 3D data that lives on a 2D surface — dimensionality reduction unfolds it.
7. **PCA recap (Unit 2)**
   - Linear projection onto top-k eigenvectors of the covariance matrix; captures maximum variance.
8. **Limitations of PCA**
   - PCA can only find linear subspaces; fails on curved manifolds.
9. **Handcrafted features — historical approach**
   - Domain experts design features: Fourier coefficients, histogram bins, peak positions.
10. **Handcrafted vs learned features**
    - Handcrafted: interpretable, require domain expertise, brittle. Learned: flexible, data-driven, require training data.

### Block B — Autoencoder formalism (Slides 11-22)
11. **The autoencoder idea**
    - Train a network to reconstruct its own input through a bottleneck — force it to learn a compressed representation.
12. **Encoder-decoder architecture**
    - Encoder: f(x) = z (maps input to latent code). Decoder: g(z) = x_hat (maps latent code back to input).
13. **The bottleneck constraint**
    - Latent dimension d_z << d_x forces the network to discover the most informative features.
14. **Autoencoder loss function**
    - Reconstruction loss: L = (1/N) sum ||x_i - g(f(x_i))||^2. The network learns the identity through a bottleneck.
15. **Hourglass topology**
    - Visualization: wide input → narrowing encoder → narrow bottleneck → widening decoder → wide output.
16. **Linear autoencoder = PCA**
    - With linear activations and MSE loss, the optimal autoencoder spans the same subspace as top-k PCA.
17. **Why nonlinearity matters**
    - Nonlinear activations allow the autoencoder to capture curved manifolds that PCA cannot.
18. **Choosing the bottleneck dimension**
    - Too large: no compression, trivial identity. Too small: lossy, important structure lost.
19. **Reconstruction error vs latent dimension**
    - Plot error vs d_z: sharp elbow indicates the intrinsic dimensionality of the data.
20. **Training an autoencoder**
    - Standard backpropagation and gradient descent. Same training loop as any neural network.
21. **Regularization in autoencoders**
    - Weight decay, dropout, or noise injection prevent the autoencoder from memorizing training data.
22. **Checkpoint: autoencoder vs PCA**
    - "When does a nonlinear autoencoder outperform PCA?" — when the data manifold is curved.

### Block C — Convolutional and denoising autoencoders (Slides 23-32)
23. **Convolutional autoencoders — motivation**
    - For image/spectral data, fully connected layers ignore spatial structure; convolutions preserve it.
24. **Encoder: strided convolutions**
    - Convolution with stride > 1 simultaneously extracts features and reduces spatial resolution.
25. **Decoder: transposed convolutions**
    - Transposed (fractionally strided) convolutions upsample the spatial resolution back to the original.
26. **Convolutional autoencoder architecture**
    - Encoder: Conv → ReLU → Conv → ReLU → Flatten → Dense → latent.
    - Decoder: Dense → Reshape → TransConv → ReLU → TransConv → output.
27. **Pooling vs strided convolutions**
    - Strided convolutions are learnable downsampling — often preferred over fixed max/avg pooling.
28. **Denoising autoencoders**
    - Train with noisy inputs and clean targets: L = ||x_clean - g(f(x_noisy))||^2.
29. **Why denoising helps**
    - Forces the encoder to learn robust features that distinguish signal from noise.
30. **Denoising autoencoder as regularization**
    - Adding input noise prevents the autoencoder from learning the trivial identity mapping.
31. **Sparse autoencoders (brief)**
    - Add sparsity penalty on the latent code: encourage most latent units to be inactive.
32. **Checkpoint: conv AE design**
    - "Given 128×128 spectral images, design an encoder with three strided conv layers reducing to a 16-dim latent code."

### Block D — Applications: compression, anomaly detection, feature extraction (Slides 33-42)
33. **Application 1: data compression**
    - Store only the latent code z instead of the full input x; decode on demand.
34. **Compression ratio and reconstruction quality**
    - Compression ratio = d_x / d_z. Quality measured by reconstruction RMSE or SSIM.
35. **Application 2: anomaly detection via reconstruction error**
    - Train on "normal" data only. Anomalies produce high reconstruction error because they differ from learned patterns.
36. **Setting the anomaly threshold**
    - Compute reconstruction error distribution on validation normals; set threshold at 95th or 99th percentile.
37. **Application 2b: anomaly detection via latent space outliers**
    - Normal data clusters in latent space; anomalies map to distant regions.
38. **Two anomaly strategies compared**
    - Reconstruction error: catches any deviation. Latent outlier: catches structural novelty. Best to use both.
39. **Application 3: feature extraction for downstream tasks**
    - Train autoencoder unsupervised; use the encoder's latent code as features for a supervised classifier or regressor.
40. **Transfer learning with autoencoders**
    - Pretrain on a large unlabeled dataset; fine-tune the encoder with a small labeled dataset.
41. **Stacking autoencoders**
    - Train layer by layer: first AE learns raw → latent1; second AE learns latent1 → latent2; etc.
42. **Checkpoint: anomaly detection design**
    - "Your model reconstructs all training samples with RMSE < 0.05. A new sample has RMSE = 0.15. Is it anomalous?"

### Block E — Engineering case studies + summary (Slides 43-50)
43. **Case study: leaf spectra compression (McClarren)**
    - 2000-dim near-infrared spectra compressed to 10-dim latent code with < 2% reconstruction error.
44. **Case study: physics simulation data reduction**
    - Large-scale FEM outputs (temperature fields) compressed by convolutional AE for fast storage and retrieval.
45. **Case study: industrial anomaly detection**
    - Manufacturing sensor data: autoencoder trained on normal operation; reconstruction spikes flag equipment faults.
46. **Uncertainty in autoencoder outputs (Neuer)**
    - Reconstruction error as uncertainty proxy; high error regions indicate extrapolation or data poverty.
47. **Lecture-essential vs exercise content split**
    - Lecture: manifold hypothesis, AE formalism, PCA connection, architecture variants, application design.
    - Exercise: build AE in PyTorch, bottleneck sweep, anomaly detection experiment, denoising demo.
48. **Exercise setup: autoencoder for 1D signals**
    - Build FC autoencoder for Gaussian pulses; sweep bottleneck dimension; inject anomalies and detect.
49. **Exam-aligned summary: 10 must-know statements**
    1. The manifold hypothesis: real data lives on a low-dimensional manifold in high-dimensional space.
    2. An autoencoder learns compressed representations by reconstructing input through a bottleneck.
    3. A linear autoencoder with MSE loss is equivalent to PCA.
    4. Nonlinear autoencoders capture curved manifolds that PCA cannot.
    5. The bottleneck dimension controls the compression-quality tradeoff.
    6. Strided convolutions perform learnable downsampling; transposed convolutions perform upsampling.
    7. Denoising autoencoders learn robust features by reconstructing clean targets from noisy inputs.
    8. Anomaly detection: high reconstruction error or latent-space outlier status flags anomalies.
    9. Autoencoder latent codes serve as learned features for downstream supervised tasks.
    10. Reconstruction error provides an uncertainty proxy for autoencoder predictions.
50. **References + reading assignment for next unit**
    - Required: Neuer Ch. 5.5, McClarren Ch. 8.
    - Optional: Murphy Ch. 12, Bishop Ch. 12.
    - Next unit: Latent Spaces and Embeddings — VAEs, structured latent spaces, and generation.

---

## Reusable equations (for slide math boxes)
- AE loss: \\(L = \frac{1}{N}\sum_{i=1}^N \|\mathbf{x}_i - g(f(\mathbf{x}_i))\|^2\\)
- Encoder: \\(\mathbf{z} = f_\phi(\mathbf{x})\\)
- Decoder: \\(\hat{\mathbf{x}} = g_\psi(\mathbf{z})\\)
- Compression ratio: \\(r = d_x / d_z\\)
- Denoising AE: \\(L = \frac{1}{N}\sum_{i=1}^N \|\mathbf{x}_i - g(f(\tilde{\mathbf{x}}_i))\|^2\\), where \\(\tilde{\mathbf{x}}_i = \mathbf{x}_i + \epsilon\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: manifold hypothesis, AE formalism, PCA connection, convolutional/denoising variants, application design principles.
- **Exercise**: PyTorch AE implementation, bottleneck dimension sweep, PCA comparison, anomaly detection experiment, denoising demo.

## Reading assignment after Unit 9
- Neuer Ch. 5.5
- McClarren Ch. 8
- Murphy Ch. 12 (skim)
- Bishop Ch. 12 (conceptual)
