# DSEM Exam "Must-Know" Reference

> **Purpose:** This document defines the examinable surface for the written exam (60% of final grade).  
> Each section lists ≤10 statements that a student must be able to reproduce, explain, or apply  
> under exam conditions. Items marked _(≤10 must-know statements…)_ are placeholders filled  
> when the corresponding week's lecture deck is authored; do **not** study a placeholder section —  
> it means content has not yet been finalised for that week.
>
> The exam tests **conceptual understanding and application**, not code syntax. Derivations are  
> only examinable where explicitly flagged in the deck.

---

## Week 1 — Python crash course + launch

1. **Data-rate growth:** modern 4D pixelated detectors produce up to ~200 TB h⁻¹ — approximately 10⁸× more than film-era instruments — making automated analysis unavoidable.
2. **PSPP chain:** Processing → Structure → Property → Performance; each arrow is a distinct ML task; structure *mediates* properties and cannot be skipped in the causal chain.
3. **CRISP-DM phases (in order):** Business/scientific understanding → Data understanding → Data preparation → Modelling → Evaluation → Deployment/monitoring; evaluation measures *generalisation*, not training accuracy.
4. **NumPy array fundamentals:** every EM dataset is an ndarray with a `shape` (e.g. `(512, 512)` for HAADF, `(256, 256, 128, 128)` for 4D-STEM) and a `dtype` (e.g. `float32`); these two attributes are always the first things to check.
5. **Vectorised operations vs. loops:** NumPy arithmetic applies element-wise to whole arrays without Python for-loops, giving ≥100× speed improvement; write `corrected = image - dark` not a pixel-by-pixel loop.
6. **Broadcasting rule:** two dimensions are compatible if they are equal or one of them is 1; shapes are aligned from the *right*; e.g. `(512, 512) − (512, 1)` is valid and subtracts each row mean from every column.
7. **Indexing is 0-based and slice stop is exclusive:** `arr[1:3]` returns elements at indices 1 and 2; `arr[-1]` is the last element; applies to all axes of an ndarray.
8. **Tensor = n-dimensional array:** a scalar is 0-D, a spectrum is 1-D, an image is 2-D, an EELS map is 3-D (x, y, energy), a 4D-STEM scan is 4-D (x, y, kx, ky); NumPy ndarray and PyTorch Tensor are both tensors.
9. **Min-max normalisation:** `(x − x.min()) / (x.max() − x.min())` scales any array to [0, 1]; standard pre-processing step before displaying or comparing EM images; implemented as a one-liner or a documented `def normalize(image)` function.
10. **Assessment structure:** 40% miniproject (reproducible notebook + report, CRISP-DM pipeline, uncertainty + explainability required) + 60% written exam; miniproject proposal due ≈ Week 6; a notebook that does not execute end-to-end scores zero on the reproducibility criterion (15% of miniproject grade).

---

## Week 2 — What is learning? EM data & noise origins

1. **Learning vs classical analysis:** in classical data analysis the human writes the rules; in machine learning the optimiser infers the rules from labelled examples. Both require domain knowledge — ML uses it to define inputs, outputs, and the training set.
2. **Model taxonomy (white / grey / black-box):** white-box = explicit mechanism with interpretable parameters (e.g. Bragg's law); black-box = high flexibility, no traceable internals (e.g. deep CNN); grey-box = physics structure with one learned sub-module (e.g. CPFEM + learned hardening law). Choice depends on how much physics is reliable at the relevant scale, not on personal preference.
3. **EM data-tensor shapes:** HAADF image `(ny, nx)`; 4D-STEM cube `(ny, nx, ky, kx)`; EELS/EDS spectrum image `(ny, nx, E)`; tilt-series `(n_tilt, ny, nx)`. The tensor shape determines the correct ML architecture family.
4. **Nyquist–Shannon theorem:** a signal with maximum frequency $f_\text{max}$ requires sampling rate $f_s \geq 2 f_\text{max}$; in imaging, pixel pitch $\Delta x \leq d/2$ to resolve features of size $d$. Practical target: 3–5× oversampling; bare Nyquist is a lower bound, not a target.
5. **Aliasing / Moiré:** when $f_s < 2 f_\text{max}$, high-frequency components fold back into lower frequencies; in TEM this produces spurious long-period Moiré fringes that are *not* real structure. Diagnosis: change camera length or tilt — real structure moves predictably, aliasing artefacts do not.
6. **Poisson noise (shot noise):** electron count $N \sim \text{Poisson}(\lambda)$ where $\lambda$ is the expected count; variance equals the mean: $\text{Var}(N) = \lambda$; therefore $\text{SNR} = \sqrt{\lambda}$. This is a fundamental quantum limit — it cannot be reduced by better electronics, only by increasing dose or improving detectors.
7. **SNR–dose scaling:** $\text{SNR} \propto \sqrt{\text{dose}}$; doubling SNR requires quadrupling dose. At low dose (beam-sensitive samples), damage caps the dose budget, so SNR is physically bounded. Denoising algorithms can recover structure but cannot invent information not collected.
8. **Gaussian noise (readout/thermal):** electronic amplifier and thermal fluctuations produce additive Gaussian noise $\mathcal{N}(0, \sigma_r^2)$ with variance *independent* of the signal. Real detectors have a Gaussian + Poisson mixture: $x_i = g \cdot \text{Poisson}(\lambda_i) + \mathcal{N}(0, \sigma_r^2)$; a variance–mean plot diagnoses the mixture.
9. **Aleatory vs epistemic uncertainty:** aleatory uncertainty is irreducible (shot noise, thermal vibrations — quantum physics); epistemic uncertainty reflects missing knowledge (calibration error, small dataset, wrong model class) and *can* be reduced by more data, better calibration, or improved models. Active acquisition strategies target epistemic uncertainty.
10. **Noise model → loss function:** Gaussian noise → MSE loss is the correct choice; Poisson noise → Poisson negative-log-likelihood loss. Using MSE on low-count EM data is a silent physics error that causes models to underfit noisy regions because it assumes signal-independent variance.

---

## Week 3 — Linear algebra & PCA

1. **Dot product and projection:** $\mathbf{a}^T\mathbf{b} = |\mathbf{a}||\mathbf{b}|\cos\theta$; the scalar projection of $\mathbf{b}$ onto unit vector $\hat{\mathbf{a}}$ is $c = \hat{\mathbf{a}}^T\mathbf{b}$; the residual $\mathbf{b} - c\hat{\mathbf{a}}$ is perpendicular to $\hat{\mathbf{a}}$. Dot product = 0 ⟹ orthogonal (the two vectors carry independent information).
2. **Data matrix convention:** for a spectral image with $N$ pixels and $D$ energy channels, form $\mathbf{X} \in \mathbb{R}^{N \times D}$ with observations in rows and features (channels) in columns. For an EELS map of shape (64, 64, 1024), reshape to (4096, 1024) before PCA.
3. **Least-squares = orthogonal projection:** the best-fit weights $\hat{\mathbf{w}}$ in $\mathbf{Xw} \approx \mathbf{y}$ are found by projecting $\mathbf{y}$ onto the column space of $\mathbf{X}$; the residual is perpendicular to all columns of $\mathbf{X}$. This geometric view gives the Normal equations $\mathbf{X}^T\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^T\mathbf{y}$ without any calculus.
4. **SVD as rotate–stretch–rotate:** any matrix $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$; $\mathbf{V}$ rotates input space (principal directions in feature space); $\boldsymbol{\Sigma}$ stretches along those directions by the singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$; $\mathbf{U}$ rotates output space. Singular values are the semi-axes of the output ellipse.
5. **PCA = SVD of centered data:** center $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{x}}\mathbf{1}^T$, then SVD. Right singular vectors $\mathbf{V}$ are the principal directions (eigenspectra); left singular vectors $\mathbf{U}$ scaled by $\boldsymbol{\Sigma}$ are the scores (chemical maps). PCA finds directions of maximum variance in decreasing order.
6. **Eckart–Young theorem:** the truncated rank-$k$ SVD $\hat{\mathbf{X}}_k = \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^T$ is the optimal rank-$k$ approximation of $\mathbf{X}$ in the Frobenius norm; reconstruction error $= \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \ldots}$. This is why PCA denoising is mathematically optimal (for Gaussian noise).
7. **Scree plot and the elbow rule:** plot variance explained per component ($\sigma_k^2 / \sum_k \sigma_k^2$) in decreasing order. Signal components form a steeply declining head; noise components form a flat floor. Keep components before the "elbow" where the curve transitions from steep to flat. Also use the 95% cumulative-variance criterion as a cross-check.
8. **PCA denoising mechanism:** signal lives in a low-dimensional subspace (rank $K$ for $K$ chemical phases); Poisson/Gaussian noise spreads across all $D$ directions. Truncating the SVD at rank $K$ projects onto the signal subspace and discards the noise. Choosing K too small loses real signal (underfitting); too large adds noise back (overfitting). The optimal K is at the scree plot elbow.
9. **Eigenspectra and score maps:** the $k$-th eigenspectrum (row of $\mathbf{V}^T$) is the spectral shape of the $k$-th variation mode. The score map (column of $\mathbf{X}_\text{centered}\mathbf{V}_k$, reshaped to image) shows where that spectral shape is strong across the sample. Eigenspectra can be negative; a PC looks like "noise" if it shows no recognisable spectral features.
10. **Ill-conditioning and correlated features:** condition number $\kappa = \sigma_\text{max}/\sigma_\text{min}$; $\kappa \gg 1$ means a small perturbation in the data produces a large swing in the fitted parameters. Correlated features (near-parallel columns in $\mathbf{X}$) cause near-singular $\mathbf{X}^T\mathbf{X}$ and high $\kappa$. Cures: standardise features (subtract mean, divide by std); use PCA pre-processing to remove correlated directions; or apply Ridge regularisation (Week 4).

---

## Week 4 — Regression, gradient descent & honest validation

1. **ERM principle:** training a model = minimising the empirical risk $\hat{R}(\mathbf{w}) = \frac{1}{N}\sum_i L(f_\mathbf{w}(\mathbf{x}_i), y_i)$; every supervised algorithm is a choice of loss function + a choice of optimiser.
2. **Loss–noise-model duality:** MSE is the correct loss when residuals are iid Gaussian; MAE is correct under Laplacian residuals; Huber combines quadratic core with linear tails. Choosing the wrong loss on low-count EM data (Poisson noise) silently biases the fit.
3. **Gradient descent update:** $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\,\nabla_\mathbf{w}\hat{R}(\mathbf{w}_t)$; $\eta$ is the learning rate; too small → slow convergence, too large → oscillation or divergence.
4. **SGD and minibatch SGD:** replacing the full-data gradient with a gradient estimated on one sample (SGD) or $b$ samples (minibatch) reduces per-step cost from $\mathcal{O}(N)$ to $\mathcal{O}(1)$ or $\mathcal{O}(b)$ while remaining unbiased in expectation. Batch sizes 32–256 are standard.
5. **Adam:** combines SGD, momentum (exponential moving average of gradients, $\beta_1 = 0.9$), and per-parameter adaptive scaling (exponential moving average of squared gradients, $\beta_2 = 0.999$). Default optimiser for neural networks; typical $\eta = 0.001$.
6. **Bias–variance decomposition:** test error = $\text{Bias}^2 + \text{Variance} + \sigma^2$. Underfitting = high bias (model too simple); overfitting = high variance (model memorised training noise). Noise floor $\sigma^2$ is irreducible. Regularisation (Ridge/Lasso) trades variance for bias.
7. **Train/validation/test protocol:** train = fit parameters; validation = tune hyperparameters (may look repeatedly); test = final one-time honest report. Any modelling decision based on test-set results converts it to a validation set.
8. **K-fold cross-validation:** split data into $k$ folds; each fold serves as test exactly once; report $\overline{\text{MSE}} \pm \text{std}$. More reliable than a single holdout; $k=5$ or $k=10$ are standard defaults.
9. **The EM leakage trap (crop-vs-specimen):** if crops from the same EM specimen appear in both train and test, the model memorises specimen identity rather than the physical property. Fix: `gkf = GroupKFold(n_splits=5)` then `gkf.split(X, y, groups=specimen_id)` (or `cross_val_score(..., cv=gkf, groups=specimen_id)`) — entire specimens are in either train or test, never both. This is the default CV strategy whenever a `specimen_id` column exists.
10. **Segmentation metrics:** IoU = $|A\cap B|/|A\cup B|$; Dice = $2|A\cap B|/(|A|+|B|) = 2\,\text{IoU}/(1+\text{IoU})$. Both range $[0,1]$; $R^2 < 0$ is possible and means the model is worse than predicting the mean. For imbalanced classification, report precision and recall, not accuracy.

---

## Week 5 — Neural networks from first principles

1. **Perceptron computation:** a single neuron computes $\hat{y} = \sigma(\mathbf{w}^T\mathbf{x} + b)$; $\mathbf{w}$ are weights, $b$ is the bias, and $\sigma$ is a nonlinear activation function. Without the bias, the decision boundary must pass through the origin.
2. **Linear separability limit:** a single perceptron draws exactly one hyperplane in input space. It cannot solve XOR or any other problem where the class boundary is non-convex or multi-connected — regardless of training time or learning rate.
3. **Non-linearity is non-negotiable:** stacking $L$ purely linear (identity-activation) layers is algebraically equivalent to a single affine map $(W^{(L)}\cdots W^{(1)})\mathbf{x} + \tilde{\mathbf{b}}$. Only nonlinear activations make depth meaningful.
4. **MLPs as feature extractors:** hidden layers learn an intermediate representation that transforms the input space so that the output layer can apply a linear map. The final layer is always a linear combination of learned features — the hidden layers are the nonlinear feature engineers.
5. **Activation function rules:** hidden layers → ReLU (or Leaky ReLU) by default. Output layer → identity (regression), sigmoid (binary probability), softmax (multi-class probability). Sigmoid and tanh are no longer recommended as hidden activations because they saturate.
6. **Vanishing gradient:** the chain rule multiplies the activation derivative at every layer during backprop. Sigmoid derivative $\sigma'(z) \leq 0.25$; through 8 layers this gives $\leq 0.25^8 \approx 2 \times 10^{-4}$ — early-layer weights barely update. ReLU derivative is 1 for $z > 0$, curing the problem.
7. **Backprop = reverse-mode automatic differentiation:** the computational graph stores all intermediate values during the forward pass; the backward pass traverses the graph in reverse, applying the chain rule at each node. Modern libraries (`loss.backward()` in PyTorch) do this automatically — you never implement it from scratch.
8. **Training protocol:** loss → compute gradient via backprop → update all weights with $\theta \leftarrow \theta - \eta\,\nabla_\theta\hat{R}$. Monitor validation loss, not training loss. Stop (or save the model) when validation loss stops improving — early stopping. Adam at $\eta = 0.001$ is the default optimiser.
9. **Initialisation:** do not set all weights to zero — symmetry is never broken and all neurons learn identically. Use He initialisation for ReLU layers ($w \sim \mathcal{N}(0, \sqrt{2/D_\text{in}})$); Xavier/Glorot for sigmoid/tanh. Framework defaults are correct; only override deliberately.
10. **When NOT to use a deep net:** $N < 200$ tabular samples (simpler models generalise better); known analytic feature map (embed the physics, do not learn it); images / spatial data without convolutional structure (dense MLP has $10^9$ weights for a 1024×1024 image — use a CNN instead, Week 6).

---

## Week 6 — CNNs for microscopy images

1. **Parameter explosion of MLP on images:** a 1024×1024 image with one dense layer of 1000 neurons has $\sim10^9$ weights before any depth; a single 3×3 conv layer with 64 filters has 576 weights — a reduction of $>10^6\times$ — because the same 9 weights are reused at every spatial position (weight sharing).
2. **Convolution = local sliding detector:** output $(I\star K)_{m,n} = \sum_{a,b} K_{a,b}\,I_{m+a,n+b}$; a 3×3 kernel uses only the 9-pixel neighborhood of each output location; this encodes locality and weight sharing simultaneously.
3. **Weight sharing encodes translation equivariance:** the same kernel is applied at every image location, so if the input feature shifts by $\delta$ pixels, the feature map shifts by $\delta$ pixels — $f(T_\delta X)=T_\delta f(X)$. Translation *invariance* (output unchanged by shift) is built gradually via pooling and global average pooling.
4. **Inductive bias:** an architectural constraint that makes physically reasonable functions easier to learn; CNN's local+shared bias is well-matched to EM images (local features, repeat patterns, hierarchical structure) and enables learning from small labelled datasets.
5. **Feature hierarchy:** Layer 1 learns edge/gradient detectors; Layer 2 learns local texture/motif detectors (e.g. grain-boundary segments); deeper layers learn phase regions, defect clusters, or global material state. This mirrors the physical hierarchy in EM images: atoms → grains → phases → microstructure.
6. **Receptive field growth:** stacking $L$ conv layers of kernel size $k$ gives effective receptive field $(k-1)L+1$; two 3×3 layers see a 5×5 field with an extra nonlinearity — more expressive and fewer parameters than one 5×5 layer.
7. **Pooling:** max-pooling keeps the strongest activation in each $2\times2$ window (no learned parameters); halves spatial resolution; provides approximate local translation invariance; the standard Conv → ReLU → MaxPool building block doubles effective context per stage.
8. **U-Net for segmentation:** encoder (successive Conv+Pool, channels double) → bottleneck → decoder (successive upsample+Conv, channels halve); skip connections concatenate encoder feature maps into the decoder at each resolution level, preserving fine spatial detail (boundary locations) that the bottleneck would otherwise lose. Output: one label per pixel at full input resolution.
9. **ResNet skip connections:** residual block output $\mathbf{y}=F(\mathbf{x})+\mathbf{x}$; learning the *residual* $F(\mathbf{x})=\mathbf{y}-\mathbf{x}$ and bypassing the gradient through the skip path solves the vanishing gradient problem, enabling networks with 50–150+ layers.
10. **CNN failure modes in EM:** domain shift (different microscope, contrast transfer, resolution) causes silent performance degradation; validate using `GroupKFold` by specimen/session; report IoU/Dice not accuracy; always inspect predicted masks qualitatively; match training and inference pixel size.

---

## Week 7 — Beating small & expensive data

1. **The labelled-data gap:** materials EM labs typically have 50–500 labelled images; ResNet-50 has 25 million parameters. At 500 images that is 50 000 parameters per image — guaranteed overfitting without augmentation, transfer learning, or synthetic data.
2. **Augmentation encodes physical invariances:** an augmentation is a claim that the physics has a symmetry. Rotating an equiaxed-grain image is valid (orientation is arbitrary); rotating an EBSD map is illegal (the colour IS the crystallographic orientation). Invalid augmentations inject label noise.
3. **Label consistency rule (geometric augmentations):** every geometric transform applied to the image must be applied identically to the mask/label at the same time. Apply the split first, then augment — never augment before splitting by specimen.
4. **Transferability decreases with depth (Yosinski 2014):** Layer 1 (edges, gradients) is ~95% transferable from ImageNet to EM; Layer 4+ (full objects) is ~10%. Fine-tuning strategy should be depth-graded: keep early layers nearly fixed, adapt later layers more, fully retrain the head.
5. **Feature extraction vs fine-tuning:** feature extraction (freeze backbone, train head only) is safe for <100 labels; fine-tuning (update backbone with low lr) gains accuracy with 100–1000 labels but risks catastrophic forgetting. Use differential learning rates: backbone lr ≈ $10^{-5}$, head lr ≈ $10^{-3}$ (ratio 100×).
6. **Catastrophic forgetting:** a randomly-initialised head produces large near-random gradients in epoch 1. Backpropagated at full lr through a pretrained backbone, these overwrite ImageNet features before the head stabilises. Symptom: training loss spikes at the start; final accuracy worse than feature extraction. Fix: gradual unfreezing + differential LRs.
7. **Gradual unfreezing protocol:** (1) train head only until plateau; (2) unfreeze last backbone block with low lr; (3) unfreeze progressively deeper blocks from top to bottom (most domain-specific first, most general last). Combines with differential LRs for defence-in-depth.
8. **Voronoi synthetic microstructures:** Voronoi tessellations assign each pixel to its nearest random seed point — giving perfect free grain-ID labels by construction. The rendering pipeline (random grain intensity, dark boundary, Poisson noise, blur) makes images look like SEM. Works for grain topology tasks because Voronoi captures the topological truth of grain networks; fails for twin detection or non-equiaxed morphologies.
9. **Sim-to-real failure rule:** synthetic data fails exactly on the feature the generator omits. A model trained on Voronoi images cannot detect annealing twins because Voronoi never generates twins. The fix is not more synthetic data — it is closing the generator's physics gap or fine-tuning on real labelled images.
10. **Active learning:** label the most uncertain samples first (uncertainty or entropy scoring). Requires a cold-start random seed batch (model's uncertainty is meaningless with zero labels). Combine uncertainty with diversity to avoid labelling near-duplicate hard cases. 50 strategically chosen labels can outperform 500 random ones.

---

## Week 8 — Unsupervised learning & autoencoders

1. **Unsupervised learning requires no labels:** the data itself is the only signal. The four families are clustering (discrete), dimensionality reduction (continuous), density estimation, and generative modelling. Autoencoders do dimensionality reduction; k-means/GMM do clustering; they are used in sequence (AE → cluster latent codes).
2. **Lloyd's algorithm (k-means):** alternately (i) assign each point to its nearest centroid and (ii) update each centroid to the mean of its assigned points. Guaranteed to converge to a *local* minimum of the within-cluster sum of squared distances $J$. Use K-means++ initialisation and multiple random restarts.
3. **K-means failure mode in EM:** raw high-dimensional spectra should not be clustered directly — Euclidean distance is dominated by the highest-intensity feature (zero-loss peak, characteristic X-ray line). Fix: cluster AE latent codes or PCA scores, not raw spectra.
4. **GMM extends k-means with soft assignments:** each point $x_i$ carries a responsibility $\gamma_{ik} = P(\text{phase}\,k\mid x_i)$ instead of a binary label. GMM allows ellipsoidal clusters (full covariance $\Sigma_k$) and is fitted by alternating E- (compute responsibilities) and M-step (update $\mu_k, \Sigma_k, \pi_k$).
5. **Linear autoencoder = PCA:** with linear activations and MSE loss, the optimal encoder/decoder satisfy $W_d W_e = U_k U_k^T$ — the PCA subspace. Adding nonlinear activations (ReLU) breaks this equivalence and allows the AE to follow *curved* data manifolds that PCA (flat subspace) cannot.
6. **Reconstruction objective is self-supervised:** $\mathcal{L} = \frac{1}{N}\sum_i \|x_i - g_\theta(f_\phi(x_i))\|^2$. No external labels needed — inputs are both inputs and targets. The bottleneck forces the latent code $z$ to retain only information sufficient for reconstruction.
7. **Denoising AE for low-dose EM:** train with corrupted input $\tilde x_i$ and clean target $x_i$. The latent code must capture *robust* features that survive noise; noise-specific features are forced out of the bottleneck. For Poisson-dominated EELS, use Poisson NLL loss or normalise spectra to unit variance before MSE.
8. **Manifold hypothesis for EM spectra:** a 1024-channel EELS spectrum lies near a manifold of intrinsic dimension ≈ number of distinct phases (3–5 for most EM experiments). PCA finds the best flat hyperplane; the AE follows the curved manifold. The AE reconstruction error at $k$ latent dimensions should be compared honestly against PCA at the same $k$.
9. **t-SNE and UMAP are visualisation tools only:** t-SNE minimises KL(P‖Q) between high-dim and low-dim neighbourhood distributions; UMAP uses fuzzy graph matching. Both preserve local structure; neither preserves inter-cluster distances or densities. Never read t-SNE cluster sizes or gaps quantitatively. Use UMAP as the 2026 default; t-SNE as a diagnostic.
10. **Anomaly detection by reconstruction error:** train an AE on normal data only; compute per-sample reconstruction error at test time. Anomalies (beam damage, contamination, novel phases) lie outside the learned manifold and reconstruct poorly. Threshold at the 99th percentile of training-set reconstruction error. A secondary score — Mahalanobis distance in the latent space — catches subtle anomalies that the AE reconstructs plausibly but encodes far from the normal cluster.

---

## Week 9 — Probability, uncertainty & Gaussian processes

- _(≤10 must-know statements — filled when this week's deck is authored)_

---

## Week 10 — Active & automated electron microscopy

- _(≤10 must-know statements — filled when this week's deck is authored)_

---

## Week 11 — Imaging inverse problems I

- _(≤10 must-know statements — filled when this week's deck is authored)_

---

## Week 12 — Imaging inverse problems II

- _(≤10 must-know statements — filled when this week's deck is authored)_

---

## Week 13 — Explainability, trust & synthesis

- _(≤10 must-know statements — filled when this week's deck is authored)_
