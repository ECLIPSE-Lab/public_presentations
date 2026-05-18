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
9. **The EM leakage trap (crop-vs-specimen):** if crops from the same EM specimen appear in both train and test, the model memorises specimen identity rather than the physical property. Fix: `GroupKFold(groups=specimen_id)` — entire specimens are in either train or test, never both. This is the default CV strategy whenever a `specimen_id` column exists.
10. **Segmentation metrics:** IoU = $|A\cap B|/|A\cup B|$; Dice = $2|A\cap B|/(|A|+|B|) = 2\,\text{IoU}/(1+\text{IoU})$. Both range $[0,1]$; $R^2 < 0$ is possible and means the model is worse than predicting the mean. For imbalanced classification, report precision and recall, not accuracy.

---

## Week 5 — Neural networks from first principles

- _(≤10 must-know statements — filled when this week's deck is authored)_

---

## Week 6 — CNNs for microscopy images

- _(≤10 must-know statements — filled when this week's deck is authored)_

---

## Week 7 — Beating small & expensive data

- _(≤10 must-know statements — filled when this week's deck is authored)_

---

## Week 8 — Unsupervised learning & autoencoders

- _(≤10 must-know statements — filled when this week's deck is authored)_

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
