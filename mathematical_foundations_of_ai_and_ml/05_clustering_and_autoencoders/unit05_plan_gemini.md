# Unit 5 Plan — Clustering & Autoencoders

## Audience + constraints
- BSc AI-Material Technology, 4th semester
- Prior knowledge: 2 semesters math, 1 physics, some chemistry
- Assumed: Units 1-4 completed (risk minimization, linear algebra, regression, NN architectures)
- **Backprop is self-study this term** — students get a handout/notebook deriving the chain rule and forward/backward pass for a 2-layer net. The deck previously living in this folder (`01_intro.qmd`, `unit5_content_50slides.md`) is repurposed as that self-study supplement and referenced from the end of Unit 4. The 90-min lecture slot is now Unit 5 = clustering + autoencoders.
- Lecture: 90 min + 90-min exercise
- Language: English (German translation later)

## Why this unit, why now
The course has so far been **supervised**: input → loss → fit. Unsupervised learning is the natural next step and the scaffold for Units 8-11 (probabilistic, latent spaces, generative). Two complementary perspectives meet here:
- **Classical clustering** (K-means, GMM/EM): closed-form, geometric, probabilistic.
- **Neural autoencoders**: nonlinear, learned, scalable.
The deliberate pairing lets students see the same idea — *finding structure without labels* — through two lenses that students will keep encountering throughout the course.

## Learning objectives
By the end of the unit, students can:
1. Distinguish supervised from unsupervised learning and recognize when each applies in a materials/engineering setting.
2. Formulate the K-means objective, run the algorithm by hand on a small dataset, and explain its convergence behavior and failure modes.
3. State the GMM likelihood, intuit the responsibility/E-step and parameter/M-step, and explain why EM is needed when latent variables are present.
4. Describe the autoencoder architecture (encoder, bottleneck, decoder, reconstruction loss) and justify why a linear AE recovers PCA.
5. Argue why nonlinearity + bottleneck enable learning curved manifolds.
6. Use autoencoders for two practical tasks: data compression (e.g., spectra) and anomaly detection (reconstruction error).

## Book-aligned content mapping (priority order)
1. **Neuer (2024) Ch. 5** — Unsupervised learning (K-means, GMM intuition, autoencoders for materials data). *Required reading.*
2. **McClarren (2021) Ch. 4** — Finding structure: data reduction and clustering (K-means worked example, K-medoids, GMM). *Required reading.*
3. **McClarren (2021) Ch. 8** — Unsupervised learning with neural networks: autoencoders. *Required reading.*
4. **Bishop (2006) Ch. 9** — Mixture models and EM (depth pointer for the curious; full derivation is **not** lecture-essential).
5. **Murphy (2012)** Ch. 11 (mixture models, EM) and Ch. 28 (deep generative models) — supplementary depth.
6. **Sandfeld** — Materials Data Science chapter on unsupervised methods (materials examples).

## 90-minute structure

| Time | Block | Content |
|------|-------|---------|
| 0–5 | Recap | Where we are in the course; what changes when labels disappear. |
| 5–12 | Unsupervised landscape | Types (clustering, density, manifold, generative); when to use each; materials motivations (no labels, expensive labels, exploration). |
| 12–28 | **K-means** | Objective (within-cluster SS), Lloyd's algorithm, convergence, k-means++ init, choosing K (elbow + silhouette), spherical assumption, sensitivity to outliers. K-medoids as a robust variant (1 slide). |
| 28–48 | **GMM + EM** | From hard to soft assignments. The mixture likelihood. Why MLE is hard with latent variables. EM as alternating optimization: E-step computes responsibilities, M-step updates parameters. **No full ELBO derivation** — intuition + 1 slide that names the lower-bound view. Comparison K-means vs GMM (constraint shapes). |
| 48–55 | Bridge | Where do clustering's "centroids" come from? They are *learned representations*. This sets up autoencoders. |
| 55–75 | **Autoencoders** | Encoder/bottleneck/decoder architecture; reconstruction MSE loss; **linear AE = PCA** (state, don't prove); nonlinearity unlocks curved manifolds; convolutional AEs (1 slide); training as standard NN with autodiff (i.e., students don't need to redo backprop). |
| 75–82 | AE variants + applications | Denoising AE (1-2 slides), sparse AE (mention), bottleneck-as-feature-extractor for downstream tasks. **Two flagship applications**: spectral compression and anomaly detection via reconstruction error. |
| 82–88 | Materials worked examples | (a) Alloy composition clustering and discovery of phase families; (b) compressing 2000-channel spectra to 16 latents; (c) reconstruction-error spike as defect signal in micrographs. |
| 88–90 | Wrap | Three exam-must-knows; reading; bridge to Unit 6 (loss landscapes — same optimizer story, harder objective). |

## Slide budget (50 slides — as written in `01_intro.qmd`)

- **Block A — Framing (slides 1–5):** where we are, self-study backprop note, the leap, learning outcomes, roadmap.
- **Block B — Unsupervised landscape (6–8):** types, what counts as structure, why it matters in materials.
- **Block C — K-means (9–16):** objective, Lloyd's algorithm, worked example, initialization, k-means++, elbow, silhouette, spherical assumption, K-medoids.
- **Block D — Hierarchical clustering (17–18):** agglomerative idea + linkage, dendrograms.
- **Block E — GMM + EM (19–26):** hard→soft, mixture model, latent variable view, E-step, M-step, alternating optimization, K-means vs GMM, BIC for choosing K.
- **Block F — Bridge to AE (27–28):** centroids → continuous latents, manifold hypothesis.
- **Block G — Autoencoders (29–35):** architecture, reconstruction loss, linear AE = PCA, why nonlinearity, choosing bottleneck, conv AE, training (autograd handles it — self-study supplement is the chain rule).
- **Block H — Variants + applications (36–41):** denoising AE, sparse AE, compression, anomaly detection, downstream features.
- **Block I — Latent space teaser (42–44):** latent as coordinate system, latent arithmetic, bridge to Units 9 + 11.
- **Block J — Materials examples + close (45–50):** alloy clustering, spectral compression, defect anomaly, three must-knows, reading + Unit 6 bridge, notebook companion + self-study reminder, learning outcomes recap.

This is denser than the original 50-slide budget intended and pulls hierarchical clustering, BIC, the manifold hypothesis, and the latent-arithmetic teaser from old units 9/10/11 — the units those topics came from no longer exist as standalone slots in the SS26 reorg.

## Reusable equations (math boxes)

- K-means objective: $J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$
- K-means update: $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$
- GMM density: $p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$
- Responsibility: $\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i;\mu_k,\Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i;\mu_j,\Sigma_j)}$
- M-step (means): $\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}$
- AE objective: $\mathcal{L}(\theta,\phi) = \frac{1}{N}\sum_i \|x_i - g_\theta(f_\phi(x_i))\|^2$
- Linear AE result: with $f_\phi(x) = W_e x$ and $g_\theta(z) = W_d z$, MSE-optimal $W_e, W_d$ span the top-$d$ PCA subspace.

## Exercise (90 min)

1. **K-means on alloy compositions** (NumPy from scratch): implement Lloyd's algorithm, run on a 4-D composition dataset, sweep $K$, plot elbow + silhouette.
2. **K-means vs GMM on synthetic data**: generate two-Gaussian mixture with elongated covariance; show K-means fails (spherical assumption), GMM (sklearn) succeeds.
3. **Train a small autoencoder** on Fashion-MNIST (or 1-D spectra if available): encoder/decoder MLP, 2-D bottleneck, plot reconstructions + 2-D latent scatter colored by class.
4. **Anomaly detection**: corrupt 5% of test images (block-out / extra noise), use reconstruction MSE to flag anomalies; report ROC-AUC.
5. **Bonus**: replace MLP encoder/decoder with a small conv stack; compare reconstruction quality at the same bottleneck dim.

The exercise *uses* PyTorch's autograd freely — by design. Backprop is a black box at the use-site here; the self-study handout is the place where students open that box.

## Assessment alignment
- Three "must-know" exam statements:
  1. K-means minimizes within-cluster sum of squares; convergence is to a local optimum and depends on initialization.
  2. EM alternates E-step (compute responsibilities) and M-step (update parameters); each step never decreases the data log-likelihood.
  3. A linear autoencoder with MSE loss recovers the PCA subspace; nonlinearity + bottleneck generalize this to curved manifolds.
- Exam-format examples: hand-trace one Lloyd iteration on 4-6 points; identify which clustering method fits a given data shape; compute reconstruction error for a small linear AE.

## Slide production notes
- Style: follow data_science_for_em revealjs pattern (consistent with units 1-4).
- Use the same recap-glossary slide pattern at the start, three-must-knows + reading at the end.
- Cite Neuer/McClarren on every borrowed figure.
- Reuse the existing `example_notebooks/` directory: the old backprop notebooks (`18.3`, `18.5`) are referenced from the *self-study supplement* (linked from Unit 4), not from this lecture. New notebooks needed: K-means, GMM, AE training, anomaly detection.

## Self-study supplement (Unit 4 → Unit 5 bridge)
Students receive a self-study packet covering backpropagation:
- The existing `01_intro.qmd` slide deck in this folder, repurposed and clearly labeled "Self-study: backpropagation."
- The existing `unit5_content_50slides.md` (rename to `backprop_self_study_content.md`).
- Notebooks `18.3_Backpropagation--Introduction_and_Example.ipynb` and `18.5_Python_Implementation_and_Example_for_the_FCN.ipynb`.
- A short reading prompt: derive the gradients for a 2-layer net by hand, then verify against `loss.backward()`. 30-min effort.
- Deliverable: include in the next exercise sheet a small "warm-up" question that requires one chain-rule derivation, so the self-study has teeth.
