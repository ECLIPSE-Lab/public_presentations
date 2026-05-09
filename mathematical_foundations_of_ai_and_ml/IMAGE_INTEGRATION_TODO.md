# MFML Image-Integration Todo (Units 5–14)

One task per unit. Each task is self-contained — copy the whole block into an Agent
dispatch prompt. Tasks are independent and can run in parallel.

## Common context (paste at top of every agent prompt)

```
Lecture root:   /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26
MFML root:      $LECTURE/_public_presentations/mathematical_foundations_of_ai_and_ml
Books root:     $LECTURE/books

Books available (each has images/ + markdown/ + table_of_contents.md):
  - bishop-pattern-recognition-and-machine-learning-2006     (PRML; 2280 figures, named *.pdf-PAGE-IDX.png)
  - murphy-machine-learning-a-probabilistic-perspective-2012 (Murphy; 3732 figures, named *.pdf-PAGE-IDX.png)
  - mcclarren-machine-learning-for-engineers                 (per-chapter dirs 477010_1_En_<N>_Chapter/, mostly TeX equation pngs + a few real figures)
  - neuer-machine-learning-for-engineers                     (per-chapter dirs 601270_1_En_<N>_Chapter/)
  - sandfeld-materials-data-science                          (961 figures, named sandfeld_-_Materials_Data_Science.pdf-PAGE-IDX.png)

Workflow for every unit (do all of these):
  1. Read the unit's plan + content file (`unitN_plan.md`, `unitN_content_50slides.md`)
     and `01_intro.qmd` so you know what topics need illustrating.
  2. Identify which book chapter(s) match the unit topic. Use each book's
     table_of_contents.md to pin down the chapter, then look at the chapter's
     markdown file (markdown/NN-<chapter-slug>.qmd) — it lists the figure
     filenames inline so you can pick by caption, not by guessing filenames.
  3. Pick 5–15 high-value figures (real diagrams/plots — NOT TeX-equation
     screenshots from the McClarren/Neuer chapter dirs).
  4. Copy the chosen images into `<unit-folder>/images/` (mkdir if missing).
     Rename them to short, descriptive snake_case (e.g. `kmeans_voronoi.png`)
     so they read cleanly in the slides. Keep a short note of the source
     (book + page) — you do NOT need to add it to slides, but keep it in
     your final report so Philipp can verify attribution later.
  5. Edit `01_intro.qmd` to actually USE the images. Insert them on the
     slides where they illustrate a point already made in the slide text.
     Use Quarto/reveal.js syntax:
         ![](images/kmeans_voronoi.png){width=70%}
     For two-column slides (most of these qmds use `:::: {.columns}` blocks),
     put the image in one column and keep the existing bullet list in the
     other. DO NOT replace existing prose — augment it. DO NOT touch the
     YAML header.
  6. Render check is NOT required — just make sure the file is well-formed
     Quarto (no broken `:::` blocks, no stray backticks).

Hard rules:
  - Don't copy TeX equation screenshots (filenames like *_TeX_Equ*.png) —
    Philipp already typesets math in LaTeX directly.
  - Don't copy book-cover or table-of-contents page scans.
  - Don't add Co-Authored-By lines anywhere.
  - Stop and report (don't guess) if a unit's topic isn't covered by any
    book — better to add zero images than misleading ones.

Report at the end:
  - Files copied (source path → dest path)
  - Slides that were edited (line ranges)
  - Topics you couldn't find a good figure for
```

---

## Unit 5 — Clustering & Autoencoders

Folder: `05_clustering_and_autoencoders/`
Status: NO `images/` dir yet — create it.

Likely-relevant chapters:
- Murphy ch 25 (Clustering) — k-means, hierarchical, spectral
- Murphy ch 12 (Latent Linear Models) — PCA / FA / autoencoder framing
- Murphy ch 28 (Deep Learning) — autoencoder figures
- Bishop ch 9 (Mixture Models and EM) — k-means/GMM diagrams
- Bishop ch 12 (Continuous Latent Variables) — PCA/PPCA
- McClarren ch 4 (Data Reduction and Clustering) — k-means + t-SNE worked example
- Sandfeld Part III (Classical ML) — clustering examples on materials data

Want figures for: k-means iterations, GMM contours, hierarchical dendrogram,
silhouette, autoencoder architecture, denoising-AE before/after, latent-space
embedding scatter.

---

## Unit 6 — Loss Landscapes & Optimization

Folder: `06_loss_landscapes_optimization/`
Status: `images/` exists (already has `landscapes.png`, `opt_paths.png`). Add to it.

Likely-relevant chapters:
- Bishop ch 5 (Neural Networks) — error-surface contour figures, gradient flow
- Murphy ch 8 (Logistic Regression) — SGD trajectories
- Murphy ch 28 (Deep Learning) — momentum / Adam comparisons, saddle points
- McClarren ch 5 (Feed-Forward NN) — SGD discussion + figures
- Neuer ch 5 — gradient-descent visualizations

Want figures for: 1D/2D loss curves, saddle points, momentum vs vanilla SGD
trajectory, Adam vs SGD comparison, learning-rate effect, flat vs sharp minima.

---

## Unit 7 — Generalization, Bias-Variance

Folder: `07_generalization_bias_variance/`
Status: `images/` exists. Add to it.

Likely-relevant chapters:
- Bishop ch 1 §1.1 + §3.2 — the classic bias-variance polynomial-fit figures
  (these are the canonical illustrations — definitely grab them)
- Bishop ch 3 (Linear Models for Regression) — regularization paths
- Murphy ch 6 (Frequentist Statistics) — bias-variance decomposition figures
- Murphy ch 7 (Linear Regression) — train/test error vs model complexity
- McClarren §1.1.1.1 (Overfitting) — overfitting case figures
- Sandfeld Part III — train/val curves on materials data

Want figures for: overfit polynomial degree sweep, training vs test error
curves (the U-shape), bias-variance tradeoff sketch, double-descent (if
present), regularization effect on weights.

---

## Unit 8 — Probabilistic View of Learning

Folder: `08_probabilistic_view_of_learning/`
Status: `images/` exists. Add to it.

Likely-relevant chapters:
- Bishop ch 1 §1.2 (probability), ch 2 (Probability Distributions) — Gaussian
  contour, Beta/Dirichlet shapes
- Bishop ch 3 (Linear Models) — predictive distribution figure
- Murphy ch 2 (Probability) — distribution panels
- Murphy ch 3 (Generative Models for Discrete Data) — Bayesian-update figs
- Murphy ch 5 (Bayesian Statistics) — posterior shrinkage figures
- McClarren §1.4 (Bayesian Probability)
- Neuer ch 2 (Mathematical Description of Data)
- Sandfeld Part II (Probabilities/Distributions/Statistics)

Want figures for: prior/likelihood/posterior, conjugate update animations
(static frames), MAP vs MLE, Gaussian likelihood contour, Beta-Bernoulli
update sequence.

---

## Unit 9 — Latent Spaces (t-SNE, UMAP, Contrastive)

Folder: `09_latent_spaces_advanced/`
Status: `images/` exists. Add to it.

Likely-relevant chapters:
- McClarren ch 4 (Data Reduction and Clustering) — has a worked t-SNE example
  with figures (foliage spectra case study)
- Bishop ch 12 (Continuous Latent Variables) — PCA, probabilistic PCA
- Murphy ch 12 (Latent Linear Models) — PCA, ICA, FA
- Murphy ch 27 (Latent Variable Models for Discrete Data) — topic models
- Murphy ch 28 (Deep Learning) — embedding visualizations

Note: UMAP and modern contrastive learning (SimCLR, etc.) are too recent for
all five books. Use book figures only for PCA / t-SNE / classical embeddings —
don't fake UMAP/contrastive figures from older diagrams.

Want figures for: PCA projection of MNIST/Iris-style data, t-SNE 2D embedding
of high-dim data, manifold/Swiss-roll, embedding-space arithmetic if present.

---

## Unit 10 — Attention & Transformers

Folder: `10_attention_transformers/`
Status: `images/` exists. Add to it.

Limitations: All five books predate transformers (Vaswani 2017) or only
mention them in passing. Be conservative.

Likely-relevant chapters:
- McClarren ch 7 (RNNs for Time Series) — RNN unrolled diagram, LSTM cell
- Neuer (check ch 6/7) — RNN/sequence figures if present
- Murphy ch 17 (Markov & HMM) — sequence-model diagrams
- Murphy ch 18 (State Space Models)
- Murphy ch 28 (Deep Learning) — has sequence-to-sequence figures

Want figures for: RNN unrolled, LSTM gate diagram, sequence-to-sequence
encoder-decoder. STOP THERE — don't try to find attention/transformer
diagrams in these books; just report "no transformer figures available, use
external assets."

---

## Unit 11 — Generative Models: VAE & Diffusion

Folder: `11_generative_vae_diffusion/`
Status: `images/` exists. Add to it.

Limitations: Diffusion is too recent for all five books. VAE is borderline —
Murphy 2012 came right around Kingma's VAE paper.

Likely-relevant chapters:
- Bishop ch 11 (Sampling Methods) — MCMC, Gibbs, ancestral sampling figures
- Bishop ch 12 (Continuous Latent Variables) — generative latent-variable
  framing
- Murphy ch 23 (Monte Carlo Inference)
- Murphy ch 24 (MCMC) — chain-mixing diagrams, Metropolis
- Murphy ch 28 (Deep Learning) — generative model figures, possibly VAE
- Bishop ch 8 (Graphical Models) — directed-graph plate notation for
  generative models

Want figures for: MCMC chain trace, ancestral sampling from a Bayes net,
plate-notation diagram of a latent-variable generative model. Report
"diffusion figures not available, use external assets" — don't fabricate.

---

## Unit 12 — Uncertainty in Predictions

Folder: `12_uncertainty_in_predictions/`
Status: `images/` exists. Add to it.

Likely-relevant chapters:
- Bishop ch 3 (Linear Models for Regression) — predictive variance bands
  (the gray/pink uncertainty-ribbon figure is iconic)
- Bishop ch 6 (Kernel Methods) and ch 9 — Gaussian Process predictive
- Murphy ch 7 (Linear Regression) — predictive distribution
- Murphy ch 15 (Gaussian Processes) — GP regression with uncertainty bands
- Murphy ch 5 (Bayesian Statistics) — posterior predictive
- Sandfeld Part II — error bars / confidence intervals

Want figures for: GP regression with 2σ band, Bayesian linear regression
predictive band, calibration plot if present, ensemble disagreement.

---

## Unit 13 — Physics-Informed Learning

Folder: `13_physics_informed_learning/`
Status: `images/` exists. Add to it.

Limitations: PINNs are too recent for these books, but adjacent material exists.

Likely-relevant chapters:
- McClarren §2.5 (Determining Governing Equations from Data) — SINDy-style
  case study with figures
- McClarren ch 5 case study (Strength of Concrete) — physics-feature engineering
- Sandfeld — domain examples on materials data
- Murphy ch 16 (Adaptive Basis Function Models) — basis-function regression
  figures

Want figures for: data-driven equation discovery output, learned-vs-true
function plots, basis-function expansions. Report explicitly if no PINN-like
figures found — that's expected and useful.

---

## Unit 14 — Explainability, Limits, Trust

Folder: `14_explainability_limits_trust/`
Status: NO `images/` dir yet — create it.

Limitations: Modern XAI (SHAP/LIME/IG) postdates these books. Be conservative.

Likely-relevant chapters:
- McClarren ch 3 (Decision Trees and Random Forests) — feature-importance
  plots, tree visualizations
- Murphy ch 16 (Adaptive Basis Function Models) — partial-dependence-style
  figures
- Bishop ch 14 (Combining Models) — boosting-stage figures
- Sandfeld — interpretability examples on materials problems if any
- Neuer — likely has discussion of interpretability for engineering

Want figures for: decision-tree visualization, feature-importance bar chart,
partial-dependence plot. Otherwise, report "modern-XAI figures not available,
use external assets" rather than padding.
