# MFML External-Assets Todo (post-book gaps)

The book-figure pass left specific gaps where the source corpus (newest book = 2024
Neuer/Sandfeld; rest 2006–2021) is too old. This file lists those gaps as one task
per unit so agents can run in parallel.

## Common context (paste at top of every agent prompt)

```
Lecture root:   /home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26
MFML root:      $LECTURE/_public_presentations/mathematical_foundations_of_ai_and_ml

You will fetch missing figures from EXTERNAL sources (arXiv papers, project docs,
Wikimedia Commons, Distill.pub) and integrate them into ONE unit's slides. Stay
strictly in your unit folder.

PREFERRED SOURCES (in priority order)
  1. Wikimedia Commons / Wikipedia (CC-BY-SA / public domain — always safe)
  2. Distill.pub (CC-BY 4.0)
  3. Project / library documentation (scikit-learn, SHAP, LIME, PyTorch, JAX, etc. —
     usually MIT/BSD/Apache, figures reusable with attribution)
  4. arXiv paper PDFs — download with curl, extract figure with `pdfimages` or by
     screenshot. Academic teaching use is fair-use; ALWAYS attribute.
  5. Author technical blogs (Jay Alammar, Lilian Weng, Sebastian Raschka,
     distill.pub posts) — only when the post explicitly permits reuse, otherwise
     note URL and let Philipp decide. Do NOT just hotlink.

WORKFLOW PER FIGURE
  1. Use WebSearch to find candidate source(s).
  2. Use WebFetch to read the candidate page and check licensing — for
     Wikimedia, look at the file page; for arXiv, use the paper's official PDF
     (links from arxiv.org/abs/...).
  3. Download with `curl -L -o /tmp/<name>.png <URL>` (or `wget`). Don't store
     in /tmp permanently — copy to <unit>/images/ with a descriptive
     snake_case name.
     - For arXiv PDFs: `curl -L -o /tmp/paper.pdf https://arxiv.org/pdf/XXXX.YYYYY`
       then either `pdftoppm -png -r 200 -f <page> -l <page> /tmp/paper.pdf
       /tmp/page` to rasterize the page (then crop manually if needed) or
       `pdfimages -png -f <page> -l <page> /tmp/paper.pdf /tmp/img` to extract
       embedded images. If neither works cleanly, save the URL + page number
       to your report so Philipp can grab it manually.
  4. If the page has a clean PNG/SVG already (Wikimedia, blog hero images), just
     download that file directly.
  5. Edit `01_intro.qmd` to use the figure. Use a captioned form so attribution
     is on-slide:
         ![Source: Vaswani et al. 2017](images/transformer_block.png){width=70%}
     For two-column slides (`:::: {.columns}` blocks) put image in one column,
     keep bullets in the other. DO NOT touch the YAML header.
  6. Add a BibTeX entry to the unit's `ref.bib` for each cited paper, in
     biblatex `@article` form matching existing style (no DOI required, but
     include eprint=XXXX.YYYYY and archiveprefix=arXiv when applicable). Then
     replace the on-slide `Source: ...` caption with a Quarto cite, e.g.
     `[@vaswani2017attention]`.

HARD RULES
  - No hotlinking. Always download to images/.
  - No Co-Authored-By lines anywhere.
  - If you can't find a clean source after a reasonable search, REPORT the
     gap with your best candidate URL — don't fabricate or use a misleading
     stand-in. Philipp prefers "no figure, will source manually" over a wrong one.
  - Stay strictly inside YOUR unit folder.

REPORT
  - Files downloaded: source URL → dest path
  - Slides edited: line ranges in 01_intro.qmd
  - BibTeX entries added to ref.bib
  - Figures you couldn't get cleanly: list URL + reason so Philipp can grab manually
```

---

## Unit 7 — Generalization & Bias-Variance

Folder: `07_generalization_bias_variance/`
Gaps to fill:
- **Double-descent curve**: classic figure from Belkin et al. 2019 ("Reconciling
  modern machine learning practice and the bias-variance trade-off") Fig 1, OR
  Nakkiran et al. 2020 ("Deep Double Descent") Fig 1. Both on arXiv (1812.11118
  and 1912.02292). Wikipedia "Double descent" page has a clean reproduction
  too — check license.
- **k-fold cross-validation diagram**: scikit-learn docs has the canonical
  fold-rotation diagram at
  https://scikit-learn.org/stable/modules/cross_validation.html — BSD-licensed.
  Wikipedia "Cross-validation (statistics)" also has one.

---

## Unit 9 — Latent Spaces (UMAP, Contrastive, CLIP)

Folder: `09_latent_spaces_advanced/`
Gaps to fill:
- **UMAP visualization**: McInnes et al. 2018 (arXiv 1802.03426) Fig 5/6 of
  MNIST or COIL-20 embedding. UMAP project page (umap-learn.readthedocs.io) has
  example gallery — BSD-3.
- **SimCLR contrastive framework**: Chen et al. 2020 (arXiv 2002.05709) Fig 2
  is the canonical augmentation+encoder+projection-head diagram. Lilian Weng's
  blog post "Contrastive Representation Learning" has a clean redraw.
- **CLIP architecture**: Radford et al. 2021 (arXiv 2103.00020) Fig 1 is the
  canonical contrastive image-text training diagram.
- **InfoNCE / NT-Xent loss diagram** (optional): SimCLR paper covers it; or
  Lilian Weng's blog.

---

## Unit 10 — Attention & Transformers

Folder: `10_attention_transformers/`
This is the unit with the longest list — the agent already added RNN/LSTM
figures from books, so focus only on transformer-era topics.

Gaps to fill:
- **Scaled dot-product attention diagram**: Vaswani et al. 2017 (arXiv
  1706.03762) Fig 2 left.
- **Multi-head attention diagram**: Vaswani et al. 2017 Fig 2 right.
- **Transformer block (full encoder/decoder)**: Vaswani et al. 2017 Fig 1.
- **Positional encoding visualization**: Jay Alammar "The Illustrated
  Transformer" has a clean version, OR plot it directly from the formula
  (a quick matplotlib inline figure works too — pcolormesh of sin/cos values).
- **ViT patch tokenization**: Dosovitskiy et al. 2020 (arXiv 2010.11929) Fig 1.
- **Attention weight heatmap**: Bahdanau et al. 2014 (arXiv 1409.0473) Fig 3
  has the iconic alignment heatmap, OR Vaswani 2017 Figs 3–5.
- **CNN-vs-ViT data efficiency**: Dosovitskiy 2020 Fig 5 (the
  pre-training-set-size sweep).
- **Scaling laws / foundation-model figure**: Kaplan et al. 2020 (arXiv
  2001.08361) Fig 1, OR Hoffmann et al. 2022 Chinchilla (arXiv 2203.15556) Fig 2.

---

## Unit 11 — Generative Models: VAE & Diffusion

Folder: `11_generative_vae_diffusion/`
Gaps to fill:
- **VAE reparameterization trick**: Doersch 2016 ("Tutorial on Variational
  Autoencoders", arXiv 1606.05908) Fig 4 is the canonical "stochastic node /
  deterministic + ε" diagram. Kingma 2019 thesis also has a clean version.
- **U-Net architecture**: Ronneberger et al. 2015 (arXiv 1505.04597) Fig 1 —
  the iconic U-shape with skip connections. Wikipedia "U-Net" has a CC version.
- **Diffusion forward/reverse process schematic**: Ho et al. 2020 DDPM (arXiv
  2006.11239) Fig 2. Lilian Weng's blog "What are Diffusion Models?" has clean
  redraws — check her permissions.
- **Diffusion noise schedule**: typically a plot of β_t vs t (linear vs cosine).
  Nichol & Dhariwal 2021 (arXiv 2102.09672) Fig 5 has the cosine-vs-linear
  comparison. Could also be plotted directly in matplotlib.
- **Score-based / SDE perspective**: Song et al. 2021 ("Score-Based Generative
  Modeling through SDEs", arXiv 2011.13456) Fig 1.
- **Classifier-free guidance**: Ho & Salimans 2022 (arXiv 2207.12598) — figures
  are sparse, may need to use Lilian Weng's redraw or skip with a "see paper"
  pointer.
- **Latent diffusion / Stable Diffusion architecture**: Rombach et al. 2022
  (arXiv 2112.10752) Fig 3.

---

## Unit 12 — Uncertainty in Predictions

Folder: `12_uncertainty_in_predictions/`
Gaps to fill:
- **Deep ensembles**: Lakshminarayanan et al. 2017 (arXiv 1612.01474) Figs
  1–2 show ensemble disagreement on toy regression.
- **MC Dropout**: Gal & Ghahramani 2016 (arXiv 1506.02142) Fig 2 / Fig 4 has
  predictive-uncertainty bands from dropout.
- **Calibration / reliability diagram**: Guo et al. 2017 ("On Calibration of
  Modern Neural Networks", arXiv 1706.04599) Fig 1 is the iconic
  over-confidence reliability diagram. Niculescu-Mizil & Caruana 2005 also.

---

## Unit 13 — Physics-Informed Learning

Folder: `13_physics_informed_learning/`
Neuer ch 6 covered most basics — these are the modern-PIML gaps.

Gaps to fill:
- **PINN architecture & loss-balance diagram**: Raissi et al. 2019 ("Physics-
  Informed Neural Networks", J. Comp. Phys.; arXiv 1711.10561) Fig 1, OR
  Karniadakis et al. 2021 Nature Reviews Physics survey — clean schematics.
- **Soft PDE constraint / λ-weight sensitivity**: Wang et al. 2021 ("Understanding
  and mitigating gradient pathologies in PINNs", arXiv 2001.04536) — has loss
  landscape figures.
- **DeepONet**: Lu et al. 2021 (arXiv 1910.03193) Fig 1 — branch + trunk net
  architecture.
- **Fourier Neural Operator (FNO)**: Li et al. 2020 (arXiv 2010.08895) Fig 2 —
  the spectral-convolution block diagram.
- **Hamiltonian Neural Networks**: Greydanus et al. 2019 (arXiv 1906.01563)
  Fig 1 / Fig 2 (mass-spring or pendulum trajectory comparison).
- **Group-equivariant / E(3) NN** (optional): Cohen & Welling 2016 (arXiv
  1602.07576) — equivariance diagrams.
- **PINN failure case**: Krishnapriyan et al. 2021 ("Characterizing possible
  failure modes in PINNs", arXiv 2109.01050) Fig 1 — failure on convection.

---

## Unit 14 — Explainability, Limits, Trust

Folder: `14_explainability_limits_trust/`
Modern-XAI gaps:
- **SHAP waterfall plot**: SHAP package docs
  (shap.readthedocs.io/en/latest/example_notebooks/) has clean rendered
  examples, MIT-licensed. Lundberg & Lee 2017 (arXiv 1705.07874) for the
  paper-figure form.
- **SHAP beeswarm / summary plot**: same SHAP docs.
- **LIME**: Ribeiro et al. 2016 ("Why Should I Trust You?", arXiv 1602.04938)
  Fig 3 (linear-approximation diagram) and Fig 4 (text/image explanation
  examples). LIME GitHub README has example outputs.
- **Integrated Gradients**: Sundararajan et al. 2017 (arXiv 1703.01365) Figs
  1–4 — attribution heatmaps on images.
- **Counterfactual explanations**: Wachter et al. 2017 (arXiv 1711.00399) and
  Mothilal et al. 2020 DiCE (arXiv 1905.07697).
- **Fairness / bias plots**: Hardt et al. 2016 ("Equality of Opportunity",
  arXiv 1610.02413) Fig 1; or IBM AI Fairness 360 docs; or Aequitas project.
- **Out-of-distribution detection**: Hendrycks & Gimpel 2017 baseline (arXiv
  1610.02136) Figs 1–2; or DeepMind Gemini OOD blog post if it has a clean
  schematic.

---

## Suggested dispatch order

Highest ROI first (most missing figures, most pedagogically central):

1. Unit 10 (8 figures — core Vaswani diagrams)
2. Unit 11 (7 figures — diffusion/VAE foundations)
3. Unit 14 (6+ figures — modern XAI suite)
4. Unit 13 (5+ figures — DeepONet/FNO/HNN)
5. Unit 12 (3 figures — small but important)
6. Unit 9 (3 figures — UMAP/SimCLR/CLIP)
7. Unit 7 (2 figures — quick win)

All 7 are independent — you can dispatch them in one parallel batch.
