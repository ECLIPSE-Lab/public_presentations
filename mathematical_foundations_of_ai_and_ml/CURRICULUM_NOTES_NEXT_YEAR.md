# Curriculum notes for SS27 (and future iterations)

Captured 2026-05-09 during SS26 mid-semester review. The SS26 reorganization
described below (units 5-14) is being applied this term; the SS27 items
needing pre-term preparation are flagged as such.

## SS26 (this year, applied now)

Units 1-4 are locked (already taught).

Backprop is **moved out of its own lecture slot** and is now self-study,
referenced from the end of Unit 4. The slide deck currently living in
`05_clustering_and_autoencoders/01_intro.qmd` (which contains the backprop
content despite the folder name) becomes the self-study supplement, plus the
two existing example notebooks (`18.3`, `18.5`). Unit 5's 90-min slot is
repurposed for **Clustering + Autoencoders**, matching what the folder name
already says — see `05_clustering_and_autoencoders/unit05_plan_gemini.md`
for the full plan.

### Final SS26 unit list

| # | Title | Status |
|---|-------|--------|
| 1 | Learning vs Data Analysis | locked, taught |
| 2 | Linear Algebra, PCA, SVD | locked, taught |
| 3 | Regression as Loss Minimization | locked, taught |
| 4 | NN Architectures and Convolutions | locked, taught (add: pointer to backprop self-study) |
| 5 | **Clustering & Autoencoders** | NEW for SS26 — see `unit05_plan_gemini.md` |
| 6 | Optimization for Deep Learning | trim from existing (~70 slides target) |
| 7 | Generalization, Regularization, Model Selection | keep |
| 8 | Probabilistic Foundations | slim; add 3-slide KL/entropy primer (needed for VAE in unit 11) |
| 9 | Latent Spaces & Advanced Representation Learning | repurposed: t-SNE, UMAP, contrastive learning, foundation embeddings (goes deeper than Unit 5's AE intro; old AE/AE-applications content is now in Unit 5) |
| 10 | **Attention & Transformers** | NEW |
| 11 | **Generative Models: VAE & Diffusion** | NEW |
| 12 | Uncertainty Quantification | keep GPs; mention deep ensembles + conformal |
| 13 | Physics-Informed Learning | keep |
| 14 | Explainability, Limits, Trust | keep |

The merge mechanic that made room for transformers + diffusion:
- Old Unit 11 (full unsupervised: K-means, GMM, EM) is now folded into the new Unit 5 (clustering half).
- Old Unit 9 (full representation learning: AE-heavy) is now folded into the new Unit 5 (AE half) at intro depth, with deeper material moving into Unit 9's new role.
- Old Unit 10 (latent spaces, t-SNE) merges with the deeper-AE material into the new Unit 9 (latent spaces & advanced representation learning).
- Net: -3 slots from merges, +2 new slots (transformers, generative). One slot was reclaimed by retiring the standalone backprop unit.

### Required SS26 follow-up work

- **Unit 5 plan → slides:** generate `01_intro.qmd` from `unit05_plan_gemini.md`. Old backprop deck must first be moved aside (rename or move to a clearly marked self-study path).
- **Unit 4 closing slides:** add 2-3 slides linking to the backprop self-study supplement, with a one-paragraph chain-rule sketch and a pointer to the deliverable in next exercise.
- **Unit 9, 10, 11 re-write:** old decks for these units exist but now serve different purposes. Plan and content files need refresh:
  - new Unit 9 = latent spaces & advanced representation
  - new Unit 10 = transformers
  - new Unit 11 = generative (VAE + diffusion)
- ~~**Unit 8:** insert KL/entropy primer slides.~~ **Done 2026-05-09** — 3 slides (entropy, KL, KL between Gaussians incl. diagonal-MVN VAE form) inserted after the MVN section in `08_probabilistic_view_of_learning/01_intro.qmd`; learning outcomes and must-know list updated.
- ~~**Unit 6:** trim from ~85 to ~70 slides.~~ **Not needed** — actual deck is ~52 headers, already matching the planned 50-slide pack (`unit6_content_50slides.md`). The "~85" estimate in this note was wrong.
- **Folder renames** for clarity (do at term end to avoid breaking references mid-term):
  - `05_clustering_and_autoencoders/` → keep (now matches content)
  - `09_representation_learning/` → consider renaming to e.g. `09_latent_spaces_advanced/` once new content is in
  - new folders `10_attention_transformers/`, `11_generative_vae_diffusion/`

## SS27 (prepare next year)

### Unit 1 — Learning vs Data Analysis
- **Trim from ~85 slides to ~60.**
- Keep: ERM framing, loss zoo, train/val/test, leakage, trust checklist, epistemology framing.
- Move out: bias-variance details (full treatment in Unit 7), uncertainty taxonomy (full in Unit 8 / 12). Keep only 1-slide previews.

### Unit 2 — Linear Algebra, PCA, SVD
- Add 1 slide previewing how SVD reappears in attention (low-rank Q/K projections), motivating Unit 10.

### Unit 3 — Regression as Loss Minimization
- Add a tree-based methods section (~15 min): decision trees, random forests as variance reduction, gradient boosting / XGBoost as bias reduction. Materials students reach for these first on tabular data; currently absent.

### Unit 4 — NN Architectures and Convolutions
- **Append a proper backprop section** (~20 min, ~10 slides) at the end. The chain-rule derivation, delta recursion, vanishing/exploding gradients, ReLU + init story. This consolidates what was self-study in SS26 into the lecture proper.
- The existing detailed backprop deck (in `05_clustering_and_autoencoders/`) becomes the source material — extract the 10-15 most essential slides.

### Unit 5 — Clustering & Autoencoders (formerly the new SS26 unit)
- Refine based on SS26 student feedback. Likely candidates: more time on EM intuition (always under-served at intro depth), more on choosing autoencoder architecture.

### Misc improvements
- **Reduce redundancy across units.** Bias-variance: 1 + 7. Aleatoric/epistemic: 1 + 8 + 12. Regularization: 1 + 3 + 7 + 8. Pick canonical homes; replace duplicates with 1-slide back-references.
- **Statistical learning theory acknowledgment** (Unit 7 or 8): one slide pointing to VC dim, Rademacher, PAC bounds — students should know the theory exists.
- **Convex optimization baseline** (Unit 6): one slide on convergence rates in the convex case before non-convex DL.
- **SVM / kernel trick** (Unit 3 or as Unit 12 sidebar): the dual formulation and kernel trick are pedagogical bridges between linear and nonlinear; currently kernels are first introduced (in GPs) without this build-up.
- **Modern workflow lecture** (candidate for an extra slot or part of Unit 14): pre-trained foundation models, fine-tuning, evaluation, MLOps basics.

## Reference: original SS26 plan (pre-reorg, for the record)

Original 14 units before this reorganization:
1. Learning vs Data Analysis · 2. LA/PCA/SVD · 3. Regression · 4. NN Architectures · **5. Backprop** · 6. Loss Landscapes · 7. Generalization · 8. Probabilistic · **9. Representation Learning** · **10. Latent Spaces** · **11. Unsupervised** · 12. Uncertainty · 13. Physics-Informed · 14. Explainability

The bolded units changed roles in the reorg.
