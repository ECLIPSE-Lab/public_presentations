# MFML Unit 03 — Regression and Classification as Loss Minimization (Redesign Spec)

**Status:** Draft for review
**Date:** 2026-04-26
**Author:** Philipp Pelz (with Claude)
**Target file:** `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`
**Course:** MFML (Mathematical Foundations of AI & ML), SS26, week 3

## 1. Goal

Replace the current ~1148-line, ~60-slide unit 03 with a tight 90-minute, **49-slide** lecture (47 content + 2 checkpoint mini-quiz slides) that:

1. Devotes the first 20 minutes to optimization, including **second-order methods** (Newton, IRLS, quasi-Newton) — currently absent from the curriculum.
2. Adds a formal treatment of **basis function expansion**, **Runge's phenomenon**, **RBFs**, and **splines** — currently absent.
3. Closes with the **exponential family / GLM / link function** unification, with **IRLS as Newton's method on a GLM** — currently absent and recommended as the "grand conclusion" by the existing `missing_topics_analysis.md`.
4. Stops duplicating material that has dedicated coverage elsewhere in the curriculum.
5. Adheres to best teaching practice: Bloom-staged objectives, one-idea-per-slide, interactive at every conceptual peak, explicit forward/backward references, two checkpoint slides.

## 2. Curricular position

This unit sits in week 3 of MFML SS26 and is part of the SS26 teaching triad (MFML → MG → ML-PC). Adjacent units that constrain the scope of unit 03:

| Unit | Owns |
|---|---|
| MFML unit 02 (`02_linear_algebra_pca_svd`) | Linear regression matrix form, normal equations, Ridge closed form + spectral view, L1/L2 geometric intuition (constraint regions), multicollinearity, pseudo-inverse, Gram matrix, kernel hint, "LA → optimization" bridge. |
| MFML unit 06 (`06_loss_landscapes_optimization`) | Full deep dive on first-order optimization: Hessian for classifying critical points, conditioning, saddle points, ill-conditioned bowls, momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW. **Multiple existing interactives.** |
| MFML unit 07 (`07_generalization_bias_variance`) | Full bias-variance decomposition, polynomial-regression example, ridge-vs-lasso geometric tradeoff, generalization gap, complexity spectrum. |
| MFML unit 08 (`08_probabilistic_view_of_learning`) | Likelihood, log-likelihood, MLE, MLE-MSE connection, Bayesian inference, MAP, regularization-as-MAP, predictive distribution. |
| ML-PC unit 02 (`unit02_physics_of_data`) | Gaussian → MSE NLL derivation (§26), Poisson → Poisson NLL (§27), Bayes / MAP estimation with full ML↔Bayes dictionary table (§28). Already taught by week 3. |

Topics genuinely absent from the curriculum and therefore owned by unit 03:

- **Newton's method, IRLS, quasi-Newton** as optimization algorithms
- **Formal basis function expansion** $\boldsymbol\phi(\mathbf x)$, polynomial / RBF / spline bases, **Runge's phenomenon**
- **Exponential family canonical form, link functions, GLMs**
- The **IRLS = Newton-on-GLM** identity that ties the optimization opener to the GLM closer

## 3. Design decisions (locked through brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Lecture length | 90 min, ~47 slides | Tight foundations course slot. |
| Regularization treatment | Light bridge + bias-variance (~3 slides) | Unit 02 owns ridge/L1/L2 derivations; unit 07 owns bias-variance; unit 08 owns MAP. Unit 03 only adds the MAP one-liner connecting losses to priors. |
| Deep-learning regularization grab bag (dropout, batch-norm, label smoothing, focal loss, etc.) | **Cut** | Belongs in deep-learning units. Currently 8+ slides in unit 03 — wrong unit. |
| GLM / exponential family thread | **Trimmed-and-targeted: 5 slides** | Don't re-derive Gaussian→MSE / Bernoulli→CE in full (one-liners suffice; ML-PC §26-28 already done by week 3). Add only what's genuinely missing: canonical form, link functions, unification table, IRLS=Newton. |
| First-order optimization depth | **Compact intro only**; momentum/Adam = 1 forward-pointer slide | Unit 06 owns the deep dive with multiple interactives. Don't duplicate. |
| Existing interactives | Skip #1 (LR dynamics — duplicate of unit 06's "LR sensitivity demo"); keep #2 Huber/MAE outlier-drag, #3 cross-entropy decision boundary, #4 polynomial overfitting, #5 convex-vs-non-convex; add #6 Newton vs GD, #7 basis-function explorer | #1 is duplicated; #6 and #7 cover the new content. |
| Materials examples | **Cut all three** (rare-defect cost, smooth property, sparse feature) | Mathematical Foundations course — materials grounding lives in ML-PC. Keep only the Ai4MatLectures notebook pointer. |

## 4. Section-by-section design

Total: 90 minutes, **49 slides** (47 content + 2 checkpoint mini-quiz slides), 8 sections. Section budgets below include the checkpoints in their respective sections.

### §1. Setup & framing — 5 min, 4 slides

| # | Slide | Notes |
|---|---|---|
| 1 | Title | Standard MFML title-slide partial. |
| 2 | Where we are in the triad | Recap unit 02 (LA, normal equations, Ridge in matrix form, kernel hint); preview that unit 06/07/08 will go deeper on optimization, bias-variance, probability. |
| 3 | Learning outcomes (Bloom-staged) | Recall ERM; Apply GD/SGD/Newton; Derive Newton update from 2nd-order Taylor; Identify the right loss for a given noise model; Analyze why GLMs unify regression & classification. |
| 4 | The supervised learning framework | $f_\mathbf{w}(\mathbf x)$, dataset $\mathcal D$, population vs empirical risk, ERM. |

### §2. Optimization: from first to second order — 20 min, 12 slides (incl. 1 checkpoint)

This is the user's stated "first 20 min" and the largest new content block.

| # | Slide | Notes |
|---|---|---|
| 5 | Optimization landscape | Convex vs non-convex (1 slide, brief). Reuse current interactive #5. |
| 6 | Gradient descent | Update rule $\mathbf w_{t+1} = \mathbf w_t - \eta \nabla f$; 1st-order Taylor justification; $\eta$ tradeoff. |
| 7 | Stochastic gradient descent | Stochastic gradient is unbiased estimator of full gradient. |
| 8 | Minibatch SGD | Variance reduction + GPU vectorization; the workhorse. |
| 9 | Forward-pointer to unit 06 | "Momentum, Nesterov, RMSProp, Adam, conditioning, saddle points → unit 06." 1 compact slide, no derivations. |
| 10 | The Hessian: when 1st-order is slow | Ill-conditioned ravines; condition number $\kappa$ sets convergence rate. **Motivates** Newton via curvature; **does not** redo the full conditioning treatment (that's unit 06). Single-slide framing only. Anchor for the next slide. |
| 11 | Newton's method | 2nd-order Taylor → $\mathbf w \leftarrow \mathbf w - \mathbf H^{-1} \nabla f$. Single-step convergence on quadratics (this is the key intuition). |
| 12 | **Interactive #6: Newton vs GD** | 2D ill-conditioned bowl ($f = ax^2 + by^2$, $a \gg b$); toggle GD/Newton step. Newton converges in one step from any start. |
| 13 | The catch | $\mathcal O(D^2)$ memory, $\mathcal O(D^3)$ inversion; not viable for deep nets. |
| 14 | Quasi-Newton (BFGS / L-BFGS) | Approximate $\mathbf H^{-1}$ from gradient differences. Classical workhorse for medium-dim convex problems. |
| 15 | IRLS preview | "Newton on a GLM has a closed-form per-iteration solve — return to this in §7." Plants the seed for the §7 payoff. |
| 16 | **Checkpoint #1: Optimization** | 3-question mini-quiz (multiple choice). Sample: (a) when does GD diverge? (b) why does Newton converge in one step on a quadratic? (c) why is Newton not used to train deep nets? |

### §3. Loss functions for regression — 12 min, 6 slides

| # | Slide | Notes |
|---|---|---|
| 17 | Loss as decision proxy | Different tasks, different penalties; outliers, asymmetry, cost. |
| 18 | MSE | Quadratic geometry; one-line "= Gaussian MLE under iid Gaussian noise" with forward-pointers to ML-PC §26 and unit 08. |
| 19 | MAE | Linear penalty, $\ell_1$ robustness, Laplacian MLE; sub-gradient at zero. |
| 20 | Huber | Quadratic core, linear tails. |
| 21 | **Interactive #2: Drag-the-outlier** | Existing interactive (lines 271–430 of current file) — keep largely verbatim. |
| 22 | Beyond Gaussian | 1-slide pointer: heteroscedastic / Poisson NLL / robust regression → ML-PC §27. |

### §4. Loss functions for classification — 8 min, 4 slides

| # | Slide | Notes |
|---|---|---|
| 23 | The 0–1 loss problem | Non-differentiable → need surrogate. |
| 24 | Cross-entropy | = Bernoulli/Categorical NLL (one-line; forward to unit 08 for MLE framework). |
| 25 | **Interactive #3: Cross-entropy & decision boundary** | Existing interactive (lines 448–590) — keep verbatim. |
| 26 | Margin-based view | Hinge / SVM in 1 slide; brief mention of calibration / proper scoring. |

### §5. Expanding linear models — basis functions — 22 min, 12 slides (incl. 1 checkpoint)

This is the second-largest new content block.

| # | Slide | Notes |
|---|---|---|
| 27 | The linearity principle | Linear *in parameters*, not in inputs. $f_{\mathbf w}(\mathbf x) = \mathbf w^T \boldsymbol\phi(\mathbf x)$. |
| 28 | Formal basis function expansion | Notation $\boldsymbol\phi: \mathbb R^d \to \mathbb R^M$; OLS in feature space — same normal equations as unit 02, applied to $\boldsymbol\phi$. |
| 29 | Polynomial basis | $\boldsymbol\phi(x) = (1, x, x^2, \ldots, x^M)^T$; design matrix is the Vandermonde. |
| 30 | Runge's phenomenon | Why high-degree global polynomials oscillate at boundaries; static figure showing degree-15 fit on $1/(1+25x^2)$. |
| 31 | **Interactive #4: Polynomial fitting & overfitting** | Existing interactive (lines 756–1003); keep but **relabel** so it reads as a Runge demo and a basis-function demo, not just an "overfitting" demo. |
| 32 | Radial basis functions | $\phi_k(x) = \exp(-\|x - \boldsymbol\mu_k\|^2/2\sigma^2)$; local support; how center placement matters. |
| 33 | Splines | Piecewise polynomials with continuity at knots; local control fixes Runge. Brief mention of B-spline basis. |
| 34 | **Interactive #7: Basis function explorer** | Radio button {polynomial / RBF / B-spline}, slider for # basis functions / knots, on noisy sin-wave dataset. Side panel shows the basis functions themselves. |
| 35 | Bias-variance picture | 1-slide diagram only (U-shape vs complexity), forward-pointer to unit 07 for full decomposition. |
| 36 | Connection to kernels | 1-line: "if $M$ is huge, use the kernel trick — see unit 02 §kernel hint." |
| 37 | Bridge: complex models need constraint | Motivates the regularization bridge in §6. |
| 38 | **Checkpoint #2: Basis functions** | 3-question mini-quiz. Sample: (a) is RBF regression linear or nonlinear, and in what sense? (b) why does Runge's phenomenon happen? (c) what stays the same and what changes when we go from polynomial to RBF basis? |

### §6. Regularization bridge — 5 min, 3 slides

| # | Slide | Notes |
|---|---|---|
| 39 | Recap from unit 02 | Ridge closed form $(\mathbf X^T\mathbf X + \lambda \mathbf I)^{-1}\mathbf X^T \mathbf y$ + L1/L2 constraint geometry. **Do not rederive.** |
| 40 | The MAP interpretation | Gaussian prior on $\mathbf w$ → L2; Laplace prior → L1. Every regularizer is a prior; every loss is an NLL. Forward to ML-PC §28 table; unit 08 for full Bayesian. |
| 41 | What lives elsewhere | Explicit list: dropout / batch-norm / label smoothing / focal / early stopping → deep-learning units; full bias-variance → unit 07; full Bayes → unit 08. (This sets honest expectations.) |

### §7. Unification — Exponential Family & GLMs — 13 min, 5 slides

The mathematical-foundations payoff. Closes the loop with §2.

| # | Slide | Notes |
|---|---|---|
| 42 | Exponential family canonical form | $p(y\mid\eta) = h(y)\exp(\eta^T T(y) - A(\eta))$. Define natural parameter $\eta$, sufficient statistic $T(y)$, log-partition $A(\eta)$, base measure $h(y)$. |
| 43 | Examples in canonical form | Gaussian (with known $\sigma^2$), Bernoulli, Poisson — derive $\eta$, $T(y)$, $A(\eta)$ for each. |
| 44 | Link function | $g(\mu) = \eta$; canonical link; mean-parameter relation $\mu = A'(\eta)$; variance $\mathrm{Var}(y) = A''(\eta)$ (the cumulant connection). |
| 45 | The unification table | Rows: {Gaussian / Bernoulli / Poisson}. Columns: {distribution, $T$ and $A$, canonical link, $\mu(\eta)$, recovered loss}. Recovers MSE / cross-entropy / Poisson NLL as three rows of one framework. Forward to ML-PC §26-28. |
| 46 | IRLS = Newton on the GLM log-likelihood | Show that the Newton step on the GLM NLL has the form $\mathbf w \leftarrow (\mathbf X^T \mathbf W \mathbf X)^{-1} \mathbf X^T \mathbf W \mathbf z$ for a working response $\mathbf z$ and weight matrix $\mathbf W$ — i.e., a weighted least-squares problem. **This closes the loop with §2's Newton material.** |

### §8. Wrap-up — 5 min, 3 slides

| # | Slide | Notes |
|---|---|---|
| 47 | Summary table | Loss → noise assumption → optimizer cheat-sheet for the lecture. |
| 48 | Forward links | Unit 06 (deep optimization), unit 07 (bias-variance), unit 08 (full probabilistic), ML-PC §26-28 (physical noise → loss). |
| 49 | Notebook companion | Ai4MatLectures Week 3 link. |

**Slide count audit:** 4 (§1) + 12 (§2 incl. ckpt #1) + 6 (§3) + 4 (§4) + 12 (§5 incl. ckpt #2) + 3 (§6) + 5 (§7) + 3 (§8) = **49 slides**.

### Pedagogical structure (best-practice teaching choices)

- **Bloom-staged learning outcomes** with explicit verbs (recall / apply / derive / analyze).
- **Forward and backward references on every section header** so students know where each topic is anchored in the curriculum and where it goes deeper.
- **One big idea per slide.** The current unit has crammed slides (e.g., 3+ ideas on the regularization-motivation slide); split where needed.
- **Interactives at every conceptual peak**: Newton (§2), Huber (§3), CE (§4), Runge (§5), basis explorer (§5).
- **Two checkpoint / mini-quiz slides** — slide 16 (after §2 optimization) and slide 38 (after §5 basis functions) — to flag misconceptions before pressing on. Counted inside the 49-slide budget.
- **Visible scaffolding**: each new section opens with "what we're going to do" and closes with "what we just did" framing.
- **Speaker notes**: keep the rich speaker-notes pattern from unit 02 — derivations, alternatives, common-misconception flags, "for the instructor" expansions.

## 5. New interactives — design specs

### Interactive #6: Newton vs GD on an ill-conditioned 2D bowl

- **Function:** $f(x, y) = a x^2 + b y^2$ with $a \gg b$ (default $a = 10$, $b = 1$).
- **Controls:**
  - Slider: $a$ (1 to 20) — adjusts conditioning.
  - Slider: $\eta$ (0.01 to 0.5) — GD learning rate.
  - Slider: # steps (1 to 30).
  - Radio button: {Gradient descent / Newton's method / Both side-by-side}.
  - Reset / re-run button.
- **Output:** Contour plot of $f$ overlaid with iterate trajectory. Newton converges to the origin in 1 step from any start; GD oscillates along the steep $y$-direction at high $\eta$.
- **Implementation:** ojs / d3 in the same style as the existing interactives. Reuse contour-rendering helpers from existing interactive #1 (which we're cutting from the file but the helpers are reusable).
- **Pedagogical hook:** 30-second visual demonstration that "second-order" isn't an abstraction — it's the right thing whenever you can afford it.

### Interactive #7: Basis function explorer

- **Dataset:** Noisy $\sin(2\pi x)$ on $x \in [0, 1]$, $N = 30$ points, $\sigma = 0.15$. Fixed seed.
- **Controls:**
  - Radio button: {Polynomial / Gaussian RBF / B-spline}.
  - Slider: # basis functions (1 to 30).
  - Slider: bandwidth $\sigma$ (RBF only, 0.05 to 0.5) — disabled for other bases.
  - Toggle: show fit / show basis functions.
- **Output:** Two stacked panels.
  - Top: data + fitted curve.
  - Bottom: the individual basis functions $\phi_k(x)$ on $[0, 1]$.
- **Implementation:** OLS solve using a 30×M design matrix; pure JS linear algebra (no ml-matrix dependency to keep the file self-contained, or use the existing approach in the polynomial-fitting interactive). Re-uses much of the polynomial-fitting interactive's framework.
- **Pedagogical hook:** Three radically different model classes share *one* normal equation. Linearity is in the parameters.

## 6. What gets cut from the current file

Listed by section name (line numbers approximate; intent is unambiguous from headings). Total cut: ~600 lines / ~30 slides.

- All deep-learning regularization grab-bag: `Early stopping`, `Data augmentation as regularizer`, `Dropout intuition`, `Batch norm side effects`, `Label smoothing`, `Noise injection methods`.
- Three materials-example slides: `Materials example: rare defect cost`, `Materials example: smooth property regression`, `Materials example: sparse feature model`.
- Empty / skeleton `Exercise N scaffold` slides.
- `Hyperparameter tuning`, `Validation-driven model selection`.
- `Class imbalance handling`, `Multi-label loss setup`, `Structured losses (light)` — replaced by 1-line forward-pointers in §4.
- `Double descent hint`, `Robust optimization preview`, `Loss landscape intuition`, `Optimization interaction` — already deeper coverage in unit 06.
- `Calibration and loss choice`, `Domain-specific loss design` — pruned (1-line in §4).
- Existing interactive #1 ("Interactive: Learning Rate Dynamics", lines ~117–173 of current file) — duplicate of unit 06's "LR sensitivity demo".
- Multi-slide bias-variance framing (`Bias-variance framing`, `Underfit vs overfit diagnostics`) — pruned to a single forward-pointer slide (§5 slide 35).
- Common pitfalls / checklist / exam-statements / unit summary slabs collapsed into one summary table (§8 slide 47).

## 7. What gets kept verbatim or near-verbatim

- Existing interactive #2 (drag-the-outlier MSE/MAE/Huber, lines 271–430).
- Existing interactive #3 (cross-entropy decision boundary, lines 448–590).
- Existing interactive #4 (polynomial fitting & overfitting, lines 756–1003) — kept but **relabeled** as a Runge / basis-function demo, not just "overfitting".
- Existing interactive #5 (convex vs non-convex 2D plot, lines 82–103).
- The MSE / MAE / Huber text content (lines 245–270).
- The hinge-loss / margin slide (lines 592–598).
- The bibliography file `ref.bib`.

## 8. What gets newly built

- §2 Newton / Hessian / quasi-Newton derivations (~6 new slides).
- Interactive #6 (Newton vs GD).
- §5 formal basis functions + Runge + RBFs + splines (~8 new slides).
- Interactive #7 (basis function explorer).
- §7 exponential family + GLM + IRLS (~5 new slides).
- Two new checkpoint / mini-quiz slides.

## 9. File-level layout

Single file rewrite of `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`. The YAML header pattern stays as it is in the current file (1920×1080, custom theme, footer, etc.). All other unit-folder files (`_metadata.yml`, `custom.scss`, `custom.css`, `eclipse_logo_small.png`, `ref.bib`, `title-slide.html`, `title-slide.scss`) remain untouched.

Final estimated file length: ~700–800 lines (down from 1148), driven primarily by removing the deep-learning regularization grab-bag and consolidating the cut sections into 1-line forward-pointers.

## 10. Risks and tradeoffs

- **Slide count drift.** Adding rich speaker notes for the new content (Newton, GLM) may push file length up. Acceptable — speaker notes do not affect slide count or render time.
- **Interactive build cost.** Interactives #6 and #7 are non-trivial JS builds, ~150 lines of ojs each. Both are achievable using patterns from the existing interactives. We don't want them so polished that they swallow other work; functional > polished is fine for week 3.
- **Curricular consistency.** The unit relies on cross-unit references (unit 02, 06, 07, 08, ML-PC §26–28). If any of those units gets renumbered or rewritten before the redesign ships, the forward-pointers will break. **Mitigation:** use slug names (`unit06_loss_landscapes`) plus section anchors when available; the README slug map is authoritative.
- **GLM section difficulty.** Exponential-family canonical form is the highest-mathematical-density slide block in the unit. Risk that students at week 3 aren't ready. **Mitigation:** sequence is supportive — by §7, students have already seen Gaussian MLE (one-liner), Bernoulli/CE, and Newton's method, so the unification is a synthesis rather than a new derivation. The unification table (slide 43) is the load-bearing slide; the others scaffold it.
- **No notebook-side deliverable in this spec.** Only the slide deck is in scope; companion notebooks live in the separate `ECLIPSE-Lab/Ai4MatLectures` repo and are not modified by this redesign.

## 11. Out of scope

- Modifications to other units (02, 06, 07, 08, ML-PC).
- Modifications to companion notebooks in `Ai4MatLectures`.
- Render / deploy changes (handled by existing CI).
- Index-page (`index.qmd`) regeneration — the unit folder slug is unchanged so no index update is needed.

## 12. Acceptance criteria

- The redesigned `01_intro.qmd` renders cleanly with `QUARTO_PYTHON=.venv/bin/python quarto preview ... --no-browser --no-watch-inputs` from the unit folder.
- All 49 designated slides are present in the order specified in §4.
- All five kept interactives still render and respond to inputs.
- The two new interactives (#6 Newton vs GD, #7 basis-function explorer) render and respond to all controls listed in §5.
- Forward / backward cross-references named in §4 are present in the slide text.
- Speaker notes accompany every section's first slide and every new derivation slide.
- The current `01_intro.qmd` is replaced — no parallel old/new files left behind.
- `missing_topics_analysis.md` either gets updated to reflect what unit 03 now covers, or is removed (unit 03 redesign supersedes it).
