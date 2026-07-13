# Overview-figure prompts — Algorithm Compendium (MFML Appendix)

One prompt per algorithm slide in `01_intro.qmd`. Each prompt is self-contained
(content description + full slide style) so it can be pasted directly into an
image-generation model. Save the generated image to the **Target file** path —
the deck already contains a dummy slide per figure that loads that exact file
from `images/`.

**Shared slide style** (already embedded in every prompt below):

> 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark
> navy background (vertical gradient #0b0f19 → #111827) with faint blue
> (#3b82f6) and violet (#8b5cf6) radial glows in the upper corners. Accent
> palette: bright blue #3b82f6/#60a5fa, violet #8b5cf6/#c4b5fd, amber #f59e0b
> for highlights, light blue-grey text #dbe4f0, white labels #f8fafc. Rounded
> cards (~16 px corner radius) with translucent dark fills and thin 1 px white
> borders at 8% opacity, soft drop shadows. Clean geometric sans-serif labels
> (Inter/Outfit style), minimal text, thin crisp arrows, generous negative
> space, no photorealism, no clutter.

---

## 1. Data splitting & k-fold cross-validation — Unit 1

**Target file:** `images/fig_01_data_splitting_kfold_cv.jpg`

```text
Flat vector infographic explaining data splitting and k-fold cross-validation, two stacked panels. Top panel: a long horizontal bar representing a dataset, divided into three segments labeled "train ~70%" (blue #3b82f6), "val ~15%" (violet #8b5cf6), and "test ~15%" (amber #f59e0b); a small padlock icon on the test segment with the caption "touched once, at the end". Bottom panel: a 5×5 grid of identical horizontal bars illustrating 5-fold CV; in each row a different single fold is highlighted amber ("held out") while the other four folds are blue ("train"), with the highlighted fold moving one position per row like a diagonal staircase; to the right an arrow collects all five held-out scores into a card reading "average → CV error". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders at 8% opacity and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 2. Ordinary least squares (+ ridge) — Unit 2

**Target file:** `images/fig_02_ols_ridge.jpg`

```text
Flat vector infographic explaining least-squares regression as orthogonal projection, two side-by-side panels. Left panel: a 2D scatter of light-blue data points with a bright blue best-fit line and thin vertical amber residual segments connecting each point to the line, labeled "residuals". Right panel: a 3D-looking geometric diagram of a tilted plane labeled "column space of X", a vector y as an arrow from the origin pointing above the plane, its orthogonal projection ŷ = Xŵ drawn on the plane, and a dashed perpendicular residual vector y − Xŵ meeting the plane at a small right-angle marker; caption "residual ⟂ column space". A small footer card contrasts "OLS: XᵀX ŵ = Xᵀy" with "Ridge: (XᵀX + λI) ŵ = Xᵀy — cures ill-conditioning". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 3. PCA via SVD — Unit 2

**Target file:** `images/fig_03_pca_svd.jpg`

```text
Flat vector infographic explaining PCA via the SVD, three panels left to right. Left panel: an elongated tilted 2D point cloud of light-blue dots with its mean marked as a white cross, and two orthogonal arrows from the mean — a long bright blue arrow labeled "PC 1 (max variance)" along the cloud's long axis and a short violet arrow labeled "PC 2" perpendicular to it. Middle panel: the same cloud projected onto PC 1, shown as dots collapsed onto a single blue line, labeled "scores Z = X_c V_k". Right panel: a small bar chart of decreasing bar heights labeled "explained variance σᵢ²", with the first bar dominant, plus a compact card reading "SVD: X_c = U Σ Vᵀ — center first!". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 4. Gradient descent, SGD, minibatch — Unit 3

**Target file:** `images/fig_04_gd_sgd_minibatch.jpg`

```text
Flat vector infographic comparing gradient descent variants on a loss landscape. A large panel shows smooth concentric elliptical contour lines of a loss function in dark blue-grey with a glowing minimum marker at the center. Three optimization trajectories start from the same point at the upper left: a smooth bright blue path with few large confident steps labeled "GD — exact gradient", a very jittery zig-zag violet path labeled "SGD — one sample, noisy but cheap", and an amber path of intermediate smoothness labeled "minibatch — variance ↓ ∝ 1/b". A small side card shows the update rule "w ← w − η·g" and a tiny legend of the three variants. Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 5. Newton's method — Unit 3

**Target file:** `images/fig_05_newton_method.jpg`

```text
Flat vector infographic explaining Newton's method versus gradient descent, two side-by-side panels. Left panel: a smooth 1D loss curve in light blue with a point on its slope; a dashed violet parabola hugging the curve at that point labeled "local quadratic model"; a single long amber arrow jumping from the point directly to the parabola's minimum, labeled "one Newton step: solve HΔ = g". Right panel: the same curve with many small blue arrows creeping down the slope labeled "gradient descent: many small steps". A footer card contrasts "quadratic convergence near optimum" against "Hessian solve is O(D³) — hence L-BFGS for small smooth problems". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 6. Logistic regression training — Unit 3

**Target file:** `images/fig_06_logistic_regression.jpg`

```text
Flat vector infographic explaining logistic regression, three panels. Left panel: a 2D scatter with two point classes (blue circles, violet triangles) separated by a straight amber decision boundary line, labeled "linear boundary wᵀx = 0". Middle panel: a smooth S-shaped sigmoid curve σ(z) = 1/(1+e⁻ᶻ) rising from 0 to 1, axis labeled z = wᵀx, with the 0.5 midpoint marked. Right panel: a compact flow diagram of cards connected by arrows: "z = wᵀx" → "p = σ(z)" → "cross-entropy loss" → a highlighted amber card "∂L/∂z = p − y" → "w ← w − η(p−y)x", captioned "gradient = error × input". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 7. MLP forward pass — Unit 4

**Target file:** `images/fig_07_mlp_forward.jpg`

```text
Flat vector infographic of a multilayer perceptron forward pass. A left-to-right network diagram: an input column of 4 light-blue nodes labeled "x", two hidden layers of 5 nodes each in blue with thin translucent connection lines between all consecutive nodes, and an output pair of nodes labeled "ŷ". Above each layer transition a small rounded card labels the operation: "z⁽ℓ⁾ = W⁽ℓ⁾a⁽ℓ⁻¹⁾ + b⁽ℓ⁾ (affine)" followed by a violet card "a⁽ℓ⁾ = σ(z⁽ℓ⁾) (nonlinearity)"; the final transition card reads "last layer linear / softmax". A subtle amber ribbon arrow flows left to right underneath the whole network labeled "forward — cache z, a for backprop". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 8. Backpropagation — Unit 4

**Target file:** `images/fig_08_backprop.jpg`

```text
Flat vector infographic of backpropagation through a 3-layer neural network. The same left-to-right node diagram as a forward pass (light-blue nodes, thin connections), but overlaid with a bold right-to-left amber ribbon arrow labeled "backward sweep — one pass, cost O(W)". At each layer, a violet badge labeled δ⁽ℓ⁾ sits on the nodes, with small arrows branching off downward to two cards per layer: "∂L/∂W⁽ℓ⁾ = δ⁽ℓ⁾(a⁽ℓ⁻¹⁾)ᵀ" and "∂L/∂b⁽ℓ⁾ = δ⁽ℓ⁾". Between layers a card shows the recursion "δ⁽ℓ⁻¹⁾ = (W⁽ℓ⁾ᵀδ⁽ℓ⁾) ⊙ σ′(z⁽ℓ⁻¹⁾)" with the ⊙ σ′ factor highlighted in amber. At the far right the loss node L(ŷ, y) glows as the starting point of the backward arrow. Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 9. 2D convolution (multi-channel) — Unit 4

**Target file:** `images/fig_09_conv2d.jpg`

```text
Flat vector infographic of multi-channel 2D convolution. Left: an input feature map drawn as three stacked translucent grids (channels C_in, slightly offset in depth, blue tones) with a thin dashed zero-padding border. Center: a small 3×3 kernel stack in violet hovering over a highlighted 3×3 input region, with faint ghost copies suggesting it slides across the image, and an arrow labeled "Σ over channels + window, + bias". Right: the resulting single output channel as an amber-tinted grid, with several output grids stacked behind it labeled "C_out feature maps". A footer card notes "params = C_out·(C_in·k_h·k_w + 1) — independent of image size" and a small side badge "1×1 conv = per-pixel MLP across channels". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 10. k-means (Lloyd + k-means++) — Unit 5

**Target file:** `images/fig_10_kmeans.jpg`

```text
Flat vector infographic of the k-means algorithm as a two-step loop. Center: a 2D scatter with three point clusters colored blue, violet and amber, each with a large star-shaped centroid marker; thin dashed lines connect a few points to their nearest centroid. Around the scatter, a circular loop of two arrows connecting two rounded cards: "ASSIGN — each point → nearest centroid" and "UPDATE — centroid → cluster mean", with a small loop caption "repeat until assignments freeze; each step lowers within-cluster SSE J". Left inset: k-means++ seeding shown as one chosen centroid and faraway points glowing brighter, captioned "seed next centroid ∝ D(x)²". Bottom-right inset: a tiny elbow curve of J versus K with the elbow circled. Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 11. EM for Gaussian mixtures — Unit 5

**Target file:** `images/fig_11_em_gmm.jpg`

```text
Flat vector infographic of expectation–maximization for a Gaussian mixture model. Center: a 2D scatter with two overlapping elliptical Gaussian components drawn as nested translucent contour ellipses, one blue and one violet; points in the overlap region are colored as a blue-to-violet gradient to show soft responsibilities γᵢₖ, with a callout "soft assignment — every point belongs fractionally to every component". A circular two-arrow loop connects two cards: "E-step — compute responsibilities γᵢₖ" and "M-step — weighted means μ, covariances Σ, weights π", with the loop captioned "log-likelihood never decreases". A small amber warning badge in a corner shows a tiny ellipse collapsing onto a single point, captioned "singular collapse — add covariance ridge". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 12. Autoencoder + anomaly detection — Unit 5

**Target file:** `images/fig_12_autoencoder.jpg`

```text
Flat vector infographic of an autoencoder with anomaly detection. Main diagram: an hourglass shape made of stacked trapezoids — a blue encoder funnel narrowing left-to-right into a slim violet bottleneck bar labeled "z, k ≪ D", then a blue decoder funnel widening back out; input card "x" on the left, output card "x̂" on the right, with a curved amber measuring bracket between them labeled "reconstruction loss ‖x − x̂‖²". Bottom strip: an anomaly-detection inset — a histogram of reconstruction errors with a vertical amber threshold line τ; points left of it green-tinted labeled "normal", a few far-right outlier bars glowing amber labeled "anomaly: error > τ", captioned "train on normal data only". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 13. Momentum & Nesterov — Unit 6

**Target file:** `images/fig_13_momentum_nesterov.jpg`

```text
Flat vector infographic comparing plain gradient descent with momentum on a ravine-shaped loss. Main panel: strongly elongated elliptical contour lines forming a narrow valley; a violet trajectory zig-zags wildly across the narrow direction labeled "GD — oscillates across the ravine", while a bright blue trajectory with a fading motion trail glides smoothly along the valley floor labeled "momentum — oscillations cancel, progress accumulates". Right inset card: Nesterov look-ahead geometry — from a point θ, a grey dashed arrow αv to a ghost look-ahead point, an amber gradient arrow evaluated at the ghost point, and the resulting corrected blue step, captioned "look ahead, then correct". Footer formula card: "v ← αv − ηg;  θ ← θ + v". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 14. AdaGrad + training stabilizers — Unit 6

**Target file:** `images/fig_14_adagrad_stabilizers.jpg`

```text
Flat vector infographic of adaptive learning rates and training stabilizers, three panels. Left panel: AdaGrad — a row of parameter icons with differently sized step arrows: parameters with tall stacked gradient-history bars get short arrows, parameters with tiny history bars get long arrows, captioned "rare features keep large steps — θ ← θ − η·g/(√G + ε)"; a small downward-decaying curve badge warns "G grows forever → rate → 0 (motivates RMSProp/Adam)". Middle panel: gradient clipping — a long spiky gradient vector being trimmed at a dashed circle of radius c, labeled "if ‖g‖ > c: rescale". Right panel: initialization — a deep stack of thin layers with equal-height variance bars flowing through, labeled "Xavier: 2/(n_in+n_out) · He: 2/n_in — keep variance constant across depth". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 15. MLE and MAP estimation — Unit 7

**Target file:** `images/fig_15_mle_map.jpg`

```text
Flat vector infographic contrasting maximum likelihood and MAP estimation. Main panel: a parameter axis θ with two smooth bell-shaped curves — a blue likelihood curve labeled "likelihood ℓ(θ)" with a dashed vertical line at its peak labeled θ̂_MLE, and a violet prior curve p(θ) centered at zero; their product drawn as an amber posterior curve whose peak θ̂_MAP is visibly pulled from the MLE peak toward the prior, with a small arrow labeled "prior pulls the estimate". Side card: a two-row mapping with icons — "Gaussian prior → ridge / L2" and "Laplace prior (sharp peak) → lasso / L1". Footer caption: "MAP = maximize ℓ(θ) + log p(θ) — every regularizer is a log-prior". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 16. CART: growing a decision tree — Unit 8

**Target file:** `images/fig_16_cart.jpg`

```text
Flat vector infographic of CART decision-tree growing, two linked panels. Left panel: a 2D feature space (axes x₁, x₂) containing blue circles and violet triangles, recursively partitioned by bold axis-aligned split lines into rectangular regions; each region tinted faintly by its majority class; the first split line highlighted amber and labeled "best (feature, threshold) — max impurity drop Δ". Right panel: the corresponding binary tree diagram with rounded rectangular decision nodes ("x₁ < t?") and leaf nodes as colored pills showing majority class / mean value; a curved arrow links each tree node to its matching split line in the feature space. Footer badges: "greedy, O(N d log N)" and "prune: minimize R(T) + α|T|". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 17. Random forest + OOB error — Unit 8

**Target file:** `images/fig_17_random_forest_oob.jpg`

```text
Flat vector infographic of a random forest with out-of-bag error. Top: a single dataset bar splitting via three arrows into three bootstrap-sample bars (some segments duplicated, drawn as repeated tiles; ~37% of tiles greyed out and labeled "OOB — never seen"). Middle: each bootstrap bar feeds a small stylized decision-tree icon in blue; above each tree a violet badge "m random features per split — decorrelate". Bottom: all trees' outputs merge via arrows into one card "average / majority vote". Right inset: an OOB card showing a grey sample routed only to the trees that never saw it, captioned "free validation ≈ k-fold CV". Footer formula badge: "Var = ρσ² + (1−ρ)σ²/B — feature sampling shrinks ρ". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 18. Gradient boosting — Unit 8

**Target file:** `images/fig_18_gradient_boosting.jpg`

```text
Flat vector infographic of gradient boosting as sequential residual fitting. A left-to-right chain of four stages connected by arrows: stage 0 is a flat horizontal line over a scatter labeled "best constant f⁽⁰⁾"; each following stage shows the same scatter with the current fit curve in blue hugging the data progressively better, and beneath it a small residual plot with shrinking violet residual bars labeled "pseudo-residuals rᵢ = −∂ℒ/∂f̂"; a tiny tree icon between stages with an amber "× η (shrinkage)" badge feeds each improvement, labeled "fit small tree to residuals, add η·h_t". A footer card contrasts "boosting: sequential BIAS reduction" with "bagging: parallel VARIANCE reduction". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 19. Permutation importance — Unit 8 / 14

**Target file:** `images/fig_19_permutation_importance.jpg`

```text
Flat vector infographic of permutation feature importance, three panels left to right. Left panel: a small data table (rows × feature columns) on a held-out set with one column highlighted violet and shown being shuffled — cell values scrambled with small curved swap arrows, captioned "shuffle ONE feature column — destroys its link to the target". Middle panel: a gauge-like score card showing baseline score s₀ dropping to a lower shuffled score, with a bold amber downward arrow labeled "score drop = importance". Right panel: a horizontal bar chart ranking features by importance, bars in descending blue gradient, the top bar highlighted; small badge "model-agnostic, held-out — prefer over MDI". Footer warning strip: "correlated twins share importance; association ≠ causation". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 20. t-SNE — Unit 9

**Target file:** `images/fig_20_tsne.jpg`

```text
Flat vector infographic of t-SNE, two main panels bridged by an arrow. Left panel: a 3D-looking cloud of points in high-dimensional space with one point highlighted and a soft Gaussian halo around it fading with distance, labeled "high-dim: Gaussian neighborhoods p_ij, σᵢ set by perplexity". Right panel: a clean 2D embedding with three well-separated island clusters of colored points (blue, violet, amber), labeled "low-dim: Student-t similarities q_ij". Bridge arrow labeled "minimize KL(P‖Q) by gradient descent". Bottom inset: two overlaid bell curves — a narrow Gaussian and a heavy-tailed t-distribution with its fat tails shaded amber, captioned "heavy tails solve the crowding problem". Footer warning strip: "cluster sizes & distances are MEANINGLESS — never quantify from t-SNE". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 21. UMAP — Unit 9

**Target file:** `images/fig_21_umap.jpg`

```text
Flat vector infographic of the UMAP pipeline as four stages connected by arrows, left to right. Stage 1: a scatter of points with thin blue edges linking each point to its nearest neighbors, labeled "k-NN graph, density-adaptive σᵢ". Stage 2: the same graph with edges of varying opacity and a small formula badge "fuzzy union: w = w₁ + w₂ − w₁w₂", labeled "fuzzy symmetrization". Stage 3: points arranged in a smooth coarse layout with a small waveform/eigenvector icon, labeled "spectral initialization (graph Laplacian)". Stage 4: a crisp final 2D embedding with compact colored clusters, labeled "SGD on graph cross-entropy". Footer strip: two knob icons labeled "n_neighbors → local ↔ global" and "min_dist → visual packing only". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 22. SimCLR + linear probe — Unit 9

**Target file:** `images/fig_22_simclr.jpg`

```text
Flat vector infographic of SimCLR contrastive learning. Left: one input image tile splits via two arrows into two augmented views (one crop-tinted, one flipped/blurred), labeled "two augmentations of the same x". Both views flow through identical stacked encoder blocks "f" then small projector blocks "g" into embedding vectors. Right: a large circle representing the unit hypersphere with embedded points on its rim: the two positive views drawn as glowing blue dots pulled together by a curved attracting arrow labeled "attract (same image)", while grey dots around the rim are pushed away by short outward arrows labeled "repel — all 2N−2 others are negatives (NT-Xent, temperature τ)". Bottom strip: linear-probe evaluation — a frozen encoder card with a snowflake icon feeding a single small linear layer, captioned "freeze f, train logistic regression — probe what the representation knows". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 23. Scaled dot-product attention — Unit 10

**Target file:** `images/fig_23_attention.jpg`

```text
Flat vector infographic of scaled dot-product attention as a matrix-flow diagram, left to right. Start: a tall input matrix X (n tokens × d_model) branching via three arrows into three slim matrices tinted blue, violet and amber labeled Q, K, V ("learned projections W^Q, W^K, W^V"). Q and K meet at a card "S = QKᵀ/√d_k" producing an n×n score grid; the √d_k factor highlighted with a small thermometer badge captioned "keeps variance ≈ 1, prevents softmax saturation". Next: a softmax card turning the grid into a row-normalized attention heatmap (each row's cells summing to 1, drawn as a blue-violet heat grid with one bright cell per row); an optional triangular grey mask overlay in a corner labeled "causal mask: −∞ above diagonal". Finally the heatmap multiplies V into the output matrix Z labeled "weighted sum of values". Boxed footer formula: "Attention(Q,K,V) = softmax(QKᵀ/√d_k)V". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 24. Multi-head attention — Unit 10

**Target file:** `images/fig_24_multihead_attention.jpg`

```text
Flat vector infographic of multi-head attention. Left: an input matrix X splits into 8 parallel horizontal lanes, each lane a slim attention-block card in a different hue of the blue–violet range labeled "head 1 … head 8, each with own W^Q, W^K, W^V in a d_k = d_model/h subspace"; small icons above three lanes hint at specialization (a position ruler, a syntax bracket pair, a link chain). Center: the lane outputs stack into one tall striped matrix labeled "Concat". Right: a final projection card "W^O — output mixer" restoring a matrix with the caption "same shape as input → blocks stack". Footer badges: "typical: d_model 512, h = 8, d_k = 64" and an amber note "more heads ≠ free capacity: d_k shrinks". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 25. Transformer block (+ positional encoding) — Unit 10

**Target file:** `images/fig_25_transformer_block.jpg`

```text
Flat vector infographic of one pre-norm transformer block, drawn as a vertical stack. Bottom: token embedding tiles plus a wavy overlay of interleaved sine/cosine curves entering via a ⊕ symbol, labeled "positional encoding — attention is permutation-equivariant, inject order". The block above: card "LayerNorm" → card "Multi-Head Attention — mix ACROSS positions", with a curved residual skip arrow bypassing both into a ⊕; then card "LayerNorm" → card "MLP — transform EACH position", with a second residual skip arrow into another ⊕. The residual skip paths drawn as bold amber curves labeled "residuals keep gradients flowing". Right side: a faded stack of identical block outlines receding upward labeled "× L (6…96)". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 26. Vision Transformer forward pass — Unit 10

**Target file:** `images/fig_26_vit.jpg`

```text
Flat vector infographic of a Vision Transformer forward pass, left to right. Left: a simple square image (stylized micrograph-like texture) sliced by thin lines into a 4×4 grid of patches that explode outward into a row of separate tiles, labeled "224² image → 196 patches of 16²". The tiles flow through a card "linear projection to d_model", a special violet tile labeled "[CLS]" is prepended, and small position-number badges attach beneath each tile ("+ positional embeddings"). Center: a stack of transformer-block cards labeled "L blocks, bidirectional attention — no mask". Right: the [CLS] tile exits alone into a classifier-head card producing class-probability bars. Bottom inset: the original image with a translucent blue-to-amber heatmap overlay on a 14×14 grid, captioned "free interpretability: [CLS]→patch attention row = importance map". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 27. VAE training step — Unit 11

**Target file:** `images/fig_27_vae.jpg`

```text
Flat vector infographic of a variational autoencoder training step. Left to right: input card "x" → blue encoder trapezoid emitting TWO small vectors, μ (blue) and σ (violet); below them a cloud icon "ε ~ N(0, I)"; the three merge at a highlighted amber card "z = μ + σ ⊙ ε — reparameterization trick" with the caption "randomness moved into ε so gradients flow through sampling"; then z passes through a decoder trapezoid to output "x̂". Two loss cards hang beneath: "L_recon = ‖x − x̂‖²" bracketing x and x̂, and "L_KL — pull q(z|x) toward N(0,I)" drawn as a small latent disc with an arrow toward a unit circle. Right inset: generation mode — dice icon sampling z from N(0,I) straight into the decoder producing a new sample. Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 28. DDPM training — Unit 11

**Target file:** `images/fig_28_ddpm_training.jpg`

```text
Flat vector infographic of DDPM training. Top strip: a row of five image tiles showing a clean sample x₀ progressively dissolving into pure static noise x_T, with an amber curved arrow leaping directly from x₀ to a middle tile x_t labeled "closed-form jump to ANY t: x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε — O(1) per sample". Bottom: the noisy tile x_t and a small clock badge "t" feed into a U-shaped network silhouette labeled "ε_θ (U-Net / DiT)"; its output vector is compared against the true noise ε in a card "L = ‖ε − ε_θ(x_t, t)‖²" with a caption "well-scaled regression target — unit variance at every t". Small side inset: two noise-schedule curves (linear vs cosine) with the cosine curve highlighted, captioned "cosine keeps signal longer". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 29. Diffusion sampling + classifier-free guidance — Unit 11

**Target file:** `images/fig_29_diffusion_sampling_cfg.jpg`

```text
Flat vector infographic of diffusion sampling with classifier-free guidance. Main strip: a right-to-left denoising chain of five tiles from pure static noise x_T to a crisp final sample x₀, connected by stepwise arrows labeled "T reverse steps, each one ε_θ call". Above one intermediate step, a vector-composition inset: two arrows from the same point — a blue arrow "ε_θ(x_t, c) conditional" and a grey arrow "ε_θ(x_t, ∅) unconditional" — combined into a longer amber arrow labeled "CFG: (1+w)·cond − w·uncond", with a small slider labeled "w: fidelity ↑, diversity ↓ (w ≈ 5–10)". Footer badges: "DDIM: deterministic, ~50 steps ≈ 1000 DDPM steps" and "training side: drop c with ~10% prob so ε_θ(·, ∅) exists". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 30. GP regression — Unit 12

**Target file:** `images/fig_30_gp_regression.jpg`

```text
Flat vector infographic of Gaussian-process regression. Main panel: a smooth posterior-mean curve in bright blue passing exactly through five white data-point dots, surrounded by a translucent blue-violet uncertainty band that pinches to near zero at each data point and balloons wide in the gaps and beyond the data range; annotation arrows: "band pinches at data" and "honest epistemic uncertainty grows away from data"; two faint grey sample-function curves wiggle inside the band labeled "posterior samples". Left inset card: an RBF kernel bump k(x,x′) = σ_f²·exp(−‖x−x′‖²/2ℓ²) with a double-headed arrow marking the length-scale ℓ. Footer badges: "predict: μ* = k*ᵀα, closed form", "fit ℓ, σ_f, σ_n by marginal likelihood (L-BFGS)", amber warning "O(N³) Cholesky". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 31. Ensembles & MC dropout — Unit 12

**Target file:** `images/fig_31_ensembles_mc_dropout.jpg`

```text
Flat vector infographic comparing deep ensembles and MC dropout for uncertainty, two side-by-side panels. Left panel "Deep ensemble (best calibrated)": five small identical network icons in a row, each with a different dice badge labeled "seed 1…5", all fed the same input tile; their five output curves overlay on a shared plot fanning apart in a region without data, with the spread shaded violet and labeled "disagreement = epistemic variance". Right panel "MC dropout (cheap)": ONE network icon drawn T times as ghost copies, each ghost with a different random pattern of crossed-out grey neurons, labeled "dropout ACTIVE at test time, T stochastic passes"; outputs again fan into a shaded spread. Footer formula card: "Var[y*] = E[σ²] (aleatoric) + Var[μ] (epistemic)" with an amber note "neither is calibrated by default — audit first". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 32. Calibration audit + temperature scaling — Unit 12

**Target file:** `images/fig_32_calibration_temperature.jpg`

```text
Flat vector infographic of calibration auditing and temperature scaling, two panels. Left panel: a reliability diagram — square plot with "stated confidence" on x and "observed accuracy" on y, a dashed white diagonal labeled "perfectly calibrated", and a violet bar-curve of binned points sagging below the diagonal with the gap shaded amber and labeled "overconfident — typical neural net"; small bin strips along the x-axis (0.5–0.6 … 0.9–1.0). Right panel: temperature scaling — a logits vector card passing through a single amber dial labeled "÷ T (one scalar, fit on validation NLL)" into a softmax card; below, two small probability bar charts "before: overconfident spike" vs "after: softened, calibrated", with the badge "ranking & accuracy unchanged". Footer warning: "never calibrate on the test set; cannot fix OOD overconfidence". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 33. Active learning & Bayesian optimization — Unit 12

**Target file:** `images/fig_33_active_learning_bo.jpg`

```text
Flat vector infographic of the active-learning / Bayesian-optimization loop. Main panel, two stacked aligned plots: top plot shows a GP posterior over an unknown function — blue mean curve, violet uncertainty band, four white observed points, and the best observed value f⁺ marked with a horizontal dashed line; bottom plot shows the Expected Improvement acquisition function as an amber curve peaking in a region that is both promising and uncertain, with a vertical dashed line rising from its peak through both plots to a starred marker labeled "x_next = argmax EI". Around the panel, a circular loop of four small cards connected by arrows: "fit GP" → "maximize acquisition" → "run experiment" → "append data" → back to "fit GP", loop titled "until budget exhausted". Side badges: "active learning: chase max σ(x)" vs "BO: EI balances exploit (high μ) + explore (high σ)". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 34. PINN training loop — Unit 13

**Target file:** `images/fig_34_pinn.jpg`

```text
Flat vector infographic of a physics-informed neural network. Center: a small MLP icon labeled "f_θ(x, t) — smooth activations (tanh)" receiving coordinate inputs. Its output feeds two parallel loss branches drawn as cards: upper blue branch "J_data — MSE at N sparse labeled points", illustrated by a plot domain containing only a handful of white measurement dots; lower violet branch "J_physics — PDE residual ℛ at M ≫ N collocation points", illustrated by the same domain densely filled with tiny violet crosses and a badge "autodiff w.r.t. INPUTS: ∂f/∂t, ∂²f/∂x² — no labels needed, target is zero". The two branches merge at an amber summation card "J = J_data + λ·J_physics" with a small balance-scale icon on λ. Footer badges: "Adam then L-BFGS polish" and warnings "ReLU kills 2nd derivatives; wrong physics → confidently wrong". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 35. Lagaris substitution — Unit 13

**Target file:** `images/fig_35_lagaris.jpg`

```text
Flat vector infographic of the Lagaris trial-solution construction for boundary conditions. Center: a large formula card "f(x) = A(x) + B(x) · NN(x)" with each term color-coded and annotated by arrows: A(x) in blue → "satisfies the BCs exactly — algebra, not training"; B(x) in violet → "vanishes on the boundary"; NN(x) in amber → "learns only the interior". Below: an interval plot from 0 to 1 with boundary posts at both ends pinned by white lock icons ("BC holds for ANY network"), a violet envelope curve x(1−x) rising in the middle and touching zero at both ends, and several faint amber candidate curves all passing exactly through the pinned boundary values. Side card with three recipe rows: "f(0)=a → f = a + x·NN", "f(0)=a, f(L)=b → linear ramp + x(L−x)·NN", "f(0)=f(1)=0 → x(1−x)·NN". Footer note: "no J_BC, no λ₂ — hard on complex geometries, then fall back to soft enforcement". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```

## 36. Perturbation sensitivity analysis — Unit 14

**Target file:** `images/fig_36_sensitivity_analysis.jpg`

```text
Flat vector infographic of one-at-a-time perturbation sensitivity analysis, three panels left to right. Left panel: an input feature vector drawn as a column of labeled slots, with ONE slot highlighted amber and a small "+Δ" nudge arrow on it while all other slots stay frozen with tiny lock icons, captioned "perturb one feature, hold the rest". Middle panel: a model box f with two output readouts side by side — f(x) and f(x+Δe_j) — and a bracket measuring their gap, labeled "S_j = |f(x+Δe_j) − f(x)| / |Δ|". Right panel: a horizontal bar chart of per-feature sensitivities in descending blue bars, with two toggle badges above: "local — why THIS prediction (at one sample)" and "global — average over dataset". Footer warning strip: "one-at-a-time misses interactions; sensitivity = association, NOT causation". Style: 16:9 presentation figure, 1920×1080, flat modern vector infographic on a dark navy background (vertical gradient #0b0f19 → #111827) with faint blue #3b82f6 and violet #8b5cf6 radial glows in the upper corners; accents blue #60a5fa, violet #c4b5fd, amber #f59e0b; light blue-grey text #dbe4f0, white labels #f8fafc; rounded translucent dark cards with thin 1px white borders and soft shadows; clean geometric sans-serif labels, minimal text, thin crisp arrows, generous negative space, no photorealism.
```
