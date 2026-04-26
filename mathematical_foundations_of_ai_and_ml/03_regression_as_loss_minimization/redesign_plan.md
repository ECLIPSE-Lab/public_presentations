# MFML Unit 03 Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd` (currently 1148 lines, ~60 slides) with a 49-slide, ~700-line redesigned lecture per `redesign_spec.md`.

**Architecture:** Single-file rewrite in place. Build the new file section-by-section using a scaffold-and-fill approach: Task 1 backs up the current file (git history is the backup) and writes a skeleton with section-marker comments; subsequent tasks replace each marker with the section's slides. Reusable interactives are extracted from the old file via git and pasted into the new positions. Each task ends in a render verification + commit.

**Tech Stack:** Quarto, RevealJS, ojs (Observable JS), d3.js, mathjax. Python-driven Quarto (`.venv/bin/python`). No notebooks executed during render (`execute.enabled: false`).

**Spec reference:** `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/redesign_spec.md`

---

## Conventions

- **Working directory:** `/home/philipp/projects/_public_presentations/`
- **Target file:** `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`
- **Render verification command (full):** `cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -30`
- **Render verification command (fast):** `cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto check 01_intro.qmd 2>&1 | tail -10`
- **Slide-count check:** `grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd` (target: 49 at the end; section comment markers do not count).
- **Commits:** No `Co-Authored-By: Claude` trailer (per repo convention).
- **Backup of old content:** the current `01_intro.qmd` is checked into `main` at commit `09384b1d` (or whatever HEAD is at Task 1 start). Use `git show HEAD:<path>` to retrieve original content during port tasks.
- **Section markers** in the scaffolded file use HTML comments: `<!-- ===== §N. Section Name ===== -->` so they're invisible in the rendered HTML and easy to find via `grep`.

---

## Task 1: Scaffold the new file with section markers

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

- [ ] **Step 1: Capture the current file's commit ref for later port tasks**

```bash
cd /home/philipp/projects/_public_presentations
git log -1 --format='%H' -- mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
```

Record the SHA — call it `$BASELINE_SHA`. Used in later port tasks via `git show $BASELINE_SHA:...`.

- [ ] **Step 2: Overwrite `01_intro.qmd` with the new YAML header + skeleton**

Use `Write` to replace the entire file with:

```markdown
---
title: |
  Mathematical Foundations of AI & ML<br>Unit 3: Regression and Classification as Loss Minimization
bibliography: ref.bib
author:
  - name: Prof. Dr. Philipp Pelz
    affiliation:
      - FAU Erlangen-Nürnberg
execute:
  eval: true
  echo: false
format:
  revealjs:
    chalkboard: true
    width: 1920
    height: 1080
    template-partials:
      - title-slide.html
    css: custom.css
    theme: custom.scss
    slide-number: c/t
    logo: "eclipse_logo_small.png"
    background-transition: fade
    footer: "© Philipp Pelz - Mathematical Foundations of AI & ML"
    menu:
      side: left
      loadIcons: true
    navigationMode: default
    controls: true
---

<!-- ===== §1. Setup & framing ===== -->

<!-- ===== §2. Optimization: from first to second order ===== -->

<!-- ===== §3. Loss functions for regression ===== -->

<!-- ===== §4. Loss functions for classification ===== -->

<!-- ===== §5. Expanding linear models — basis functions ===== -->

<!-- ===== §6. Regularization bridge ===== -->

<!-- ===== §7. Unification — Exponential Family & GLMs ===== -->

<!-- ===== §8. Wrap-up ===== -->

::: {#refs}
:::
```

- [ ] **Step 3: Render check**

Run: `cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -10`
Expected: render succeeds (the deck has only a title slide; that's fine).

- [ ] **Step 4: Commit**

```bash
cd /home/philipp/projects/_public_presentations
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: scaffold new file with section markers"
```

---

## Task 2: §1 Setup & framing — slides 1–4

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

- [ ] **Step 1: Replace the §1 marker with the four setup slides**

Use `Edit` to replace `<!-- ===== §1. Setup & framing ===== -->` with:

````markdown
<!-- ===== §1. Setup & framing ===== -->

## Regression and Classification as Loss Minimization

::: {.incremental}
- **The bridge** from data analysis (Unit 2) to learning: turn modeling into an optimization problem.
- **Today's thesis:** loss → optimizer → model class — three threads that unify in the exponential family.
- **Scope:** the mathematical scaffolding behind every supervised learner you'll meet for the rest of this course.
:::

## Where we are in the triad

::: {.columns}
::: {.column width="50%"}
**Just done (Unit 2):**

- Linear regression in matrix form, normal equations.
- Ridge & Lasso closed forms; L1 vs L2 geometry.
- Multicollinearity, pseudo-inverse, kernel hint.
:::
::: {.column width="50%"}
**Coming later:**

- **Unit 6** — full optimization deep dive (momentum, Adam, conditioning, saddle points).
- **Unit 7** — bias-variance decomposition.
- **Unit 8** — full probabilistic learning (MLE, MAP, posterior).
- **ML-PC Unit 2** — already saw Gaussian → MSE, Poisson → Poisson NLL, Bayes/MAP table.
:::
:::

::: {.notes}
Be explicit with students: this unit deliberately *does not* re-derive ridge regression, the bias-variance decomposition, the full MLE/MAP framework, or the optimization deep dive. Those have homes. What this unit *owns* is: Newton's method and IRLS as optimization algorithms, formal basis-function expansion + Runge + splines, and the GLM/exponential-family unification. Everything else is recap or forward-pointer.
:::

## Learning outcomes

By the end of this unit, students can:

::: {.incremental}
- **Recall** the ERM principle and write down the supervised learning objective.
- **Apply** gradient descent, SGD/minibatch SGD, and Newton's method to small problems.
- **Derive** the Newton update from a 2nd-order Taylor expansion and explain its single-step convergence on quadratics.
- **Identify** the right loss function for a given noise model (Gaussian, Bernoulli, Poisson).
- **Analyze** how the exponential-family / GLM framework unifies regression and classification under one likelihood.
- **Recognize** Runge's phenomenon and choose between polynomial, RBF, and spline bases.
:::

## The supervised learning framework

::: {.incremental}
- **Data:** $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ drawn iid from an unknown $p(\mathbf{x}, y)$.
- **Hypothesis:** parameterized predictor $f_{\mathbf{w}}: \mathbf{x} \mapsto \hat{y}$.
- **Loss:** $L(\hat{y}, y)$ scores a single prediction.
- **Population risk:** $R(\mathbf{w}) = \mathbb{E}_{(\mathbf{x},y) \sim p}[L(f_{\mathbf{w}}(\mathbf{x}), y)]$ — what we *want*.
- **Empirical risk:** $\hat R(\mathbf{w}) = \frac{1}{N}\sum_i L(f_{\mathbf{w}}(\mathbf{x}_i), y_i)$ — what we *can compute*.
- **ERM:** $\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \hat R(\mathbf{w})$.
:::
````

- [ ] **Step 2: Verify slide count**

Run: `grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`
Expected: `4` (slides for §1).

- [ ] **Step 3: Render check**

Run: `cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5`
Expected: render succeeds.

- [ ] **Step 4: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §1 setup slides 1-4"
```

---

## Task 3: §2 Optimization — first-order content (slides 5–9)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

This task adapts existing first-order content (currently lines 60–206 of the original file) into a tighter form. The interactive #5 (convex/non-convex plot) is reused.

- [ ] **Step 1: Retrieve the convex/non-convex ojs block from the original**

Run: `git show $BASELINE_SHA:mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd | sed -n '69,103p'`

You'll use the ojs Plot.plot block from this slice for slide 5.

- [ ] **Step 2: Replace the §2 marker with slides 5–9**

Use `Edit` to replace `<!-- ===== §2. Optimization: from first to second order ===== -->` with the content below. Note: the §2 marker stays at the *end* of the section (just before §3) so subsequent §2 tasks can append before it.

The replacement is the original marker → §2 header comment + slides 5–9 + a *new sub-marker* for slides 10–16:

````markdown
<!-- ===== §2. Optimization: from first to second order ===== -->

## Optimization landscape: convex vs non-convex

::: {.columns}
::: {.column width="50%"}
::: {.incremental}
- **Convex:** any local minimum is global. (E.g. MSE for linear regression.)
- **Non-convex:** local minima, saddle points, plateaus. (Anything with a hidden layer.)
- **Practical message:** convex problems have one *correct* answer; non-convex problems have one we can *find*.
:::
:::
::: {.column width="50%"}
```{ojs}
//| fig-align: center
Plot.plot({
  grid: true,
  x: {domain: [-3, 3]},
  y: {domain: [-1, 10]},
  aspectRatio: 1,
  marks: [
    Plot.line(Array.from({length: 100}, (_, i) => {
      let x = -3 + i * 6 / 99;
      return {x: x, y: x*x};
    }), {x: "x", y: "y", stroke: "blue"}),
    Plot.line(Array.from({length: 100}, (_, i) => {
      let x = -3 + i * 6 / 99;
      return {x: x, y: x*x + 3*Math.cos(2*x) + 3};
    }), {x: "x", y: "y", stroke: "red"}),
    Plot.text([{x: 0, y: -0.5, text: "Convex"}], {x: "x", y: "y", text: "text", fill: "blue"}),
    Plot.text([{x: 1.5, y: 8, text: "Non-Convex"}], {x: "x", y: "y", text: "text", fill: "red"})
  ]
})
```
:::
:::

## Gradient descent

::: {.incremental}
- **Idea:** to minimize, step in the steepest descent direction.
- **First-order Taylor:** $f(\mathbf{w} - \eta \nabla f(\mathbf{w})) \approx f(\mathbf{w}) - \eta \|\nabla f(\mathbf{w})\|^2 < f(\mathbf{w})$ for small $\eta > 0$.
- **Update:** $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t)$.
- **Learning rate** $\eta$:
    - too small → slow;
    - too large → overshoots, diverges.
:::

## Stochastic gradient descent (SGD)

::: {.incremental}
- **Cost of full GD:** $\nabla \hat R = \frac{1}{N}\sum_i \nabla L_i$ — $\mathcal{O}(N)$ work per step.
- **Stochastic estimator:** pick one $i$ uniformly at random; use $\nabla L_i$ as the gradient.
- **Unbiased:** $\mathbb{E}_i[\nabla L_i] = \nabla \hat R$ — in expectation we're still doing GD.
- **Behaviour:** noisy steps, fast initial progress, eventually bounces near the minimum.
:::

## Minibatch SGD

::: {.incremental}
- **Compromise:** average over a minibatch of size $b$ (typically 32–256).
- **Update:** $\displaystyle \mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{b}\sum_{i \in \mathcal{B}_t} \nabla L_i(\mathbf{w}_t)$.
- **Why $b$ matters:**
    1. **Variance reduction** — gradient estimate concentrates around the true gradient.
    2. **Vectorization** — modern GPUs do $b$ samples in parallel almost for free.
- **The default** for every modern deep-learning training loop.
:::

## Beyond vanilla SGD — see Unit 6

::: {.incremental}
- **Momentum, Nesterov, RMSProp, AdaGrad, Adam, AdamW** — all first-order, all standard.
- **Conditioning, saddle points, plateaus, mode connectivity** — landscape pathologies.
- **Unit 6 owns this with its own deep-dive interactives.**
- For the rest of *this* unit we ask a different question: **what does second-order information buy us?**
:::

<!-- ===== §2 (continued): second-order content ===== -->
````

- [ ] **Step 3: Render check + slide-count check**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5
```

Expected: slide count = 9 (4 + 5 new). Render succeeds.

- [ ] **Step 4: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §2 first-order optimization slides 5-9"
```

---

## Task 4: §2 second-order content — Hessian, Newton, BFGS, IRLS preview (slides 10, 11, 13, 14, 15)

Slide 12 (the Newton-vs-GD interactive) is built in Task 5; we leave a placeholder for it here.

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

- [ ] **Step 1: Replace the second-order sub-marker with slides 10, 11, [INTERACTIVE PLACEHOLDER], 13, 14, 15**

Use `Edit` to replace `<!-- ===== §2 (continued): second-order content ===== -->` with:

````markdown
<!-- ===== §2 (continued): second-order content ===== -->

## When first-order is slow: the Hessian

::: {.incremental}
- **Hessian** $\mathbf{H} = \nabla^2 f$ — the matrix of second derivatives. It encodes local curvature.
- **Eigenvalues of $\mathbf{H}$** are the principal curvatures. Their ratio is the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$.
- **GD's pain:** in an elongated bowl ($\kappa \gg 1$), GD oscillates across the steep direction while crawling along the shallow one. Convergence rate scales like $\frac{\kappa - 1}{\kappa + 1}$.
- **Unit 6 owns the full conditioning treatment**; here we use it only as motivation for Newton.
:::

## Newton's method

::: {.incremental}
- **Second-order Taylor:** $f(\mathbf{w} + \Delta) \approx f(\mathbf{w}) + \nabla f^T \Delta + \tfrac{1}{2} \Delta^T \mathbf{H}\, \Delta$.
- **Minimize the quadratic** in $\Delta$: $\Delta = -\mathbf{H}^{-1} \nabla f$.
- **Update:** $\boxed{\;\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{H}^{-1} \nabla f(\mathbf{w}_t)\;}$
- **Key property:** if $f$ *is* a quadratic, the update lands on the minimum **in a single step** — independent of the starting point and of $\kappa$.
- **For non-quadratic $f$:** Newton converges *quadratically* near the optimum (error squares each iteration).
:::

::: {.notes}
Walk through the derivation: take $\nabla_\Delta$ of the quadratic approximation, set to zero, solve. Emphasize that this is *the* reason Newton scales independently of $\kappa$ on quadratics — it computes the exact minimum of the local quadratic model, and on a quadratic the local model *is* the function. For non-quadratics the local model is only correct to second order, so we need to iterate, but each iteration shrinks the error quadratically near the optimum.
:::

<!-- INTERACTIVE_NEWTON_VS_GD: filled in by Task 5 -->

## Newton's catch

::: {.incremental}
- **Memory:** $\mathbf{H}$ is $D \times D$ — $\mathcal{O}(D^2)$ to store.
- **Compute:** inverting (or solving with) $\mathbf{H}$ is $\mathcal{O}(D^3)$.
- For a deep network with $D = 10^9$ parameters, $\mathbf{H}$ has $10^{18}$ entries. **Not happening.**
- For *medium-dimensional* convex problems ($D \lesssim 10^4$), Newton is a realistic and excellent choice.
:::

## Quasi-Newton: BFGS and L-BFGS

::: {.incremental}
- **Idea:** don't form $\mathbf{H}$. *Approximate* $\mathbf{H}^{-1}$ from a sequence of gradient differences.
- **BFGS update:** maintain $\mathbf{B}_t \approx \mathbf{H}^{-1}$, update it rank-2 each step using $\mathbf{s}_t = \mathbf{w}_{t+1} - \mathbf{w}_t$ and $\mathbf{y}_t = \nabla f_{t+1} - \nabla f_t$.
- **L-BFGS** (limited-memory BFGS): keep only the last $m$ pairs $(\mathbf{s}_k, \mathbf{y}_k)$ — memory drops from $\mathcal{O}(D^2)$ to $\mathcal{O}(mD)$.
- **Workhorse** for medium-dim convex problems and for fine-tuning small models. `scipy.optimize.minimize(method='L-BFGS-B')`, `torch.optim.LBFGS`.
:::

## Newton on a GLM = IRLS — preview

::: {.incremental}
- For *generalized linear models* (next: §7), Newton's method has a remarkably clean form.
- The Hessian factors cleanly through the design matrix — each Newton step becomes a **weighted least-squares solve**.
- That's the **Iteratively Reweighted Least Squares (IRLS)** algorithm. We'll close the loop with this in §7.
:::
````

- [ ] **Step 2: Slide-count + render**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5
```

Expected: slide count = 14 (9 + 5 new; the interactive placeholder is a comment, not a `## `). Render succeeds.

- [ ] **Step 3: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §2 second-order text slides 10-11,13-15"
```

---

## Task 5: §2 Interactive #6 — Newton vs GD on an ill-conditioned bowl (slide 12)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

This is one of the two new ojs interactives. Function: $f(x,y) = a x^2 + b y^2$, $a \gg b$. Toggle between GD step and Newton step.

- [ ] **Step 1: Replace the interactive placeholder with the Newton vs GD slide**

Use `Edit` to replace `<!-- INTERACTIVE_NEWTON_VS_GD: filled in by Task 5 -->` with:

````markdown
## Interactive: Newton vs Gradient Descent

::: {.columns}
::: {.column width="30%"}
$f(x,y) = a\,x^2 + b\,y^2$ on a 2D ill-conditioned bowl ($b = 1$ fixed).

```{ojs}
//| echo: false
viewof a_cond = Inputs.range([1, 30], {value: 10, step: 1, label: "Conditioning a"})
viewof eta_gd = Inputs.range([0.01, 0.55], {value: 0.18, step: 0.01, label: "GD η"})
viewof n_steps = Inputs.range([1, 30], {value: 8, step: 1, label: "# steps"})
viewof method = Inputs.radio(["Gradient descent", "Newton's method", "Both"], {value: "Both", label: "Algorithm"})
```

- **GD update:** $\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla f$.
- **Newton update:** $\mathbf{w} \leftarrow \mathbf{w} - \mathbf{H}^{-1} \nabla f$ — for this quadratic, $\mathbf{H} = \mathrm{diag}(2a, 2b)$.
- Watch Newton land on the origin in one step regardless of $a$, while GD oscillates as $a$ grows.
:::

::: {.column width="70%"}
```{ojs}
//| echo: false
//| fig-align: center
chart_newton_vs_gd = {
  // Trajectory builders
  function gdSteps(a, b, eta, k, x0, y0) {
    const pts = [{x: x0, y: y0, iter: 0}];
    let x = x0, y = y0;
    for (let i = 1; i <= k; i++) {
      x = x - eta * 2 * a * x;
      y = y - eta * 2 * b * y;
      pts.push({x, y, iter: i});
      if (Math.abs(x) > 8 || Math.abs(y) > 8) break;
    }
    return pts;
  }
  function newtonSteps(a, b, k, x0, y0) {
    // For diagonal H, Newton step = -(1/2a)*(2a*x) = -x and -y → reaches origin in 1 step.
    const pts = [{x: x0, y: y0, iter: 0}];
    let x = x0, y = y0;
    for (let i = 1; i <= k; i++) {
      x = 0; y = 0;
      pts.push({x, y, iter: i});
    }
    return pts;
  }

  const a = a_cond, b = 1, x0 = -3.5, y0 = 2.5;
  const gd = gdSteps(a, b, eta_gd, n_steps, x0, y0);
  const nt = newtonSteps(a, b, n_steps, x0, y0);

  // Contours of f(x,y) = a x^2 + b y^2 → ellipses sqrt(c/a), sqrt(c/b)
  const contours = [];
  for (let c = 0.5; c <= 30; c *= 1.6) {
    for (let t = 0; t < Math.PI * 2; t += 0.05) {
      contours.push({x: Math.sqrt(c/a) * Math.cos(t), y: Math.sqrt(c/b) * Math.sin(t), level: c});
    }
  }

  const showGD = method !== "Newton's method";
  const showNT = method !== "Gradient descent";

  return Plot.plot({
    grid: true,
    x: {domain: [-5, 5]},
    y: {domain: [-5, 5]},
    aspectRatio: 1,
    width: 720, height: 600,
    marks: [
      Plot.dot(contours, {x: "x", y: "y", r: 1, fill: "lightgray"}),
      ...(showGD ? [
        Plot.line(gd, {x: "x", y: "y", stroke: "#e67e22", strokeWidth: 2}),
        Plot.dot(gd, {x: "x", y: "y", fill: "#e67e22", r: 4})
      ] : []),
      ...(showNT ? [
        Plot.line(nt, {x: "x", y: "y", stroke: "#3498db", strokeWidth: 2, strokeDasharray: "4,4"}),
        Plot.dot(nt, {x: "x", y: "y", fill: "#3498db", r: 5})
      ] : []),
      Plot.dot([{x: 0, y: 0}], {x: "x", y: "y", fill: "black", r: 5}),
      Plot.dot([{x: x0, y: y0}], {x: "x", y: "y", stroke: "black", r: 6})
    ]
  });
}
```
:::
:::
````

- [ ] **Step 2: Render check**

Run: `cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -10`
Expected: render succeeds. Open `_site/.../01_intro.html` (or run `quarto preview` once locally) and verify:
  - Three sliders + radio button render.
  - Newton trajectory (blue dashed) jumps to origin in one step.
  - GD trajectory (orange) visibly oscillates as `a` is increased to ~20.

Run: `grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`
Expected: 15.

- [ ] **Step 3: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §2 interactive #6 Newton vs GD slide 12"
```

---

## Task 6: §2 Checkpoint #1 (slide 16)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

- [ ] **Step 1: Append the checkpoint slide before the §3 marker**

Use `Edit` to replace `<!-- ===== §3. Loss functions for regression ===== -->` with the checkpoint slide followed by the same §3 marker (so subsequent tasks can still find it):

````markdown
## Checkpoint: optimization

1. A learning rate $\eta$ is "too large." What's the symptom you'd see in a GD trajectory?
   - (a) Slow, monotone decrease toward the minimum.
   - (b) Trajectory oscillates and may diverge.
   - (c) Stuck at a saddle point.
   - (d) Newton's method takes over.
2. Why does Newton's method converge in **one step** on a quadratic, regardless of conditioning?
   - (a) Because the Hessian is identity for quadratics.
   - (b) Because the second-order Taylor approximation *is* the function for quadratics, so we land on the exact minimum of the local model.
   - (c) Because the gradient is zero at the minimum.
   - (d) It doesn't — that's a myth.
3. We don't use plain Newton's method to train deep networks. Why not?
   - (a) Newton diverges on non-convex losses.
   - (b) Storing and inverting an $D\times D$ Hessian for $D = 10^9$ parameters is infeasible ($\mathcal{O}(D^2)$ memory, $\mathcal{O}(D^3)$ inversion).
   - (c) Adam is provably faster.
   - (d) Newton's method requires labeled data.

::: {.notes}
**Answers:** 1(b), 2(b), 3(b). For 2, push back on (a) — students often think "single-step convergence" requires a special $\mathbf H$. Push back on (c) — having zero gradient at the minimum is true but doesn't explain *single-step*. The point is that Newton minimizes the local quadratic model exactly, and on a quadratic the model equals the function. For 3, also note (a) is half-true (Newton can step toward saddles, not literally diverge) — the binding constraint is memory/compute.
:::

<!-- ===== §3. Loss functions for regression ===== -->
````

- [ ] **Step 2: Slide-count + render**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5
```

Expected: slide count = 16. Render succeeds.

- [ ] **Step 3: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §2 checkpoint #1 slide 16"
```

---

## Task 7: §3 Regression losses — port + adapt (slides 17–22)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

The drag-the-outlier interactive (slide 21) is at lines 271–430 of the baseline. Port it verbatim. Other slides are tighter rewrites of existing content.

- [ ] **Step 1: Retrieve the existing interactive #2 ojs block**

Run: `git show $BASELINE_SHA:mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd | sed -n '271,430p' > /tmp/u3_int2.qmd`

You'll paste the ojs block (the `## Interactive: Regression Robustness` heading and its content from this slice) into the new slide 21 verbatim. The heading should be relabeled to **`## Interactive: Drag-the-outlier (MSE / MAE / Huber)`**.

- [ ] **Step 2: Replace the §3 marker with slides 17–22**

Use `Edit` to replace the `<!-- ===== §3. Loss functions for regression ===== -->` marker with the content below. **Where the comment `[PASTE_INT2_OJS_BLOCK_HERE]` appears, paste the inner content of the existing interactive #2 — that is, the entire body of the existing `## Interactive: Regression Robustness` slide (everything between that `## ` line and the next `## ` line in `/tmp/u3_int2.qmd`)**, retaining its `:::{.columns}` structure.

````markdown
<!-- ===== §3. Loss functions for regression ===== -->

## Loss as decision proxy

::: {.incremental}
- A loss $L(\hat y, y)$ is a *quantitative penalty* for being wrong — not a fundamental quantity. We design it.
- Different application contexts → different penalties:
    - Should errors grow quadratically (smooth tails) or linearly (robust to outliers)?
    - Are false positives and false negatives equally costly?
    - Do we need calibrated probabilities, or just labels?
- The loss encodes the *engineering question* you're asking.
:::

## Mean squared error (MSE)

::: {.incremental}
- $L_{\text{MSE}}(\hat y, y) = (\hat y - y)^2$.
- **Geometry:** smooth convex bowl — gradient descent's ideal landscape.
- **Probabilistic identity:** minimizing MSE = MLE assuming iid Gaussian residuals $\varepsilon \sim \mathcal{N}(0, \sigma^2)$.
- See **ML-PC Unit 2 §26** for the full Gaussian → MSE derivation; see **Unit 8** for the formal MLE machinery.
:::

## Mean absolute error (MAE)

::: {.incremental}
- $L_{\text{MAE}}(\hat y, y) = |\hat y - y|$.
- **Linear penalty** → much less sensitive to outliers than MSE.
- **Probabilistic identity:** MLE under Laplacian residuals.
- **Optimization caveat:** non-differentiable at zero. Use sub-gradient methods, or smooth via Huber.
:::

## Huber loss

::: {.incremental}
- Piecewise definition with parameter $\delta$:
$$L_\delta(\hat y, y) = \begin{cases} \tfrac{1}{2}(\hat y - y)^2 & |\hat y - y| \le \delta \\ \delta(|\hat y - y| - \tfrac{1}{2}\delta) & |\hat y - y| > \delta \end{cases}$$
- **Quadratic** in the small-error regime (smooth optimization), **linear** in the tails (outlier-robust).
- **Standard tool** for industrial / engineering data where most points are clean but occasional glitches occur [@neuer2024machine].
:::

## Interactive: Drag-the-outlier (MSE / MAE / Huber)

[PASTE_INT2_OJS_BLOCK_HERE]

## Beyond Gaussian: heteroscedastic and count data

::: {.incremental}
- **Heteroscedastic noise** (variance varies with signal): predict $\sigma^2$ alongside $\hat y$; loss becomes $\frac{(y - \hat y)^2}{2\hat\sigma^2} + \tfrac{1}{2}\log\hat\sigma^2$.
- **Poisson / count data** (low-dose imaging, photon counting): use **Poisson NLL** $L = \hat\mu - y\log\hat\mu$, *not* MSE. See **ML-PC Unit 2 §27**.
- **Heavy-tailed noise:** use Student-$t$ or log-cosh.
- **The principle:** loss = NLL of the noise model. We'll formalize this in §7.
:::
````

- [ ] **Step 3: Replace `[PASTE_INT2_OJS_BLOCK_HERE]` with the actual ojs body**

Open `/tmp/u3_int2.qmd`, find the body of the existing `## Interactive: Regression Robustness` slide (the `:::{.columns}...:::` structure containing the d3 chart), and `Edit` the placeholder to insert it. The block runs ~150 lines. Remove the original heading; keep only the body.

- [ ] **Step 4: Slide-count + render**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5
```

Expected: slide count = 22. Render succeeds. Manually verify the drag-the-outlier interactive still works.

- [ ] **Step 5: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §3 regression losses slides 17-22"
```

---

## Task 8: §4 Classification losses — port + adapt (slides 23–26)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

Cross-entropy interactive at original lines 448–590; hinge slide at original lines 592–598.

- [ ] **Step 1: Retrieve the cross-entropy interactive ojs block**

Run: `git show $BASELINE_SHA:mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd | sed -n '448,590p' > /tmp/u3_int3.qmd`

- [ ] **Step 2: Replace the §4 marker with slides 23–26**

Use `Edit` to replace `<!-- ===== §4. Loss functions for classification ===== -->` with:

````markdown
<!-- ===== §4. Loss functions for classification ===== -->

## The 0–1 loss problem

::: {.incremental}
- For binary classification with $y \in \{0, 1\}$, the natural loss is $L = \mathbf{1}\{\hat y \ne y\}$.
- **The bug:** non-differentiable, gradient is zero almost everywhere — gradient descent has nothing to follow.
- **The fix:** use a *surrogate loss* — a smooth differentiable function that bounds the 0–1 loss from above.
- Surrogates also let us output **calibrated probabilities**, not just labels.
:::

## Cross-entropy = Bernoulli / Categorical NLL

::: {.incremental}
- **Binary:** model $p(y=1\mid\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x})$ with sigmoid $\sigma$.
- **Cross-entropy loss:** $L = -[y \log p + (1-y)\log(1-p)]$.
- **Identity:** this is the negative log-likelihood of a Bernoulli$(p)$ on $y$.
- **Multi-class:** softmax + categorical cross-entropy = NLL of a Categorical distribution.
- **Behaviour:** *confident-but-wrong* predictions are punished hugely (NLL → ∞). See **Unit 8** for the full MLE framework.
:::

## Interactive: Cross-entropy & decision boundary

[PASTE_INT3_OJS_BLOCK_HERE]

## Margin-based view: hinge & calibration

::: {.incremental}
- **Hinge loss** ($y \in \{-1,+1\}$): $L = \max(0, 1 - y\hat y)$. **Goal:** correct classification *with a margin*, not just correct labels — the SVM principle.
- **Once correctly classified beyond the margin**, the gradient is zero — no further "improvement" is needed. Crisp decision-boundary geometry, but no probabilities.
- **Proper scoring rules** (Brier, log-loss): the loss is *minimized in expectation by the true probability*. Cross-entropy is proper; misclassification rate is not.
- **Choose the loss to match what you need** — labels (hinge) or calibrated probabilities (cross-entropy).
:::
````

- [ ] **Step 3: Replace `[PASTE_INT3_OJS_BLOCK_HERE]` with the body of the original cross-entropy interactive**

From `/tmp/u3_int3.qmd`, take the body of `## Interactive: Cross-Entropy & Decision Boundary` (everything after the heading, before the next `## `) and `Edit` it into place.

- [ ] **Step 4: Slide-count + render**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5
```

Expected: slide count = 26. Render succeeds. Manually verify the cross-entropy interactive's three sliders + decision boundary work.

- [ ] **Step 5: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §4 classification losses slides 23-26"
```

---

## Task 9: §5 Basis functions — text content (slides 27–30, 32–33)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

Slide 31 (existing polynomial interactive, relabeled) is Task 10. Slide 34 (new basis-explorer interactive) is Task 11. We leave placeholders for both.

- [ ] **Step 1: Replace the §5 marker with slides 27–30 + placeholder + 32–33 + placeholder**

Use `Edit` to replace `<!-- ===== §5. Expanding linear models — basis functions ===== -->` with:

````markdown
<!-- ===== §5. Expanding linear models — basis functions ===== -->

## The linearity principle

::: {.incremental}
- A "linear model" is **linear *in the parameters* $\mathbf{w}$** — *not* in the inputs $\mathbf{x}$.
- Replace raw $\mathbf{x}$ with a feature vector $\boldsymbol\phi(\mathbf{x})$. The predictor becomes
$$f_{\mathbf{w}}(\mathbf{x}) = \mathbf{w}^T \boldsymbol\phi(\mathbf{x}).$$
- The model is now non-linear in $\mathbf{x}$ but **all the linear-regression machinery still applies** to $\mathbf{w}$.
- This is one of the most useful conceptual moves in ML.
:::

## Formal basis-function expansion

::: {.incremental}
- **Map** $\boldsymbol\phi: \mathbb{R}^d \to \mathbb{R}^M$, $\boldsymbol\phi(\mathbf{x}) = (\phi_1(\mathbf{x}), \ldots, \phi_M(\mathbf{x}))^T$.
- **Design matrix:** $\boldsymbol\Phi \in \mathbb{R}^{N \times M}$ with $\boldsymbol\Phi_{ij} = \phi_j(\mathbf{x}_i)$.
- **Normal equations** (from Unit 2, applied to $\boldsymbol\Phi$):
$$\hat{\mathbf{w}} = (\boldsymbol\Phi^T \boldsymbol\Phi)^{-1} \boldsymbol\Phi^T \mathbf{y}.$$
- The same OLS / Ridge / Lasso closed forms — they just live in feature space now.
:::

## Polynomial basis

::: {.incremental}
- **Univariate:** $\phi_j(x) = x^{j}$ for $j = 0, 1, \ldots, M$.
- The design matrix is the **Vandermonde matrix** $\boldsymbol\Phi_{ij} = x_i^{j}$.
- Recovers polynomial regression. Nice and familiar — but watch out:
:::

## Runge's phenomenon

::: {.incremental}
- High-degree polynomials *interpolated at evenly spaced points* oscillate wildly near the boundary.
- Classic example: fit $f(x) = 1/(1 + 25 x^2)$ on $[-1, 1]$ with degree-15 polynomial → boundary error grows with degree, not shrinks.
- **Lesson:** higher complexity ≠ better fit. Global polynomials are a *poor* basis when you need flexibility *and* boundedness.
- We'll see two fixes: localize the basis (RBFs, splines) or constrain the weights (regularization, §6).
:::

<!-- INTERACTIVE_POLYNOMIAL: filled in by Task 10 -->

## Radial basis functions (RBFs)

::: {.incremental}
- Pick **centers** $\boldsymbol\mu_1, \ldots, \boldsymbol\mu_M$ and a **bandwidth** $\sigma$.
- Each basis function is **localized**:
$$\phi_k(\mathbf{x}) = \exp\!\left(-\tfrac{\|\mathbf{x} - \boldsymbol\mu_k\|^2}{2\sigma^2}\right).$$
- $\sigma$ controls width: small $\sigma$ → sharp local "bumps", large $\sigma$ → smooth global features.
- Center placement matters: equispaced, $k$-means on data, or *every data point is a center* (recovers a kernel method — see Unit 2).
:::

## Splines

::: {.incremental}
- **Idea:** stitch low-degree polynomials together at fixed *knots* with continuity constraints.
- **Cubic spline:** piecewise degree-3 polynomials, continuous in value, 1st, and 2nd derivative — visually indistinguishable from a smooth curve.
- **B-spline basis:** an explicit set of $M$ basis functions that span the same space; each B-spline has *local support*, so the design matrix is sparse and conditioning is excellent.
- **Why splines fix Runge:** the basis is *local*. Increasing complexity adds knots, not global oscillation modes.
:::

<!-- INTERACTIVE_BASIS_EXPLORER: filled in by Task 11 -->
````

- [ ] **Step 2: Slide-count + render**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5
```

Expected: slide count = 32 (26 + 6 new; placeholders are comments). Render succeeds.

- [ ] **Step 3: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §5 basis functions text slides 27-30,32-33"
```

---

## Task 10: §5 Polynomial fitting interactive — port + relabel (slide 31)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

Original interactive at lines 756–1003 of the baseline.

- [ ] **Step 1: Retrieve the interactive**

Run: `git show $BASELINE_SHA:mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd | sed -n '756,1003p' > /tmp/u3_int4.qmd`

- [ ] **Step 2: Replace the placeholder with the relabeled slide**

Use `Edit` to replace `<!-- INTERACTIVE_POLYNOMIAL: filled in by Task 10 -->` with:

````markdown
## Interactive: Runge's phenomenon — polynomial fitting

[PASTE_INT4_OJS_BLOCK_HERE]
````

Then in a follow-up `Edit`, replace `[PASTE_INT4_OJS_BLOCK_HERE]` with the *body* of the existing `## Interactive: Polynomial Fitting & Overfitting` slide from `/tmp/u3_int4.qmd` (everything after the original heading, before the next `## `).

The interactive is being **relabeled** — its original heading was about "overfitting"; in the redesign it doubles as the Runge demo. The body content is unchanged.

- [ ] **Step 3: Render check + manual interactive check**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5
```

Expected: slide count = 33. Render succeeds. Increase the polynomial-degree slider in the rendered HTML and visually confirm the boundary oscillation appears (Runge's phenomenon).

- [ ] **Step 4: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §5 polynomial-fitting interactive slide 31"
```

---

## Task 11: §5 Interactive #7 — basis-function explorer (slide 34)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

The second of the two new interactives. Three radio-button modes (polynomial, RBF, B-spline) over a noisy $\sin(2\pi x)$ dataset.

- [ ] **Step 1: Replace the placeholder with the basis-function explorer slide**

Use `Edit` to replace `<!-- INTERACTIVE_BASIS_EXPLORER: filled in by Task 11 -->` with:

````markdown
## Interactive: Basis function explorer

::: {.columns}
::: {.column width="28%"}
Same data, three bases. Notice that **all three** fits use the *same normal equations* $\hat{\mathbf{w}} = (\boldsymbol\Phi^T\boldsymbol\Phi)^{-1}\boldsymbol\Phi^T\mathbf{y}$ — only $\boldsymbol\Phi$ changes.

```{ojs}
//| echo: false
viewof basis_kind = Inputs.radio(["Polynomial", "Gaussian RBF", "B-spline (cubic)"], {value: "Polynomial", label: "Basis"})
viewof n_basis = Inputs.range([2, 25], {value: 8, step: 1, label: "# basis fns"})
viewof rbf_sigma = Inputs.range([0.03, 0.5], {value: 0.12, step: 0.01, label: "RBF σ (RBF only)"})
viewof show_basis = Inputs.toggle({value: true, label: "Show basis fns"})
```

- **Polynomial** → global, prone to Runge.
- **RBF** → localized; bandwidth $\sigma$ trades smoothness vs flexibility.
- **B-spline** → piecewise polynomial with local support; clean conditioning.
:::

::: {.column width="72%"}
```{ojs}
//| echo: false
//| fig-align: center
chart_basis = {
  // Fixed seeded dataset: noisy sin(2 pi x)
  function rng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }
  const rand = rng(42);
  function nrand() {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const N = 30, sigma_y = 0.15;
  const data = [];
  for (let i = 0; i < N; i++) {
    const x = i / (N - 1);
    const y = Math.sin(2 * Math.PI * x) + sigma_y * nrand();
    data.push({x, y});
  }

  function buildPhi(xs, M, kind) {
    const Phi = xs.map(() => new Array(M).fill(0));
    if (kind === "Polynomial") {
      for (let i = 0; i < xs.length; i++)
        for (let j = 0; j < M; j++) Phi[i][j] = Math.pow(xs[i], j);
    } else if (kind === "Gaussian RBF") {
      const centers = M === 1 ? [0.5] : Array.from({length: M}, (_, k) => k / (M - 1));
      for (let i = 0; i < xs.length; i++)
        for (let j = 0; j < M; j++)
          Phi[i][j] = Math.exp(-((xs[i] - centers[j]) ** 2) / (2 * rbf_sigma * rbf_sigma));
    } else {
      // Cubic B-spline basis with M basis functions and uniform knots on [0,1].
      // Use de Boor recursion; clamp endpoints by extending knots.
      const k = 3; // cubic
      const nKnots = M + k + 1;
      // Open uniform knot vector on [0, 1]:
      const knots = [];
      for (let i = 0; i < nKnots; i++) {
        if (i <= k) knots.push(0);
        else if (i >= nKnots - 1 - k) knots.push(1);
        else knots.push((i - k) / (M - k));
      }
      function N_jk(j, k_, x) {
        if (k_ === 0) {
          return (knots[j] <= x && x < knots[j + 1]) || (x === 1 && j === M - 1) ? 1 : 0;
        }
        const d1 = knots[j + k_] - knots[j];
        const d2 = knots[j + k_ + 1] - knots[j + 1];
        const t1 = d1 > 0 ? (x - knots[j]) / d1 * N_jk(j, k_ - 1, x) : 0;
        const t2 = d2 > 0 ? (knots[j + k_ + 1] - x) / d2 * N_jk(j + 1, k_ - 1, x) : 0;
        return t1 + t2;
      }
      for (let i = 0; i < xs.length; i++)
        for (let j = 0; j < M; j++) Phi[i][j] = N_jk(j, k, xs[i]);
    }
    return Phi;
  }

  // Solve normal equations via QR-ish path; small system, use Cholesky-style direct solve.
  function solveNormal(Phi, y) {
    const N = Phi.length, M = Phi[0].length;
    const A = Array.from({length: M}, () => new Array(M).fill(0));
    const b = new Array(M).fill(0);
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < M; j++) {
        let s = 0;
        for (let n = 0; n < N; n++) s += Phi[n][i] * Phi[n][j];
        A[i][j] = s;
      }
      let s = 0;
      for (let n = 0; n < N; n++) s += Phi[n][i] * y[n];
      b[i] = s;
      A[i][i] += 1e-8; // tiny ridge for numerical stability
    }
    // Gaussian elimination
    for (let i = 0; i < M; i++) {
      let piv = i;
      for (let r = i + 1; r < M; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
      [A[i], A[piv]] = [A[piv], A[i]];
      [b[i], b[piv]] = [b[piv], b[i]];
      for (let r = i + 1; r < M; r++) {
        const f = A[r][i] / A[i][i];
        for (let c = i; c < M; c++) A[r][c] -= f * A[i][c];
        b[r] -= f * b[i];
      }
    }
    const w = new Array(M).fill(0);
    for (let i = M - 1; i >= 0; i--) {
      let s = b[i];
      for (let j = i + 1; j < M; j++) s -= A[i][j] * w[j];
      w[i] = s / A[i][i];
    }
    return w;
  }

  const xs = data.map(p => p.x);
  const ys = data.map(p => p.y);
  const Phi = buildPhi(xs, n_basis, basis_kind);
  const w = solveNormal(Phi, ys);

  // Curve
  const grid = Array.from({length: 200}, (_, i) => i / 199);
  const Phigrid = buildPhi(grid, n_basis, basis_kind);
  const yhat = Phigrid.map(row => row.reduce((s, v, j) => s + v * w[j], 0));
  const fit = grid.map((x, i) => ({x, y: yhat[i]}));

  // Basis fns curves (only if toggled)
  const basisCurves = [];
  if (show_basis) {
    for (let j = 0; j < n_basis; j++) {
      for (let i = 0; i < grid.length; i++) {
        basisCurves.push({x: grid[i], y: Phigrid[i][j], k: j});
      }
    }
  }

  return Plot.plot({
    grid: true,
    width: 950,
    height: 600,
    x: {domain: [0, 1]},
    y: {domain: [-1.4, 1.4]},
    marks: [
      Plot.line(grid.map(x => ({x, y: Math.sin(2 * Math.PI * x)})), {x: "x", y: "y", stroke: "#bdc3c7", strokeWidth: 2, strokeDasharray: "3,3"}),
      ...(show_basis ? [Plot.line(basisCurves, {x: "x", y: "y", z: "k", stroke: "lightblue", strokeOpacity: 0.6})] : []),
      Plot.dot(data, {x: "x", y: "y", fill: "#e74c3c", r: 4}),
      Plot.line(fit, {x: "x", y: "y", stroke: "#2c3e50", strokeWidth: 3})
    ]
  });
}
```
:::
:::
````

- [ ] **Step 2: Render check + manual interactive check**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -10
```

Expected: slide count = 34. Render succeeds. Manually verify in the rendered HTML:
- The radio button toggles between three bases.
- Increasing # basis fns shows the fit getting more flexible.
- For RBF, σ slider visibly controls width.
- For polynomial mode at $M = 20$, you can see Runge-like oscillation at the boundaries.
- The light-blue basis-function curves render when "Show basis fns" is on.

- [ ] **Step 3: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §5 interactive #7 basis-function explorer slide 34"
```

---

## Task 12: §5 Closing slides + checkpoint #2 (slides 35–38)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

- [ ] **Step 1: Replace the §6 marker with slides 35–38 + the §6 marker**

Use `Edit` to replace `<!-- ===== §6. Regularization bridge ===== -->` with:

````markdown
## The bias-variance picture (one-frame summary)

::: {.columns}
::: {.column width="50%"}
::: {.incremental}
- **High bias** (underfit): basis is too rigid — wrong systematic shape.
- **High variance** (overfit): basis is too flexible — fits noise.
- **Total expected error** = $\text{Bias}^2 + \text{Variance} + \sigma^2_{\text{irreducible}}$.
- **Sweet spot** depends on $N$: more data lets you afford more flexibility.
- **Full decomposition + math:** Unit 7.
:::
:::
::: {.column width="50%"}
```{mermaid}
%%| echo: false
graph TD
    Complexity[Model Complexity →]
    Total[Total Error]
    Bias["Bias²"]
    Var[Variance]
    Bias --> Total
    Var --> Total
    Complexity --> Bias
    Complexity --> Var
    style Total fill:#f96,stroke:#333
```
:::
:::

## Connection to kernels

::: {.incremental}
- If $M$ is huge (or infinite), forming $\boldsymbol\Phi^T\boldsymbol\Phi$ is impossible.
- **Kernel trick:** any algorithm that depends only on $\langle\boldsymbol\phi(\mathbf{x}_i), \boldsymbol\phi(\mathbf{x}_j)\rangle$ can compute that inner product *without ever materializing $\boldsymbol\phi$*.
- $k(\mathbf{x},\mathbf{x}') = \exp(-\|\mathbf{x}-\mathbf{x}'\|^2/(2\sigma^2))$ → infinite-dim feature space, finite computation.
- **See Unit 2** "Kernel hint from inner products" — full treatment in advanced courses (Gaussian processes, SVMs).
:::

## Bridge: complex models need a constraint

::: {.incremental}
- We can make our hypothesis class arbitrarily flexible by stacking basis functions.
- With finite data, *more flexibility = more variance = worse generalization* (without help).
- **The fix:** constrain the parameters. Penalize complexity.
- That's regularization — and it has a beautifully clean Bayesian interpretation.
:::

## Checkpoint: basis functions

1. We say RBF regression is a "linear model." In what sense?
   - (a) The map $\mathbf{x} \mapsto \hat y$ is a straight line.
   - (b) The model is **linear in the parameters $\mathbf{w}$**, even though it's nonlinear in $\mathbf{x}$.
   - (c) Both (a) and (b).
   - (d) Neither — RBF is a fundamentally nonlinear model.
2. Why does Runge's phenomenon happen?
   - (a) Because we're using the wrong loss.
   - (b) Because high-degree polynomials are *global* — local data changes affect the whole curve, and equispaced interpolation forces large oscillations near the boundary.
   - (c) Because polynomial regression overfits noise.
   - (d) Because the design matrix is singular at high degree.
3. Going from polynomial basis to B-spline basis with the same $M$:
   - (a) Changes the model class entirely; OLS no longer applies.
   - (b) Changes only the design matrix $\boldsymbol\Phi$; the normal equations are identical.
   - (c) Forces us to use gradient descent.
   - (d) Adds regularization automatically.

::: {.notes}
**Answers:** 1(b), 2(b), 3(b). Discuss why (c) in Q1 is wrong: although RBF is nonlinear in $\mathbf{x}$, the *learning problem* is linear in $\mathbf{w}$ — that's why we can use OLS. For Q2, push back on (c) — Runge's phenomenon happens *even when there's no noise at all*; it's a property of the basis, not of the noise. For Q3, this is *the* punchline of §5: changing basis = changing $\boldsymbol\Phi$, nothing else.
:::

<!-- ===== §6. Regularization bridge ===== -->
````

- [ ] **Step 2: Slide-count + render**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5
```

Expected: slide count = 38. Render succeeds.

- [ ] **Step 3: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §5 bias-variance, kernel pointer, checkpoint #2 slides 35-38"
```

---

## Task 13: §6 Regularization bridge (slides 39–41)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

- [ ] **Step 1: Replace the §6 marker with slides 39–41**

Use `Edit` to replace `<!-- ===== §6. Regularization bridge ===== -->` with:

````markdown
<!-- ===== §6. Regularization bridge ===== -->

## Recap from Unit 2: Ridge & Lasso

::: {.columns}
::: {.column width="50%"}
- **Ridge** ($\ell_2$ penalty $\lambda\|\mathbf{w}\|_2^2$):
$$\hat{\mathbf{w}}_{\text{ridge}} = (\boldsymbol\Phi^T\boldsymbol\Phi + \lambda\mathbf{I})^{-1}\boldsymbol\Phi^T\mathbf{y}.$$
Spectral effect: shifts every eigenvalue by $+\lambda$.
:::
::: {.column width="50%"}
- **Lasso** ($\ell_1$ penalty $\lambda\|\mathbf{w}\|_1$): no closed form, but the constraint region has corners → **exact sparsity**.
- **Constraint geometry** (sphere vs diamond) — see Unit 2 for the full picture.
:::
:::

::: {.fragment}
*We do not rederive these here — Unit 2 owns the derivations.*
:::

## The MAP interpretation: every regularizer is a prior

::: {.incremental}
- **MAP estimate:** $\hat{\mathbf{w}} = \arg\max_{\mathbf{w}}\left[\log p(\mathbf{y}\mid\mathbf{w}) + \log p(\mathbf{w})\right]$.
- The first term is the **negative loss** (likelihood). The second term is the **prior**.
- **Gaussian prior** $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2\mathbf{I})$ → $\log p(\mathbf{w}) \propto -\tfrac{1}{2\tau^2}\|\mathbf{w}\|_2^2$ → **Ridge**.
- **Laplace prior** $\mathbf{w}_j \sim \text{Laplace}(0, b)$ → $\log p(\mathbf{w}) \propto -\tfrac{1}{b}\|\mathbf{w}\|_1$ → **Lasso**.
- **Punchline:** every loss is an NLL; every regularizer is a prior. See **ML-PC Unit 2 §28** for the full ML↔Bayes table; **Unit 8** for the formal posterior treatment.
:::

## What lives elsewhere (and why)

- **Dropout, batch-norm, label smoothing, focal loss, early stopping, data augmentation** → deep-learning units. Implementation tricks, not foundations.
- **Full bias-variance decomposition** → Unit 7.
- **Full Bayesian (posterior, predictive, evidence)** → Unit 8.
- **First-order optimizer zoo (Adam, RMSProp, Nesterov)** → Unit 6.
- **Physical noise → loss derivations** (Gaussian, Poisson, Weibull) → ML-PC Unit 2.
- *This* unit's contribution: the *abstraction* that ties them together — coming next.
````

- [ ] **Step 2: Slide-count + render**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5
```

Expected: slide count = 41. Render succeeds.

- [ ] **Step 3: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §6 regularization bridge slides 39-41"
```

---

## Task 14: §7 GLM unification — exponential family + IRLS (slides 42–46)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

The mathematical-foundations payoff. All new content.

- [ ] **Step 1: Replace the §7 marker with slides 42–46**

Use `Edit` to replace `<!-- ===== §7. Unification — Exponential Family & GLMs ===== -->` with:

````markdown
<!-- ===== §7. Unification — Exponential Family & GLMs ===== -->

## Exponential family: canonical form

::: {.incremental}
- A distribution belongs to the **exponential family** if its density has the form
$$p(y\mid\eta) = h(y)\,\exp\!\left(\eta^T T(y) - A(\eta)\right).$$
- $\eta$ — **natural parameter** (the parameter we'll learn).
- $T(y)$ — **sufficient statistic** of the data.
- $A(\eta)$ — **log-partition / cumulant function** (normalizer).
- $h(y)$ — **base measure** (does not depend on $\eta$).
:::

::: {.notes}
This canonical form is *one* of the most useful identities in statistical ML. Its power is unification: **dozens** of common distributions fit it, and the same theorems apply to all of them. Tell students: the next two slides will show that Gaussian, Bernoulli, and Poisson — three distributions they think of as totally different — are *the same object* with different $T$ and $A$.
:::

## Three examples in canonical form

| Distribution | $T(y)$ | $A(\eta)$ | $\eta$ in terms of usual parameter |
|---|---|---|---|
| **Gaussian** ($\sigma^2$ known) | $y$ | $\eta^2/2$ | $\eta = \mu$ |
| **Bernoulli** | $y$ | $\log(1 + e^{\eta})$ | $\eta = \log\frac{p}{1-p}$ (logit) |
| **Poisson** | $y$ | $e^{\eta}$ | $\eta = \log\mu$ |

- Verify: plug each into $p(y\mid\eta) = h(y)\exp(\eta\, T(y) - A(\eta))$ and you recover the original density.
- $h(y)$ for Poisson is $1/y!$; for Bernoulli, $h(y) = 1$; for Gaussian (with $\sigma^2$ known), $h(y) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-y^2/2\sigma^2)$.

## Link function and the mean–parameter map

::: {.incremental}
- For an exponential family, **mean** $\mu = \mathbb{E}[T(Y)] = A'(\eta)$ and **variance** $\mathrm{Var}[T(Y)] = A''(\eta)$. ($A$ is convex.)
- A **GLM** says: model the mean as a function of features.
$$g(\mu) = \mathbf{w}^T \boldsymbol\phi(\mathbf{x}),$$
where $g$ is the **link function**. The right-hand side is the *linear predictor*.
- **Canonical link:** choose $g$ such that $g(\mu) = \eta$. Then $\eta = \mathbf{w}^T\boldsymbol\phi(\mathbf{x})$ — directly modelling the natural parameter.
- Canonical links by distribution: Gaussian → identity; Bernoulli → logit ($\sigma^{-1}$); Poisson → log.
:::

## The unification table

| Distribution | Canonical link | $\mu(\eta)$ | NLL → recovered loss |
|---|---|---|---|
| **Gaussian** | identity | $\mu = \eta$ | $\tfrac{1}{2}(y - \mu)^2$ → **MSE** |
| **Bernoulli** | logit | $\mu = \sigma(\eta) = \frac{1}{1+e^{-\eta}}$ | $-y\log\mu - (1-y)\log(1-\mu)$ → **cross-entropy** |
| **Poisson** | log | $\mu = e^{\eta}$ | $\mu - y\log\mu$ → **Poisson NLL** |

::: {.fragment}
**MSE, cross-entropy, and Poisson NLL are not three losses — they are one loss applied to three different distributions.** Forward-pointers: ML-PC Unit 2 §26 (Gaussian → MSE), §27 (Poisson → Poisson NLL), §28 (full ML↔Bayes table).
:::

## IRLS = Newton's method on a GLM

::: {.incremental}
- For a GLM with canonical link, the log-likelihood NLL of $N$ iid samples has gradient
$$\nabla_{\mathbf{w}} \ell = \boldsymbol\Phi^T (\boldsymbol\mu - \mathbf{y})$$
and Hessian
$$\mathbf{H} = \boldsymbol\Phi^T \mathbf{W} \boldsymbol\Phi, \quad \mathbf{W} = \mathrm{diag}(A''(\eta_i)).$$
- **Newton step** $\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{H}^{-1}\nabla\ell$ rearranges to
$$\boxed{\;\mathbf{w}_{t+1} = (\boldsymbol\Phi^T \mathbf{W}_t \boldsymbol\Phi)^{-1} \boldsymbol\Phi^T \mathbf{W}_t\, \mathbf{z}_t\;}$$
with **working response** $\mathbf{z}_t = \boldsymbol\Phi\mathbf{w}_t + \mathbf{W}_t^{-1}(\mathbf{y} - \boldsymbol\mu_t)$.
- **This is just weighted least squares on $(\boldsymbol\Phi, \mathbf{z}_t)$ with weights $\mathbf{W}_t$.** Hence **Iteratively Reweighted Least Squares**.
- **Logistic regression's classical solver is IRLS** — and now we know why: it's the Newton iteration on a Bernoulli GLM.
- **The loop closes:** §2 introduced Newton's method abstractly; §7 shows it's the principled solver for any GLM.
:::

::: {.notes}
This is the densest slide in the unit. Walk through the IRLS reduction explicitly:
1. Start from Newton step on GLM NLL: $\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{H}^{-1}\nabla\ell$.
2. Substitute $\nabla\ell = \boldsymbol\Phi^T(\boldsymbol\mu - \mathbf{y})$ and $\mathbf{H} = \boldsymbol\Phi^T\mathbf{W}\boldsymbol\Phi$.
3. Algebra gives the WLS form on the right with the working response $\mathbf{z}$.
4. WLS is the same closed-form normal-equations solve students saw in Unit 2 — but now reweighted at each iteration.

**Pedagogical payoff:** logistic regression isn't trained by some special algorithm; it's just Newton on the Bernoulli GLM, which happens to have a clean WLS form. Same story for Poisson regression, gamma regression, etc. Every classical regression model is a GLM with a chosen distribution + link, and they all share IRLS as their natural solver.
:::
````

- [ ] **Step 2: Slide-count + render**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -5
```

Expected: slide count = 46. Render succeeds. Visually inspect the unification table renders correctly with mathjax.

- [ ] **Step 3: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §7 GLM unification slides 42-46"
```

---

## Task 15: §8 Wrap-up (slides 47–49)

**Files:**
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd`

- [ ] **Step 1: Replace the §8 marker with the three wrap-up slides**

Use `Edit` to replace `<!-- ===== §8. Wrap-up ===== -->` with:

````markdown
<!-- ===== §8. Wrap-up ===== -->

## Summary: loss → noise → optimizer

| Loss | Noise model / distribution | Canonical optimizer |
|---|---|---|
| **MSE** | Gaussian residuals | OLS closed-form / GD on quadratic / Newton (1 step) |
| **Cross-entropy** (binary) | Bernoulli | IRLS = Newton on Bernoulli GLM |
| **Cross-entropy** (multi-class) | Categorical | IRLS / GD on the softmax NLL |
| **Poisson NLL** | Poisson counts | IRLS = Newton on Poisson GLM |
| **Huber** | Heavy-tailed (between Gaussian & Laplacian) | GD / sub-gradient |
| **MAE** | Laplacian | Sub-gradient / quantile regression |
| **Hinge** | (No probabilistic interpretation — margin-based) | SGD / coordinate descent |

::: {.fragment}
**Three threads — loss, optimizer, model class — meet in the GLM framework.**
:::

## Forward links

::: {.incremental}
- **Unit 4–5** — neural networks: replace $f_{\mathbf{w}}(\mathbf{x}) = \mathbf{w}^T\boldsymbol\phi(\mathbf{x})$ with a learned, parametric $\boldsymbol\phi$.
- **Unit 6** — full first-order optimization: momentum, adaptive methods, saddle dynamics.
- **Unit 7** — bias-variance decomposition with full math.
- **Unit 8** — full probabilistic learning: MLE, MAP, Bayesian posterior, predictive distribution.
- **Unit 12** — uncertainty quantification: Bayesian regression, GPs.
- **ML-PC Unit 2** §26–28 — physical noise → loss derivations (already covered).
:::

## Notebook companion + references

::: {.callout-note icon=false}
### Week 3 notebook: Regression from Scratch — TensileTestDataset
- [Open rendered notebook →](https://eclipse-lab.github.io/Ai4MatLectures/notebooks/MFML/week03_regression_tensile.html)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ECLIPSE-Lab/Ai4MatLectures/blob/main/notebooks/MFML/week03_regression_tensile.ipynb)

Implements OLS, gradient descent, Newton's method, and basis-function regression on a real materials dataset. References to Murphy ch. 6 (linear regression), Bishop ch. 3, McElreath ch. 4–9.
:::
````

- [ ] **Step 2: Slide-count + render — final state of the deck**

```bash
grep -c '^## ' mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -10
```

Expected: slide count = 49. Render succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd
git commit -m "unit 03 redesign: §8 wrap-up slides 47-49"
```

---

## Task 16: Final cleanup — archive missing-topics doc, full render, README sanity check

**Files:**
- Modify or delete: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/missing_topics_analysis.md`
- Modify: `mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.qmd` (any final touch-ups)
- Read: `README.md` (only to verify slug map is unchanged)

- [ ] **Step 1: Re-read the redesigned deck end-to-end**

Use `Read` to scan the entire `01_intro.qmd` from top to bottom. Verify:
- Exactly 49 `## ` headings.
- Section markers (`<!-- ===== §N ===== -->`) all in correct positions.
- All cross-references (Unit 2/6/7/8, ML-PC Unit 2 §26-28) appear in the slide text.
- All five preserved interactives (#2 drag-the-outlier, #3 cross-entropy, #4 polynomial, #5 convex/non-convex) plus #6 Newton-vs-GD and #7 basis explorer are present.
- Bibliography callouts (`[@neuer2024machine]`, `[@bishop2006pattern]`, etc.) still resolve — `ref.bib` is unchanged.

- [ ] **Step 2: Update or delete `missing_topics_analysis.md`**

Per spec §12 acceptance criteria, the missing-topics analysis is now superseded. Replace its content with a single-line note pointing to the redesign:

```bash
cd /home/philipp/projects/_public_presentations/mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization
```

Use `Write` to overwrite `missing_topics_analysis.md` with:

```markdown
# Superseded

The topics flagged in this document (exponential family / GLMs, Newton/IRLS, formal basis-function expansion) have been incorporated into the redesigned `01_intro.qmd` per `redesign_spec.md` and `redesign_plan.md`. This file is kept only for historical reference.
```

(The user can choose to `git rm` it later; we keep it as a stub for now to preserve the historical context.)

- [ ] **Step 3: Final full render + line-count sanity**

```bash
cd mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization && QUARTO_PYTHON=../../.venv/bin/python quarto render 01_intro.qmd 2>&1 | tail -20
wc -l 01_intro.qmd
grep -c '^## ' 01_intro.qmd
```

Expected:
- Render succeeds with no errors.
- Line count is roughly 700–900 (down from 1148).
- `## ` count is exactly 49.

- [ ] **Step 4: Manual smoke test of the rendered HTML**

Open `_site/mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/01_intro.html` (or wherever Quarto renders) in a browser. Click through all 49 slides. Specifically verify:
- All seven interactives respond to inputs.
- Unification table in §7 renders with proper LaTeX.
- Bibliography links (`[@neuer2024machine]` etc.) render as inline citations.

- [ ] **Step 5: README sanity check**

```bash
grep -A2 -B2 'regression_as_loss_minimization' README.md
```

Expected: the slug `03_regression_as_loss_minimization` still appears in the published-URL slug map. No changes needed (folder name unchanged).

- [ ] **Step 6: Commit**

```bash
git add mathematical_foundations_of_ai_and_ml/03_regression_as_loss_minimization/missing_topics_analysis.md
git commit -m "unit 03 redesign: archive missing-topics doc; redesign supersedes it"
```

---

## Plan complete

After Task 16, the unit 03 redesign is complete:
- `01_intro.qmd`: 49 slides, 7 interactives, full GLM thread, full second-order thread, full basis-function thread.
- `redesign_spec.md`: design.
- `redesign_plan.md`: this file (the implementation record).
- `missing_topics_analysis.md`: stubbed, superseded.

If the GitHub-pages CI deploy fails for any unrelated reason, push to a branch and check `.github/workflows/quarto-deploy.yml` logs. Per repo CLAUDE.md, do **not** run `quarto publish` locally — let CI own `gh-pages`.
