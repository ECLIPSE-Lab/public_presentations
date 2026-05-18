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

- _(≤10 must-know statements — filled when this week's deck is authored)_

---

## Week 3 — Linear algebra & PCA

- _(≤10 must-know statements — filled when this week's deck is authored)_

---

## Week 4 — Regression, gradient descent & honest validation

- _(≤10 must-know statements — filled when this week's deck is authored)_

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
