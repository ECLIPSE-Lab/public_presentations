# MFML Unit 5 — 50-Slide Content Pack (English)

## Unit theme
**How neural networks learn: backpropagation and gradient flow**

## Core source mapping (book-priority aligned)
- **Neuer (2024)**: Ch. 4.5.4 (training of neural networks, cost function, chain rule derivation, delta rule, backpropagation).
- **McClarren (2021)**: Ch. 5.2–5.3.2 (single-layer training, derivative computation, deep network training, delta propagation, initialization).
- **Bishop (2006)**: Ch. 5.3 (error backpropagation algorithm, Jacobian matrix, efficiency analysis).
- **Murphy (2012)**: Ch. 16.3 (backpropagation, automatic differentiation, computational graphs).
- **Goodfellow et al. (2016)**: Ch. 6.5 (computational graphs, back-propagation and other differentiation algorithms).

---

## Slide-by-slide content (target: 50)

### Block A — Orientation and motivation (Slides 1-8)
1. **Title + Unit 5 positioning**
   - Backpropagation as the engine that makes neural network training feasible.
2. **Recap: what we know so far**
   - Network architecture (Unit 4), loss functions (Unit 3), gradient descent (Unit 1).
3. **The central question of Unit 5**
   - How do we compute gradients efficiently for networks with millions of parameters?
4. **Learning outcomes for Unit 5**
   - Chain rule mastery, forward/backward pass, gradient pathologies, ReLU motivation.
5. **Why not just finite differences?**
   - O(W) evaluations per gradient vs O(1) with backprop; infeasible for large networks.
6. **Historical context**
   - Rumelhart, Hinton & Williams (1986); rediscovery and impact on deep learning.
7. **Computational graph intuition**
   - Any computation as a directed acyclic graph of elementary operations.
8. **Roadmap of today's 90 min**
   - Chain rule, forward/backward pass, gradient flow analysis, practical implications.

### Block B — Chain rule and forward pass (Slides 9-17)
9. **Univariate chain rule review**
   - If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x).
10. **Multivariate chain rule**
    - Partial derivatives and summation over intermediate variables.
11. **Chain rule in matrix form**
    - Jacobian matrices and the chain of matrix multiplications.
12. **Forward pass: layer-by-layer computation**
    - Input -> linear transform -> activation -> next layer; storing intermediate values.
13. **Forward pass notation**
    - z^(l) = W^(l) a^(l-1) + b^(l), a^(l) = sigma(z^(l)).
14. **Worked example: 2-layer network forward pass**
    - Concrete numeric example with 2 inputs, 2 hidden units, 1 output.
15. **Why store intermediate activations?**
    - Required for efficient backward pass; memory-compute tradeoff.
16. **Cost function at the output**
    - J = (1/N) sum L(y_hat, y); everything before is a composition of differentiable functions.
17. **Mini-checkpoint question**
    - "How many multiplications does the forward pass require for an L-layer network?"

### Block C — Backpropagation algorithm (Slides 18-30)
18. **The key insight: reverse-mode differentiation**
    - Compute all partial derivatives in one backward sweep through the graph.
19. **Output layer gradient**
    - dJ/dw_ok = delta_o * z_k where delta_o = (y_hat - y) * o'(u_o).
20. **Hidden layer gradient via chain rule**
    - dJ/dw_kj = delta_k * x_j where delta_k = delta_o * w_ok * sigma'(u_k).
21. **The delta recursion (general form)**
    - delta_i^(l) = sigma'(z_i^(l)) * sum_j w_ji^(l+1) * delta_j^(l+1).
22. **Bias gradient**
    - dJ/db_k = delta_k; bias gradients are delta values directly.
23. **Backpropagation algorithm: pseudocode**
    - Step-by-step: forward pass, compute output delta, propagate deltas backward, accumulate gradients.
24. **Worked example: backward pass**
    - Same 2-layer network from slide 14; compute all deltas and weight updates numerically.
25. **Weight update rule**
    - w <- w - eta * dJ/dw; connection to gradient descent from Unit 1.
26. **Batch vs stochastic gradient computation**
    - Full batch, mini-batch, SGD; gradient estimates and noise tradeoffs.
27. **Computational cost analysis**
    - Forward pass: O(W); backward pass: O(W); total gradient: O(W) not O(W^2).
28. **Backprop vs finite differences**
    - Backprop: one forward + one backward pass; finite differences: W+1 forward passes.
29. **Multiple outputs and loss functions**
    - Cross-entropy, softmax output; delta_o changes but structure remains identical.
30. **Recap: the backpropagation pipeline**
    - Forward -> store -> output delta -> backward delta propagation -> gradient accumulation -> update.

### Block D — Gradient flow and stability (Slides 31-40)
31. **Gradient flow through deep networks**
    - Gradient at layer l is product of L-l Jacobian terms; depth amplifies or attenuates signal.
32. **Vanishing gradients explained**
    - Sigmoid derivative max 0.25; product of many factors < 1 shrinks gradient exponentially.
33. **Vanishing gradients: consequences**
    - Early layers stop learning; network behaves as if shallow; training stalls.
34. **Exploding gradients explained**
    - Large weight norms cause gradient products to grow exponentially; loss diverges.
35. **Exploding gradients: mitigation**
    - Gradient clipping, careful initialization, architectural choices.
36. **ReLU and gradient flow**
    - ReLU derivative is 0 or 1; no saturation for positive inputs; enables deep training.
37. **ReLU variants and dead neurons**
    - Leaky ReLU, ELU, GELU; preventing zero-gradient regions.
38. **Weight initialization strategies**
    - Xavier/Glorot initialization; He initialization for ReLU; variance preservation across layers.
39. **The Jacobian matrix perspective**
    - J = da^(l)/da^(l-1); singular values of Jacobian control gradient magnitude.
40. **Checkpoint MCQ slide**
    - "A 10-layer network with sigmoid activation and initial weights ~1: vanishing or exploding?"

### Block E — Materials translation + exercise bridge (Slides 41-50)
41. **Materials example 1: training a property-prediction network**
    - Predicting tensile strength from composition and processing; monitoring gradient norms during training.
42. **Materials example 2: deep vs shallow for spectral classification**
    - Why a 10-layer sigmoid network fails where a 4-layer ReLU network succeeds.
43. **Materials example 3: process optimization gradient pathologies**
    - Multi-step manufacturing pipeline as computational graph; gradient flow through process stages.
44. **Practical diagnostic: gradient norm plots**
    - Monitoring per-layer gradient statistics to detect training pathologies early.
45. **Automatic differentiation vs manual backprop**
    - Modern frameworks (PyTorch, JAX) automate backprop; understanding internals still essential.
46. **Lecture-essential vs exercise content split**
    - Lecture: derivations, delta recursion, gradient flow theory. Exercise: implementation, visualization, debugging.
47. **Exercise setup: manual backprop for a 2-layer network**
    - Pen-and-paper derivation then NumPy implementation of forward and backward pass.
48. **Exercise extension: sigmoid vs ReLU gradient visualization**
    - Train identical architectures with different activations; plot gradient magnitudes per layer.
49. **Exam-aligned summary: 10 must-know statements**
    1. Backpropagation is efficient application of the chain rule in reverse order.
    2. Forward pass computes and stores all intermediate activations.
    3. Backward pass propagates delta signals from output to input.
    4. Computational cost of backprop is O(W), same order as forward pass.
    5. Vanishing gradients arise from repeated multiplication by values < 1.
    6. Exploding gradients arise from repeated multiplication by values > 1.
    7. ReLU enables gradient flow by providing constant derivative of 1 for positive inputs.
    8. Xavier/He initialization preserves variance across layers.
    9. The Jacobian matrix describes sensitivity of one layer to perturbations in the previous.
    10. Gradient diagnostics (norm plots, loss curves) are essential engineering practice.
50. **References + reading assignment for next unit**
    - Required: Neuer Ch. 4.5.4-4.5.5, McClarren Ch. 5.2-5.3.2.
    - Optional: Bishop Ch. 5.3, Goodfellow Ch. 6.5.
    - Next unit: optimization algorithms and learning rate schedules (SGD, momentum, Adam).

---

## Reusable equations (for slide math boxes)
- Chain rule (univariate): \\(\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}\\)
- Chain rule (multivariate): \\(\frac{\partial J}{\partial w_{ik}} = \frac{\partial J}{\partial f}\frac{\partial f}{\partial \text{net}_k}\frac{\partial \text{net}_k}{\partial w_{ik}}\\)
- Forward pass: \\(z^{(\ell)} = W^{(\ell)}a^{(\ell-1)} + b^{(\ell)}, \quad a^{(\ell)} = \sigma(z^{(\ell)})\\)
- Delta (output): \\(\delta_o = (\hat{y} - y)\frac{do}{du_o}\\)
- Delta (hidden): \\(\delta_k = \delta_o\, w_{ok}\,\frac{d\sigma}{du_k}\\)
- Delta recursion (deep): \\(\delta_i^{(\ell)} = \sigma'(z_i^{(\ell)})\sum_j w_{ji}^{(\ell\to\ell+1)}\delta_j^{(\ell+1)}\\)
- Weight update: \\(w \leftarrow w - \eta\,\frac{\partial J}{\partial w}\\)
- Gradient flow product: \\(\frac{\partial J}{\partial a^{(\ell)}} = \prod_{m=\ell}^{L}\left(\text{diag}(\sigma'(z^{(m)}))\,W^{(m+1)}\right)\frac{\partial J}{\partial a^{(L+1)}}\\)

## Lecture vs exercise allocation (strict)
- **Lecture**: chain rule derivation, backpropagation algorithm, gradient flow theory, vanishing/exploding analysis, Jacobian interpretation.
- **Exercise**: manual gradient computation, NumPy forward/backward implementation, gradient visualization, activation function comparison.

## Reading assignment after Unit 5
- Neuer Ch. 4.5.4–4.5.5
- McClarren Ch. 5.2–5.3.2
- Bishop Ch. 5.3 (skim)
- Goodfellow Ch. 6.5 (conceptual)
