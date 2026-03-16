# Unit 5 Plan — Mathematical Foundations of AI & ML

## Audience + constraints
- BSc AI-Material Technology, 5th semester
- Prior knowledge: 2 semesters math, 1 physics, some chemistry
- Assumed: Units 1-4 completed (risk minimization, linear algebra, loss landscapes, neural network architecture)
- Lecture: 90 minutes + 90-minute exercise
- Language: English (German translation later)

## Learning objectives (Unit 5)
By the end of the unit, students can:
1. Derive the chain rule for composite functions and apply it to layered neural network architectures.
2. Trace forward and backward passes through a small network and compute all partial derivatives manually.
3. Explain why backpropagation achieves O(W) computational cost and why this matters for scalability.
4. Diagnose vanishing and exploding gradient problems from the structure of activation functions and weight matrices.
5. Justify the role of ReLU and careful initialization in enabling gradient flow through deep networks.

## Book-aligned content mapping (priority order)
1. Neuer (2024): Ch. 4.5.4 (Training of Neural Networks, backpropagation derivation, delta rule, gradient descent).
2. McClarren (2021): Ch. 5.2–5.3.2 (single-layer and deep network training, chain rule, delta propagation, initialization).
3. Bishop (2006): Ch. 5.3 (error backpropagation, Jacobian matrix, efficiency of backprop).
4. Murphy (2012): Ch. 16.3 (backpropagation and automatic differentiation).
5. Goodfellow et al. (2016): Ch. 6.5 (computational graphs, back-propagation algorithm).

## 90-minute structure
- 0-10 min: Recap Unit 4 (network architecture) and motivate: how does the network actually learn?
- 10-25 min: Chain rule review, computational graphs, forward pass mechanics
- 25-45 min: Backpropagation algorithm derivation (output layer, hidden layers, delta recursion)
- 45-60 min: Gradient flow analysis (vanishing/exploding gradients, role of activation derivatives)
- 60-75 min: ReLU revolution, initialization strategies, Jacobian perspective
- 75-85 min: Materials/engineering training dynamics examples and practical diagnostics
- 85-90 min: Summary + exercise handoff

## Exercise (90 min)
- Manual backpropagation: compute all gradients for a 2-layer network with 2 hidden units (pen-and-paper then NumPy)
- Implement forward and backward pass in NumPy from scratch (no autograd)
- Visualize gradient magnitudes across layers for sigmoid vs ReLU activation
- Observe vanishing gradient effect by training a 5-layer network with sigmoid vs ReLU
- Bonus: implement gradient clipping and compare training stability

## Assessment alignment
- Written exam prep continues with explicit derivation and notation practice.
- Every unit introduces 3-5 exam-relevant "must-know" statements.
- Backpropagation derivation is a classic exam question format.

## Slide production notes
- Style: follow data_science_for_em revealjs pattern
- Include structured recap + glossary slide
- Add citation placeholders for each reused concept/figure
