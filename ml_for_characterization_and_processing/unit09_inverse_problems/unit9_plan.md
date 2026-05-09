# Unit 9: Inverse Problems and Process Maps - Lecture Plan

## Objective
The objective of this unit is to introduce the concept of "Inverse Problems" in materials science and how Physics-Informed Machine Learning (PIML) can solve them. Students will learn to transition from forward modeling (Processing -> Properties) to inverse design (Properties -> Processing) by embedding physical laws into the learning process.

## Slide Structure (approx. 50 slides)

### Part 1: Forward vs. Inverse Problems (8 slides)
- **The Forward Path (3 slides):**
  - Processing $\to$ Structure $\to$ Property.
  - Well-posed, usually unique solution.
- **The Inverse Path (3 slides):**
  - "I want $Y$ yield strength, what $T$ and $v$ do I need?"
  - Often ill-posed: Multiple process paths lead to the same property.
- **Why Inverse is Hard (2 slides):**
  - Non-uniqueness, instability to noise.
  - The "Inverse Design" dream in materials genomics.

### Part 2: Physics-Informed Machine Learning (PIML) (12 slides)
- **Introduction (3 slides):**
  - "Why hide information from an algorithm when it would improve training?" (Neuer 6.1.1).
  - Reducing data requirements via prior knowledge (Sandfeld 19.6).
- **Three Types of Physics Injection (3 slides):**
  - **Observational Bias**: Physics-based data (simulations).
  - **Learning Bias**: Modified loss functions (PINNs).
  - **Inductive Bias**: Specialized architectures (Symmetry-aware).
- **PINNs: Physics-Informed Neural Networks (6 slides):**
  - Architecture: Simple MLP + Automatic Differentiation.
  - The Loss Function: $L_{total} = L_{data} + L_{PDE} + L_{BC}$.
  - Residuals: Forcing the network to satisfy the PDE (e.g., Heat equation).

### Part 3: Automatic Differentiation and Loss Engineering (10 slides)
- **The Engine: Automatic Differentiation (4 slides):**
  - Analytical vs. Numerical vs. Automatic (Neuer 6.3.1).
  - Using `GradientTape` (TensorFlow/PyTorch) to differentiate the network itself.
- **Custom Loss Functions (3 slides):**
  - Encoding constraints: Mass conservation, energy balance.
  - Penalty methods for "illegal" regions of the process space.
- **The Training Loop (3 slides):**
  - Balancing data loss vs. physics loss.
  - Dynamic weighting of loss components.

### Part 4: Inverse Design and Process Maps (12 slides)
- **Mapping the Inverse Space (4 slides):**
  - Using GANs for microstructure generation (Sandfeld 19.5).
  - Reversible Neural Networks (RevNets) for bijective mapping.
- **Process Maps Revisited (4 slides):**
  - From 2D boundaries (Unit 8) to high-dimensional "Instruction Sets."
  - Finding the global optimum via PINNs.
- **Operator Learning (4 slides):**
  - DeepONet: Learning the mathematical operator, not just the solution (Sandfeld 19.6).
  - Trunk vs. Branch networks.

### Part 5: Case Studies and Industrial Application (8 slides)
- **Inverse Heat Transfer in AM (4 slides):**
  - Predicting laser parameters from a desired temperature history.
- **Explainability through Physics (2 slides):**
  - If the physics loss is low, the model is physically consistent (Neuer 6.1.2).
- **Summary & Takeaways (2 slides):**
  - Forward for prediction, Inverse for design.
  - Physics is the best regularizer.

## Materials Sources
- **Sandfeld (2024):** Ch 19.5 (GANs), Ch 19.6 (PINNs, Operator Learning).
- **Neuer (2024):** Ch 6 (Physics-informed learning, Data enrichment, Automatic differentiation).
- **McClarren (2021):** Reference for PDE-based modeling.
