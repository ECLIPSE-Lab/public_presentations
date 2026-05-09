# Unit 4: From Classical Metrics to Learned Representations - Lecture Plan

## Objective
The objective of this unit is to bridge the gap between traditional microstructure quantification (stereology) and the foundational principles of artificial neural networks. Students will learn how the simple artificial neuron forms the building block for deep learning in materials science.

## Slide Structure (approx. 50 slides)

### Part 1: Classical Microstructure Metrics (10 slides)
- **Stereology Revisited (3 slides):**
  - Volume fractions, grain sizes, aspect ratios.
  - The limits of hand-crafted features: What do they miss?
- **PSPP as a Feature Space (3 slides):**
  - Manually selecting descriptors for property prediction.
  - Case study: Predicting hardness from grain size (Hall-Petch).
- **The Bottleneck (4 slides):**
  - Complexity of morphology vs. simplicity of scalar metrics.
  - Transition to "Learning the Representation."

### Part 2: The Foundational Neuron (12 slides)
- **Biological Inspiration (3 slides):**
  - Soma, Dendrites, Axon, Synapse (Sandfeld Ch 17.1).
  - Synaptic plasticity and learning.
- **The McCulloch-Pitts (MCP) Neuron (1943) (5 slides):**
  - Boolean inputs $\{0, 1\}$, Unweighted sum.
  - Threshold Logic Unit (TLU).
  - Boolean Decisions: AND, OR problems (Sandfeld Ch 17.1.3).
- **The Rosenblatt Perceptron (1958) (4 slides):**
  - Weighted sum $\sum w_i x_i + b$.
  - The Perceptron Learning Rule: $w \leftarrow w + \eta (d-y)x$.
  - Geometric interpretation: The Decision Boundary.

### Part 3: ADALINE and Gradient Descent (8 slides)
- **ADALINE (Adaptive Linear Neuron) (4 slides):**
  - Continuous activation during training.
  - The Delta Rule (Widrow-Hoff).
  - Minimizing Mean Squared Error (MSE).
- **Introduction to Gradient Descent (4 slides):**
  - The Cost Surface $J(w, b)$.
  - Moving against the gradient: $-\nabla J$.
  - Learning rate $\eta$ and convergence.

### Part 4: The XOR Problem and Multi-Layer Networks (10 slides)
- **The Linear Limit (4 slides):**
  - Why a single neuron fails at XOR.
  - Linear separability vs. non-linear complexity.
- **Hidden Layers (3 slides):**
  - Combining neurons to solve non-linear problems.
  - Feature extraction in the hidden space.
- **The Feed-Forward Architecture (3 slides):**
  - Input, Hidden, and Output layers.
  - Matrix notation for layer operations: $\mathbf{a} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$.

### Part 5: Activation Functions (10 slides)
- **The Need for Non-linearity (2 slides):**
  - Linear layers only collapse to a single linear layer.
- **Classic Activations (4 slides):**
  - Step Function (Perceptron).
  - Sigmoid (Logistic): Mapping to $[0, 1]$.
  - Tanh: Mapping to $[-1, 1]$.
- **Modern Activations (4 slides):**
  - ReLU (Rectified Linear Unit): Solving the vanishing gradient.
  - Leaky ReLU and Softmax (for classification).

## Materials Sources
- **Sandfeld (2024):** Ch 17.1-17.3 (Biological neuron, MCP, Perceptron, ADALINE, Boolean problems).
- **Neuer (2024):** Ch 4.1-4.2 (Supervised learning strategy, Cost functions).
- **McClarren (2021):** Ch 2 (Linear models), Ch 5.1 (Simple NN, XOR, Hidden layers).
