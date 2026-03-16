# Unit 4: From Classical Metrics to Learned Representations - Slide Content (50 Slides)

## Part 1: Classical Microstructure Metrics (10 slides)

**Slide 1: Title Slide**
- **Unit 4: From Classical Metrics to Learned Representations**
- Machine Learning in Materials Processing & Characterization (ML-PC)
- Topic: Transitioning from Stereology to Neural Networks

**Slide 2: Stereology Revisited**
- **Quantification**: The bedrock of materials science.
- Standard metrics: Volume fractions ($\phi$), grain size ($d$), surface-to-volume ratio ($S_V$).
- How we've characterized structure for a century.

**Slide 3: Stereological Descriptors**
- [Image showing grain size and phase fraction measurements]
- Scalar features that condense complex 3D structures into single numbers.
- These are our "hand-crafted" features.

**Slide 4: Limits of Hand-Crafted Metrics (I)**
- Loss of information: How much "shape" is lost in an average grain size?
- Topological connectivity and local clustering are often ignored.

**Slide 5: Limits of Hand-Crafted Metrics (II)**
- Feature engineering bottleneck: What if the key descriptor for fatigue hasn't been invented yet?
- We need the model to "learn" the most relevant feature.

**Slide 6: PSPP as a Feature Space**
- Processing $\to$ Structure (Metric $d$) $\to$ Property (Hardness $H$).
- Hall-Petch: $H = H_0 + k \cdot d^{-1/2}$.
- A linear model using a physical descriptor.

**Slide 7: Why Not Just Metrics?**
- Modern materials (high-entropy alloys, nanocomposites) have multi-scale complexities.
- Simple metrics are insufficient to describe the "S" in PSPP completely.

**Slide 8: Transitioning the Paradigm**
- From **Predicting with Descriptors** to **Learning Representations**.
- Moving from manually extracted features to automated feature learning.

**Slide 9: The Target: Property Prediction**
- We want a function $f(\mathbf{S}) \to \mathbf{P}$.
- $f$ must handle high-dimensional spatial data (images) or spectral data.

**Slide 10: Summary: Part 1**
- Classical metrics are valuable but lose topological context.
- We need the ability to extract complex features automatically.

---

## Part 2: The Foundational Neuron (12 slides)

**Slide 11: Biological Inspiration**
- Nature's information processor: The Neuron.
- 86 billion in the human brain, each connected to thousands of others.

**Slide 12: Anatomy of a Neuron (Sandfeld Fig 17.1)**
- [Diagram of Soma, Dendrites, Axons, and Synapses]
- **Dendrites**: Inputs. **Soma**: Aggregation/Decision. **Axon**: Output signal.

**Slide 13: Synaptic Plasticity**
- "Neurons that fire together, wire together" (Hebb's rule).
- The strength of connections (synapses) changes during learning.

**Slide 14: The McCulloch-Pitts (MCP) Neuron (1943)**
- First computational model designed to mimic biological neurons.
- A "Threshold Logic Unit" (TLU).

**Slide 15: MCP Mechanics**
- Inputs $x_i \in \{0, 1\}$.
- Aggregate function: $a = \sum x_i$.
- Activation $\phi$: Unit step function at threshold $\theta$.
- $y = 1$ if $\sum x_i \ge \theta$, else $0$.

**Slide 16: MCP for Boolean Decisions**
- AND Problem: $y=1$ only if $x_1$ AND $x_2$ are $1$.
- OR Problem: $y=1$ if $x_1$ OR $x_2$ are $1$.
- MCP can solve these by tuning $\theta$.

**Slide 17: Limits of MCP**
- Inputs are unweighted. All features are equally important.
- This is not how we learn or perceive.

**Slide 18: The Rosenblatt Perceptron (1958)**
- Introducing **Weights** ($w_i$) and **Bias** ($b$).
- $a = \sum w_i x_i + b$.
- The foundation of supervised learning.

**Slide 19: Perceptron Architecture (Sandfeld Fig 17.2)**
- [Diagram of weights, sum, and step activation]
- Inputs can now be continuous variables (e.g., temperature, pixel intensity).

**Slide 20: Learning in the Perceptron**
- How do we find $w_i$? The Perceptron Learning Rule.
- $w \leftarrow w + \eta (d - y)x$
- $\eta$: Learning rate. $d$: Desired target. $y$: Current output.

**Slide 21: Geometric Interpretation**
- The Perceptron defines a **Hyperplane** (a decision boundary) in the feature space.
- Separating two classes with a straight line/plane.

**Slide 22: Summary: Part 2**
- MCP used fixed thresholds and unweighted sums.
- Perceptron introduced weights and an iterative learning rule.

---

## Part 3: ADALINE and Gradient Descent (8 slides)

**Slide 23: ADALINE (Adaptive Linear Neuron)**
- Widrow-Hoff (1960).
- Key difference: Training uses the *linear* output $a$ before the step function.

**Slide 24: The Delta Rule**
- Minimizing Mean Squared Error (MSE): $J = \frac{1}{2} (d - a)^2$.
- The update is proportional to the continuous error.

**Slide 25: Why ADALINE is Better?**
- Provides a "smoother" landscape for optimization.
- Foundation for modern stochastic gradient descent.

**Slide 26: The Cost Surface J**
- [Visualizing the error as a 3D bowl in parameter space]
- We want to reach the "bottom" of the bowl.

**Slide 27: Introduction to Gradient Descent**
- "Walk downhill" in the weight space.
- $\nabla J = \frac{\partial J}{\partial w_i}$.
- Update rule: $w_{new} = w_{old} - \eta \nabla J$.

**Slide 28: The Learning Rate $\eta$**
- Too large: Oscillations, no convergence.
- Too small: Extremely slow learning.

**Slide 29: Convergence and Stability**
- Finding the global minimum in linear systems.
- Nonlinear systems (Deep Learning) have more complex landscapes.

**Slide 30: Summary: Part 3**
- ADALINE introduced the concept of continuous error minimization.
- Gradient descent is the engine that drives neural network training.

---

## Part 4: The XOR Problem and Multi-Layer Networks (10 slides)

**Slide 31: The Linear Limit**
- Single Perceptrons and ADALINEs can only solve **Linearly Separable** problems.
- They can draw a line, but not a curve or a circle.

**Slide 32: The XOR Problem**
- [Show XOR plot where classes cannot be separated by one line]
- Minsky & Papert (1969) proved a single neuron fails here.

**Slide 33: The "AI Winter"**
- The realization that single-layer networks were limited led to reduced funding in the 70s.
- The solution was hidden in plain sight: **Layers**.

**Slide 34: Multi-Layer Perceptrons (MLP)**
- Combining multiple neurons into layers.
- "Hidden" layers between input and output.

**Slide 35: Hidden Layers as Feature Extractors**
- The first layer might find "edges" or "peaks".
- The next layer combines these into "shapes" or "signatures".

**Slide 36: Solving XOR with Hidden Layers**
- [Visualizing how two hidden neurons can create a non-linear boundary]
- Two lines combine to "sandwich" the XOR outputs.

**Slide 37: Feed-Forward Architecture**
- Information flows in one direction: Input $\to$ Hidden $\to$ Output.
- No cycles (contrary to RNNs).

**Slide 38: Matrix Notation of a Layer**
- $\mathbf{a}^{(1)} = \phi(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)})$
- Powerful and efficient for modern GPU computing.

**Slide 39: Deep Neural Networks (DNN)**
- Simply means "MLP with many hidden layers."
- Each layer adds a level of abstraction.

**Slide 40: Summary: Part 4**
- Layers enable networks to solve non-linearly separable problems.
- Each hidden unit creates a new "learned" feature.

---

## Part 5: Activation Functions (10 slides)

**Slide 41: Why Nonlinearity?**
- If all neurons are linear, a multi-layer network is just a single linear model.
- $\mathbf{W}_2(\mathbf{W}_1 \mathbf{x}) = (\mathbf{W}_2 \mathbf{W}_1)\mathbf{x} = \mathbf{W}_{eff} \mathbf{x}$.
- Activation functions break the linearity.

**Slide 42: The Step Function (Perceptron)**
- $y = 1$ if $a \ge 0$, else $0$.
- Not differentiable at 0. Bad for gradient descent.

**Slide 43: The Sigmoid (Logistic) Function**
- $\sigma(x) = \frac{1}{1 + e^{-x}}$.
- S-shaped, maps to $(0, 1)$.
- Good for interpreting outputs as probabilities.

**Slide 44: The Hyperbolic Tangent (tanh)**
- Maps to $(-1, 1)$.
- Zero-centered, often better for hidden layers than Sigmoid.

**Slide 45: The ReLU (Rectified Linear Unit)**
- $f(x) = \max(0, x)$.
- The "standard" for modern deep learning.
- Fast computation and prevents vanishing gradients.

**Slide 46: Vanishing Gradients**
- Sigmoid/tanh saturate for large $x$, making their derivative nearly zero.
- Learning stops for deep networks. ReLU solves this for $x > 0$.

**Slide 47: Leaky ReLU**
- $f(x) = x$ if $x > 0$, else $\alpha x$.
- Prevents "dying" neurons by allowing a small gradient for negative $x$.

**Slide 48: Softmax for Multi-class**
- Normalizes output to a probability distribution over multiple classes.
- Used in the final layer of classification models.

**Slide 49: Choosing the Right Function**
- Rule of thumb: Start with ReLU for hidden layers, Softmax for classification, Linear/Sigmoid for regression/binary.

**Slide 50: Summary & References**
- From hall-petch metrics to learned representations.
- Single neurons use weights and biases.
- Layers and non-linearities enable complex "deep" learning.
- References: McClarren Ch 5, Neuer Ch 4, Sandfeld Ch 17.
