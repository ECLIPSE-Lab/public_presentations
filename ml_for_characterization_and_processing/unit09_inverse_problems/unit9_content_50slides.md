# Unit 9: Inverse Problems and Process Maps - Slide Content (50 Slides)

## Part 1: Forward vs. Inverse Problems (8 slides)

**Slide 1: Title Slide**
- **Unit 9: Inverse Problems and Process Maps**
- Machine Learning in Materials Processing & Characterization (ML-PC)
- Topic: Physics-Informed Learning and Inverse Design

**Slide 2: The Forward Path**
- **Processing $\to$ Structure $\to$ Property.**
- Classic Prediction: If I use $P=200W$, what is the grain size?
- Well-posed: A unique set of inputs leads to a unique physical state.

**Slide 3: The Engineering Question (The Inverse Problem)**
- **Property $\to$ Processing.**
- "I need a yield strength of 500 MPa. How should I print/cast it?"
- This is what engineers actually care about!

**Slide 4: Why Inverse is Hard (I): Non-Uniqueness**
- Many processing paths can lead to the same property.
- Multiple $(P, v)$ combinations might give the same microstructure.
- How do we choose the "best" one?

**Slide 5: Why Inverse is Hard (II): Sensitivity**
- Small changes in desired properties can require massive jumps in processing.
- Inverse mappings are often unstable and discontinuous.

**Slide 6: Traditional Approach: Trial-and-Error**
- Iterative experiments $\to$ Refine the process $\to$ Test properties.
- Extremely slow and expensive for complex materials.

**Slide 7: The "Inverse Design" Dream**
- A model where we input **Desired Performance** and the output is the **Manufacturing Recipe**.
- Moving from "discovery by accident" to "discovery by design."

**Slide 8: Summary: Part 1**
- Forward modeling is easy but insufficient.
- Inverse modeling is hard but essential for design.

---

## Part 2: Physics-Informed Machine Learning (PIML) (12 slides)

**Slide 9: Introduction to PIML**
- "Why hide information from an algorithm when it would improve its training and performance?" (Neuer 6.1.1).
- PIML = Data + Scientific Law.

**Slide 10: Reducing the Data Burden**
- (Sandfeld 19.6): Physical constraints reduce the "Solution Space."
- The network no longer has to learn the law of conservation of energy from scratch.
- Enables learning from tiny datasets.

**Slide 11: Three Types of Physics Injection (I)**
- **Observational Bias**: Training on simulation data (FEA, Phase Field).
- The network "observes" physics-consistent samples.

**Slide 12: Three Types of Physics Injection (II)**
- **Learning Bias**: Forcing the network to satisfy equations during training.
- Penalizing "unphysical" predictions in the loss function.

**Slide 13: Three Types of Physics Injection (III)**
- **Inductive Bias**: Specialized architectures.
- Example: A network where the output is guaranteed to be symmetric by design.

**Slide 14: PINNs: Physics-Informed Neural Networks**
- Raissi et al. (2017).
- A regular MLP that "knows" its own partial differential equations (PDEs).

**Slide 15: The PINN Loss Function**
- $L_{total} = L_{data} + L_{physics}$.
- $L_{data}$: How well does it fit the experimental points?
- $L_{physics}$: Does it satisfy the PDE at every sampled point?

**Slide 16: The Residual Loss**
- If the Heat Equation is $\frac{\partial T}{\partial t} = \alpha \nabla^2 T$, then $R = \frac{\partial T}{\partial t} - \alpha \nabla^2 T$.
- We want $R = 0$ everywhere in the process volume.

**Slide 17: Collocation Points**
- Unlike standard ML, we don't need labels at every point for the physics loss.
- We can sample random points $(\mathbf{x}, t)$ and check the PDE residual there.

**Slide 18: PINNs for Forward Problems**
- Solving a PDE without a mesh (meshless solver).
- Faster than FEA for certain parameterized problems.

**Slide 19: PINNs for Inverse Problems**
- Using measurements of $T(\mathbf{x}, t)$ to find the unknown thermal conductivity $\alpha$.
- The physics loss "drags" the parameters toward their physical truth.

**Slide 20: Summary: Part 2**
- PINNs turn physical laws into regularizers.
- They bridge the gap between black-box ML and white-box simulation.

---

## Part 3: Automatic Differentiation and Loss Engineering (10 slides)

**Slide 21: How does a PINN "know" a PDE?**
- To compute $\frac{\partial T}{\partial x}$, we need the derivative of the network output with respect to the input.
- We use **Automatic Differentiation (AD)**.

**Slide 22: Analytic vs. Numerical vs. AD (Neuer 6.3.1)**
- Numerical: $\frac{f(x+h)-f(x)}{h}$ (Approximation, sensitive to $h$).
- AD: Exact gradients computed via the chain rule during the forward pass.

**Slide 23: GradientTape in Action**
- [Code snippet showing `with tf.GradientTape() as tape:`]
- We can compute second and third-order derivatives effortlessly.

**Slide 24: Engineering the Loss Function**
- Not just one term!
- $L = w_1 L_{experiment} + w_2 L_{PDE} + w_3 L_{Boundary\_Condition}$.
- Choosing the weights $w_i$ is a "hyper-inverse" problem.

**Slide 25: The Dynamic Loss Weighting**
- If the physics loss is 100x larger than the data loss, the model will ignore the data.
- We must balance them dynamically during training.

**Slide 26: Encoding Constraints: Energy Balance**
- Penalizing the model if the total energy in the system changes unexpectedly.
- Forcing the model to "follow the law."

**Slide 27: Penalty Methods**
- Penalizing values outside physical ranges (e.g., negative concentrations or absolute temperatures $< 0 K$).

**Slide 28: Convergence Challenges**
- PINN loss landscapes are often very complex and full of local minima.
- Requires careful initialization and often higher-order optimizers (like L-BFGS).

**Slide 29: Integration of Data Enrichment (Neuer 6.2)**
- Providing derived features (gradients, integrals) as inputs to simplify the task.

**Slide 30: Summary: Part 3**
- Automatic differentiation is the engine of PINNs.
- Loss engineering is the art of teaching physics to a network.

---

## Part 4: Inverse Design and Process Maps (12 slides)

**Slide 31: Mapping the Inverse Space**
- We want a map where $P_{desired} \to \text{Instructions}$.
- How to handle the non-uniqueness?

**Slide 32: Generative Models for Design**
- Using **GANs** to generate valid microstructures (Sandfeld 19.5).
- Input a "target descriptor" $\to$ Generator creates a realistic micrograph.

**Slide 33: The Latent Space of Microstructures**
- [Diagram of an Autoencoder mapping images to a low-dim space]
- Searching the latent space for the point that yields the target property.

**Slide 34: Reversible Neural Networks (RevNets)**
- Architectures that are bijective by design.
- Forward: $X \to Y$. Inverse: $Y \to X$ using the same weights!
- Perfect for ensuring consistency between prediction and design.

**Slide 35: Process Maps Revisited**
- (Ref. Unit 8) From 2D boundaries to high-dimensional instruction sets.
- Using PINNs to find the "Path of Least Resistance" in the process map.

**Slide 36: Optimization-based Inverse Design**
- 1. Train a surrogate $P = f(S, Proc)$.
- 2. Fix $P$ and $S$.
- 3. Use Gradient Descent on the *inputs* (Proc) to find the best recipe.

**Slide 37: Operator Learning (Sandfeld 19.6)**
- Learning the **Map between function spaces**, not just points.
- "Give me a boundary condition function, I give you the stress field function."

**Slide 38: DeepONet Architecture**
- **Branch Network**: Processes the input function (e.g., sensor log).
- **Trunk Network**: Processes the coordinates $(x, y, z, t)$.
- Their dot product yields the solution.

**Slide 39: Trunk vs. Branch Intuition**
- The trunk learns the **Physics of Space**.
- The branch learns the **Physics of the Specific Case**.

**Slide 40: Advantages of Operator Learning**
- Once trained, it's near-instantaneous.
- Generalizes across different boundary conditions without retraining (unlike basic PINNs).

**Slide 41: Inverse Operator Design**
- Finding the input function (e.g., laser path) that minimizes the residual of a target state.

**Slide 42: Summary: Part 4**
- Inverse design requires moving from points to distributions or functions.
- Generative models and Operator learning are the key tools.

---

## Part 5: Case Studies and Industrial Application (8 slides)

**Slide 43: Inverse Heat Transfer in 3D Printing**
- Task: Control the cooling rate $dT/dt$ to ensure a specific phase transformation.
- Model: PINN predicts the required Laser Power $P(t)$ for a target $T(t)$.

**Slide 44: Discovering Unknown Parameters**
- Using experimental micrographs to "back-calculate" the surface tension or diffusion coefficients.
- The model learns the physics and the material properties simultaneously.

**Slide 45: Explainability through Physics (Neuer 6.1.2)**
- If the model satisfies the Heat Equation, we know *why* it predicts a certain temperature.
- Trust is built via physical consistency, not just test accuracy.

**Slide 46: Industrial Process Corridors (Neuer 6.4)**
- Using PIML to define not just a safe window, but an **optimal corridor** for energy efficiency.

**Slide 47: The "Physics-Augmented" Lab**
- Real-time PINN-based monitoring.
- Correcting process drift *before* the property target is missed.

**Slide 48: Limits of PINNs**
- Can they handle phase transitions? (Discontinuities are hard for MLPs).
- Scaling to full industrial parts (Geometric complexity).

**Slide 49: Summary & Top Takeaways**
1. **Inverse Design** is the "Holy Grail" of materials informatics.
2. **PIML** embeds laws into the loss function via Automatic Differentiation.
3. **PINNs** allow learning with minimal data and provide physical grounding.
4. **Operator Learning** enables lightning-fast inverse mapping for complex domains.

**Slide 50: References & Exercise**
- Sandfeld Ch 19, Neuer Ch 6.
- Exercise: Build a 1D PINN to find the diffusion coefficient $D$ from a concentration profile.
