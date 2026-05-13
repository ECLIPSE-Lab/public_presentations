# Unit 13: Physics-Informed and Constrained ML (50 Slide Draft)

## Section 1: Introduction & The Limits of Pure Data-Driven ML (Slides 1-5)

1. **Slide 1: Title Slide**
   - Title: Unit 13: Physics-Informed and Constrained Machine Learning
   - Course: ML for Characterization and Processing (SS26)
   - Prof. Dr. Philipp Pelz
   - Focus: Scientific ML, PINNs, and Material Processing

2. **Slide 2: Recap: Deep Learning in Materials Science**
   - We’ve seen CNNs for microstructures, RNNs for time-series.
   - Most models are purely data-driven "Black-Boxes."
   - Requirement: Large datasets for good performance.

3. **Slide 3: The "Greediness" of Neural Networks**
   - NNs are powerful function approximators but lack physical intuition.
   - Problem: They optimize the loss function *by any means necessary.*
   - Results can be unphysical: Negative densities, violating thermodynamics, zero viscosity.

4. **Slide 4: Why "Pure" ML Fails in Materials Engineering**
   - Data Scarcity: Generating high-quality materials data is expensive (TEM, XRD, experiments).
   - Extrapolation: Physics models extrapolate; data-driven models often fail outside training bounds.
   - Small shifts in measurement conditions can lead to wild, nonsensical predictions.

5. **Slide 5: The Goal of Scientific Machine Learning (SciML)**
   - Goal: Combine "White-Box" (Physics) with "Black-Box" (ML).
   - "Grey-Box" models: Leverage data AND physical laws.
   - Why? Better generalization, lower data requirements, and trust.

## Section 2: Taxonomy of Bias (Slides 6-12)

6. **Slide 6: How can we include physical knowledge?**
   - Three main routes: Observational, Learning, and Structural Bias.
   - (Ref: Karniadakis et al., Nature Reviews Physics 2021).

7. **Slide 7: Observational Bias (Data Enrichment)**
   - Prior knowledge enters through the data themselves.
   - Enriching small experimental datasets with large-scale simulations.
   - Ensuring datasets reflect symmetries (rotation, reflection) or invariants.

8. **Slide 8: Example: Symmetries in Microstructures**
   - If a material property is isotropic, rotating the image shouldn't change the output.
   - Data Augmentation: Rotating/flipping images encodes the "prior" of isotropy.

9. **Slide 9: Learning Bias (The "Soft" Constraint)**
   - Prior knowledge enters through the Loss Function.
   - Add a "Physics-Residual" term that penalizes non-physical predictions.
   - The "PINN" (Physics-Informed Neural Network) approach.

10. **Slide 10: Structural Bias (The "Hard" Constraint)**
    - Prior knowledge enters through the Network Architecture itself.
    - Example: Enforcing monotonicity by using non-negative weights.
    - Result: The network *cannot* violate the constraint by design.

11. **Slide 11: The Bias Tradeoff**
    - Observational: Easy but requires lots of augmented data.
    - Learning: Powerful but constraints are only "soft" (may still violate).
    - Structural: Hardest to design but provides mathematical guarantees.

12. **Slide 12: Visualizing the Constraint Space**
    - (Graphic: Data-driven solutions vs Physics-feasible region).
    - Goal: Force the NN to stay within the "Physics-Feasible" manifold.

## Section 3: Data Enrichment & Observational Bias (Slides 13-20)

13. **Slide 13: Data Enrichment in Processing**
    - (Ref: Neuer 2024, Ch 6.2).
    - Active design of input data based on physical relationships.
    - Why hide what we already know?

14. **Slide 14: Choosing the Right Preprocessing**
    - FFT for oscillating signals (melt pool vibrations, motor current).
    - Derivatives for sharpening features (strain rates, temperature gradients).
    - Functional transformations (log, exp) to linearize laws.

15. **Slide 15: Case Study: Motor Current (Neuer Example)**
    - Identifying "Good" vs "Bad" operating modes.
    - Raw time series are noisy and high-dimensional.
    - FFT reveals characteristic oscillation frequencies of failure.

16. **Slide 16: FFT-Enriched Networks**
    - Classic Network: Inputs = Time series.
    - PINN (Enriched): Inputs = Time series + FFT components.
    - Result: Improved classification accuracy and explainability.

17. **Slide 17: Automatic Optimization of Data Enrichment**
    - Brute-force: Testing multiple operators ($d/dt$, $FFT$, $\log$).
    - Using a weighted sum of transformations to find the best representation.

18. **Slide 18: Expert Knowledge on Data Objects**
    - Storing meta-information with sensors (Unit, Uncertainty, Transformation rules).
    - Enabling digital twins that "know" their own physical limits.

19. **Slide 19: Statistical Enrichment (Neuer 6.4.1)**
    - Incorporating measurement uncertainty into the training process.
    - Randomly drawing samples from the uncertainty distribution (Gaussian/Laplacian).
    - Increasing robustness against sensor noise.

20. **Slide 20: Summary: Observational Bias**
    - Don't just give raw numbers; give physically relevant features.
    - Uncertainty is a physical reality; include it.

## Section 4: Physics-Informed Neural Networks (PINNs) (Slides 21-35)

21. **Slide 21: PINNs: The Fundamental Idea**
    - (Ref: Sandfeld 19.6; Raissi et al. 2019).
    - Neural Network as a mesh-free function approximator $\mathcal{N}(\boldsymbol{x}, t; \boldsymbol{\theta})$.

22. **Slide 22: The Workflow of a PINN**
    - 1. Define the Governing Equation (e.g., Heat Equation).
    - 2. Define the Network Architecture.
    - 3. Calculate Derivatives using Automatic Differentiation.
    - 4. Minimize the Combined Loss.

23. **Slide 23: Automatic Differentiation (AD)**
    - Not numerical differentiation (finite diff), not symbolic differentiation.
    - Exact calculation of derivatives via the chain rule at the machine level.
    - Key Tool: `tf.GradientTape` or `torch.autograd`.

24. **Slide 24: AD: How it works (The Chain Rule)**
    - Breaking down $y = f(g(h(x)))$ into a graph.
    - Storing partial derivatives during the forward pass.
    - Computing the exact gradient in the backward pass.

25. **Slide 25: Example: Derivatives of a Simple Network**
    - (Graphic: Input $x \to \sigma(w \cdot x + b) \to y$).
    - Computing $dy/dx$ exactly for any point $x$.

26. **Slide 26: The Combined Loss Function**
    - $J(\boldsymbol{\theta}) = \lambda_{data} J_{data} + \lambda_{pde} J_{pde} + \lambda_{bc} J_{bc}$.
    - $J_{data}$: Discrepancy with experimental points.
    - $J_{pde}$: Discrepancy with the physical law.

27. **Slide 27: Case Study: 1D Heat Equation**
    - $\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$.
    - Material processing: Solidification or heat treatment.
    - The PINN learns to satisfy this balance at 10,000 "Collocation Points."

28. **Slide 28: Collocation Points vs. Training Data**
    - Training data: Real measurements (sparse).
    - Collocation points: Random coordinates in space/time (dense).
    - At collocation points, we *only* evaluate the PDE residual.

29. **Slide 29: PINNs as Mesh-Free Solvers**
    - Traditional (FEM/FDM): Require a mesh, can be slow for high dimensions.
    - PINNs: No mesh needed, can handle irregular geometries easily.

30. **Slide 30: Solving Forward Problems**
    - "Given the physics and BCs, find the temperature field $T(x, t)$."
    - PINN acts as a simulator.

31. **Slide 31: Solving Inverse Problems (The Powerhouse)**
    - "Given noisy measurements of $T$, what is the thermal conductivity $\alpha$?"
    - Discovery of material parameters from observation.
    - PINN learns the parameters $\alpha$ *during* training.

32. **Slide 32: Example: Discovering Elasticity**
    - Using strain field measurements (DIC) to identify local stiffness variations.
    - The PINN "fits" the material property to satisfy equilibrium.

33. **Slide 33: PINN Convergence and Stability**
    - PINNs can be hard to train (competitive losses).
    - Choosing the right $\lambda$ weights is crucial (Learning Rate Annealing).

34. **Slide 34: Limitations of PINNs**
    - Often slower than traditional solvers for simple geometries.
    - Not a "Silver Bullet" for all multi-physics problems.

35. **Slide 35: Summary: PINNs**
    - They bridge the gap between simulation and data.
    - Mesh-free, handles inverse problems naturally.

## Section 5: Hard Constraints & Advanced Topics (Slides 36-42)

36. **Slide 36: Hard Constraints: Why?**
    - Soft constraints (loss terms) can be violated if the weight is too low.
    - Some physics MUST be satisfied (e.g., conservation of mass).

37. **Slide 37: Boundary Conditions: Lagaris Substitution**
    - (Ref: Lagaris et al. 1998; Neuer 6.3.3).
    - $g(t) = x_0 + t \mathcal{N}(t)$.
    - At $t=0$, $g(t) = x_0$ regardless of network output.
    - Satisfaction of BCs by architecture design.

38. **Slide 38: Monotonicity Constraints**
    - Some relations are inherently monotonic (e.g., Hardening curves).
    - Enforcing positive derivatives by constraining weight signs or activations.

39. **Slide 39: Dimensional Consistency**
    - Physics models must be dimensionally homogeneous (Length, Time, Mass).
    - Encoding unit awareness into the input/output layers.

40. **Slide 40: Mixture Density Networks (MDNs) (Neuer 6.4.4)**
    - Predicting a probability distribution instead of a single value.
    - Dealing with multi-modal physical states (e.g., phase transitions).

41. **Slide 41: Stochastic Enrichment & Trust**
    - Evaluating model "trust" by perturbing inputs within measurement error.
    - Identifying regions where the model is "confused."

42. **Slide 42: Process Corridors (Neuer 6.5)**
    - Visualizing the envelope of physically allowed trajectories.
    - Detecting anomalies that violate the "physical corridor."

## Section 6: Future Frontiers & Conclusion (Slides 43-50)

43. **Slide 43: Operator Learning: Beyond Functions**
    - (Ref: Sandfeld 19.6; Lu et al. 2021).
    - We don't just want to solve ONE case; we want to learn the general OPERATOR.

44. **Slide 45: DeepONet: The Architecture**
    - Trunk Network (Coordinate space).
    - Branch Network (Initial/Boundary condition space).
    - Product: A simulator that can generalize to NEW geometries/conditions instantly.

45. **Slide 45: DeepONet in Materials Processing**
    - Accelerating multi-scale models (e.g., coupling grain growth with heat flow).
    - Real-time simulation for control loops.

46. **Slide 46: Governing Equation Discovery (SINDy)**
    - Extracting the "hidden" law from raw data.
    - $\dot{x} = \boldsymbol{\Theta}(x, t) \boldsymbol{\xi}$.
    - Using sparse regression to find the simplest law that explains the physics.

47. **Slide 47: Integration: Putting it all together**
    - Use FFT/Expert features (Unit 13).
    - Apply PINN constraints for physics.
    - Quantify uncertainty (Unit 12).

48. **Slide 48: The Role of the Materials Scientist**
    - From "Labeler" to "Teacher."
    - Designing the right constraints is the new "Feature Engineering."

49. **Slide 49: Take-Home Messages**
    - Pure data-driven models are often insufficient for materials science.
    - PINNs bridge data and PDE residuals.
    - AD is the technological engine.
    - Hard constraints provide reliability.

50. **Slide 50: References & Further Reading**
    - Sandfeld (2024), Neuer (2024), McClarren (2021).
    - Next Unit: Integration & Reflection.
