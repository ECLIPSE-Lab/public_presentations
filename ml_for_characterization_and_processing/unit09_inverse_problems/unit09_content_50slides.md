# Unit 09: Inverse Problems and Process Maps (50 Slide Draft)

## Section 1: Intro & Forward/Inverse Duality (Slides 1-5)

1. **Slide 1: Title Slide**
   - Title: Unit 09: Inverse Problems and Process Maps
   - Course: ML for Characterization and Processing (SS26)
   - Prof. Dr. Philipp Pelz
   - Focus: Inverse Problems, Parameter Discovery, and SINDy

2. **Slide 2: Recap: Forward Modeling**
   - Given a process (e.g., $P, v$ in laser melting), find the resulting microstructure.
   - This is what we usually do: Mapping cause to effect.

3. **Slide 3: What is an Inverse Problem?**
   - Given a desired property (e.g., specific hardness), find the required processing parameters.
   - This is the real engineering goal: Designing the manufacturing route.

4. **Slide 4: The Challenge: Non-uniqueness**
   - One "cause" usually has one "effect" (forward).
   - Multiple "causes" might lead to the *same* "effect" (inverse).
   - This makes the problem "Ill-posed."

5. **Slide 5: Why Inverse Problems Fail in Standard ML**
   - Standard NNs are many-to-one.
   - If we flip the mapping, the network will average the different causes, giving a "blurred," non-physical result.

## Section 2: Ill-posedness & Regularization (Slides 6-12)

6. **Slide 6: Hadamard’s Definition of Well-Posedness**
   - 1. A solution exists.
   - 2. The solution is unique.
   - 3. The solution depends continuously on the data.

7. **Slide 7: When Problems are Ill-Posed**
   - In materials, small changes in noise can lead to wildly different processing parameters.
   - The "butterfly effect" in inversion.

8. **Slide 8: Regularization: The Savior**
   - Using "Prior Knowledge" to pick the most likely cause.
   - L2 (Ridge) favors small, smooth changes.
   - L1 (LASSO) favors sparse, "clean" solutions.

9. **Slide 9: Physics as a Regularizer**
   - We don't just want ANY cause; we want one that obeys the laws of physics.
   - Integration with Unit 13 (PINNs).

10. **Slide 10: Tikhonov Regularization**
    - $\min_{\mathbf{w}} \| \mathbf{y} - f(\mathbf{x}) \|^2 + \lambda \| \Gamma \mathbf{x} \|^2$.

11. **Slide 11: Bayesian View of Inverse Problems**
    - Instead of one solution, we get a posterior distribution (Unit 12).

12. **Slide 12: Summary: Regularization**
    - Inversion is impossible without constraints.

## Section 3: Process Maps & Corridors (Slides 13-20)

13. **Slide 13: What is a Process Map?**
    - A visualization of the relationships between processing inputs and material outcomes.
    - Example: Laser power vs. Scanning speed map for density.

14. **Slide 14: Defining the "Safe Operating Window"**
    - Where do we get high quality with no defects?
    - Identifying regions of "Ball-up," "Keyholing," or "Lack-of-Fusion."

15. **Slide 15: Process Corridors (Neuer Ch 6.5)**
    - Real processes aren't static; they drift (e.g., humidity, sensor fatigue).
    - A "Corridor" describes the envelope of safe trajectories.

16. **Slide 16: Multi-Dimensional Feasibility Regions**
    - In high-dimensional spaces (Composition + Processing), maps are hard to visualize.
    - Using PCA or t-SNE (Unit 10) to project the process space.

17. **Slide 17: ML for Mapping**
    - Using a classifier (SVM/NN) to find the boundary of the safe region.
    - Decision boundaries = Process Window Limits.

18. **Slide 18: Sensitivity Analysis on the Map**
    - Which parameter is the most critical? (Unit 7 Sensitivity).
    - Importance of $P$ vs. $v$.

19. **Slide 19: Anomaly Detection on the Map**
    - If a point lies outside the established process corridor, it's a potential failure.

20. **Slide 20: Summary: Process Maps**
    - The "Navigation Chart" for materials engineering.

## Section 4: Parameter Discovery with PINNs (Slides 21-35)

21. **Slide 21: PINNs for Inverse Problems**
    - (Ref: Sandfeld Ch 19.6; Raissi et al. 2019).
    - Instead of just solving the PDE, we discover the coefficients!

22. **Slide 22: Case Study: Discovering Diffusivity**
    - We measure the concentration profile $c(x, t)$.
    - We don't know the diffusion coefficient $D$.
    - The PINN takes $(x, t)$ and $c$ as input, and learns $D$.

23. **Slide 23: The Inverse Loss Function**
    - $J = J_{data} + J_{pde}(D)$.
    - Both weights $\boldsymbol{\theta}$ AND the parameter $D$ are updated via gradient descent.

24. **Slide 24: Why is this better than fitting?**
    - Handles noisy, irregular experimental data.
    - No need to pre-specify the solution form.

25. **Slide 25: Discovering Variable Parameters**
    - What if $D$ depends on temperature? $D = D_0 \exp(-Q/RT)$.
    - The PINN can learn $D_0$ and $Q$.

26. **Slide 26: Example: Melt Pool Shapes**
    - Using images of melt pools to discover the thermal conductivity of a powder bed.
    - The "Inverse Heat Transfer Problem."

27. **Slide 27: Accuracy and Convergence**
    - Can we really trust the discovered $D$?
    - Sensitivity to measurement noise.

28. **Slide 28: Comparison: Traditional Inverse Methods vs. PINNs**
    - Traditional (Adjoint methods): Fast but complex to implement.
    - PINNs: Easy to implement (Automatic Differentiation), flexible.

29. **Slide 29: Inverse Problems in Tomography**
    - Reconstructing a 3D volume from 2D projections as an inverse problem.
    - Learning the "Inversion Operator" (Unit 13).

30. **Slide 30: Multi-physics Inversion**
    - Coupling mechanical and thermal data to discover constitutive laws.

31. **Slide 31: Handling Noisy Gradients**
    - Physics-informed regularization prevents the model from fitting the noise.

32. **Slide 32: From Data to Digital Twin**
    - The discovered parameters allow for a physical "Digital Twin" of the process.

33. **Slide 33: Summary: PINN Inversion**
    - Making the data "speak" physical laws.

## Section 5: Equation Discovery (SINDy & McClarren) (Slides 36-45)

36. **Slide 36: Symbolic Regression: From Data to Formulas**
    - We have data, but we want the "Law."
    - Can we "discover" $F = ma$?

37. **Slide 37: SINDy (Sparse Identification of Non-linear Dynamics)**
    - (Ref: Brunton et al. 2016).
    - Idea: The true law is sparse in the space of all possible functions.

38. **Slide 38: The Function Library $\boldsymbol{\Theta}$**
    - Constructing a "Library" of candidate functions: $1, x, x^2, \sin(x), \exp(x)$.
    - $\dot{\mathbf{x}} = \boldsymbol{\Theta}(\mathbf{x}) \boldsymbol{\xi}$.

39. **Slide 39: Sparse Regression (LASSO)**
    - Using L1 penalty to force most coefficients in $\boldsymbol{\xi}$ to be zero.
    - Result: The model picks the simplest formula that works.

40. **Slide 40: Case Study: Pendulum (McClarren 2.5)**
    - Discovering the gravitational law from time-series of a swinging mass.
    - The model identifies $g \sin(\theta)$ as the only significant term.

41. **Slide 41: Discovering Constitutive Laws in Materials**
    - Stress-Strain relationships: Discovering the form of hardening.
    - $\sigma = K \epsilon^n$ vs. $\sigma = \sigma_0 + \alpha \sqrt{\rho}$.

42. **Slide 42: Why Sparsity is the "Physical" Prior**
    - Nature is usually simple. Occam's Razor for ML.

43. **Slide 43: SINDy with PINNs**
    - Using PINNs to denoise the data before feeding it into SINDy.

44. **Slide 44: Advantages over "Black-Box" NNs**
    - Interpretability: You get an actual formula.
    - Generalization: Formulas work outside the training box.

45. **Slide 45: Summary: Equation Discovery**
    - The ultimate goal of SciML: Automated Science.

## Section 6: Synthesis & Outlook (Slides 46-50)

46. **Slide 46: Designing Experiments for Inversion**
    - Which data points are the most informative for parameter discovery?
    - Active learning for inverse problems.

47. **Slide 47: Limitations of ML Inversion**
    - Non-identifiability: If the physics is insensitive to a parameter, we can't discover it.
    - Data quality is paramount.

48. **Slide 48: Integration: Unit 9-13 Feedback Loop**
    - Inversion (Unit 9) $\to$ Physics (Unit 13) $\to$ Uncertainty (Unit 12).

49. **Slide 49: Take-Home Messages**
    - Inverse problems are the key to process design.
    - PINNs turn parameter fitting into an optimization problem.
    - SINDy uses sparsity to discover the underlying physical laws.
    - Process corridors provide the safety envelope for industrial use.

50. **Slide 50: References & Further Reading**
    - Brunton et al. (2016): SINDy Paper.
    - McClarren (2021), Sandfeld (2024), Neuer (2024).
    - Next Unit: Generalization & Robustness.
