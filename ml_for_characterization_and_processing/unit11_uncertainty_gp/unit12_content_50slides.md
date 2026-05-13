# Unit 12: Uncertainty-Aware Regression & Gaussian Processes (50 Slide Draft)

## Section 1: Intro & The Cost of Uncertainty (Slides 1-5)

1. **Slide 1: Title Slide**
   - Title: Unit 12: Uncertainty-Aware Regression & Gaussian Processes
   - Course: ML for Characterization and Processing (SS26)
   - Prof. Dr. Philipp Pelz
   - Focus: Bayesian ML, GPs, and Uncertainty Quantification (UQ)

2. **Slide 2: Recap: Regression in Materials Science**
   - We've used NNs, SVR, and Linear Regression to predict properties (e.g., strength vs grain size).
   - Most models give a single number (Point Estimate).
   - Problem: What if the model is "confused" because of sparse data?

3. **Slide 3: The Danger of Point Estimates**
   - Imagine a model predicting the yield strength of a new alloy.
   - Point Estimate: 800 MPa.
   - Real Strength: 600 MPa (because of a phase transition not in training data).
   - If we don't know the uncertainty, we might build a bridge that fails.

4. **Slide 4: Trust as a Function of Uncertainty**
   - Trust = Prediction + Confidence.
   - A model that says "I don't know" is more useful than a model that lies confidently.
   - Applications: Quality control, safety-critical processing, experiment planning.

5. **Slide 5: Why UQ in Materials Science?**
   - Experiments are expensive (few data points $\to$ high uncertainty).
   - Process drift (conditions change over time).
   - Sensor noise (aleatoric uncertainty).

## Section 2: Aleatoric vs Epistemic Uncertainty (Slides 6-12)

6. **Slide 6: What is Uncertainty?**
   - It’s not just "error." It’s our lack of perfect knowledge.
   - We distinguish two fundamental types (Neuer 2.2.3).

7. **Slide 7: Aleatoric Uncertainty (Statistical)**
   - "Uncertainty due to randomness."
   - Examples: Measurement noise in a TEM, thermal fluctuations during solidification.
   - Irreducible: More data won't change the noise level of the sensor.

8. **Slide 8: Epistemic Uncertainty (Systemic)**
   - "Uncertainty due to lack of knowledge."
   - Examples: Using a model trained on aluminum to predict copper.
   - Reducible: More training data in the copper regime reduces this uncertainty.

9. **Slide 9: Visualizing the Difference**
   - (Graphic: Model fits with shaded error bars).
   - Aleatoric: Constant width (noise floor).
   - Epistemic: "Balloons" in regions where there is no data.

10. **Slide 10: Why Scientists Care about Epistemic Uncertainty**
    - It tells us WHERE to perform the next experiment (Active Learning).
    - It flags "Out of Distribution" (OOD) cases.

11. **Slide 11: Total Uncertainty**
    - $\sigma_{total}^2 = \sigma_{aleatoric}^2 + \sigma_{epistemic}^2$.
    - Most ML models only capture aleatoric (via the loss function) OR epistemic (via the architecture).

12. **Slide 12: Summary: Taxonomy of Uncertainty**
    - Aleatoric = Noise (Physics of the sensor).
    - Epistemic = Knowledge (Coverage of the training set).

## Section 3: Bayesian Regression (Slides 13-20)

13. **Slide 13: The Bayesian Philosophy**
    - (Ref: Bishop 3.3).
    - Treat weights $\mathbf{w}$ as probability distributions, not single numbers.
    - $p(\mathbf{w} | \text{Data}) \propto p(\text{Data} | \mathbf{w}) p(\mathbf{w})$.

14. **Slide 14: The Role of the Prior $p(\mathbf{w})$**
    - What we believe before seeing data.
    - Example: We know material constants (Young's modulus) must be positive.
    - We can encode physics into the prior!

15. **Slide 15: The Likelihood $p(\text{Data} | \mathbf{w})$**
    - How well do the current weights explain the observed data?
    - Usually assumed to be Gaussian noise.

16. **Slide 16: The Posterior $p(\mathbf{w} | \text{Data})$**
    - Our updated belief after seeing experiments.
    - As we get more data, the distribution of $\mathbf{w}$ narrows (Evidence accumulates).

17. **Slide 17: Predictive Distribution**
    - $p(y^* | x^*, \text{Data}) = \int p(y^* | x^*, \mathbf{w}) p(\mathbf{w} | \text{Data}) d\mathbf{w}$.
    - We average the predictions of ALL possible models, weighted by their probability.

18. **Slide 18: The Result: Predictive Mean and Variance**
    - Mean: The "Best Guess."
    - Variance: The "Error Bar."
    - $var(y^*) = \sigma_{noise}^2 + x^{*T} \mathbf{S}_N x^*$.

19. **Slide 19: Why Bayesian Linear Regression is not enough**
    - Limited to linear combinations of basis functions.
    - Hard to scale to complex high-dimensional manifolds.

20. **Slide 20: Transition to GPs**
    - What if we just defined a distribution over FUNCTIONS instead of weights?

## Section 4: Gaussian Processes (GPs) (Slides 21-35)

21. **Slide 21: Gaussian Processes (GPs): Introduction**
    - (Ref: Rasmussen & Williams 2006).
    - A GP is a collection of random variables, any finite number of which have a joint Gaussian distribution.

22. **Slide 22: From Weights to Function Space**
    - Instead of $y = \mathbf{w}^T \phi(x)$, we say $f(x) \sim \mathcal{GP}(m(x), k(x, x'))$.
    - $m(x)$: Mean function (usually 0).
    - $k(x, x')$: Covariance function (The Kernel).

23. **Slide 23: The Kernel: Defining Similarity**
    - The kernel tells us: "If $x$ and $x'$ are close, then $f(x)$ and $f(x')$ should be similar."
    - This is the soul of the GP.

24. **Slide 24: Popular Kernels: RBF (Gaussian)**
    - $k(x, x') = \sigma^2 \exp(-\frac{|x-x'|^2}{2l^2})$.
    - Smooth, infinitely differentiable. Good for slow-changing material properties.

25. **Slide 25: Popular Kernels: Matern**
    - Less smooth than RBF.
    - Better for real physical data (e.g., surface roughness) where "jumps" occur.

26. **Slide 26: Popular Kernels: Periodic**
    - For data that repeats (e.g., crystal lattices, seasonal process data).

27. **Slide 27: GP Regression: The Math (Conceptual)**
    - Given training points $(X, Y)$, what is $Y^*$ at $X^*$?
    - We solve for the conditional distribution $P(Y^* | X^*, X, Y)$.
    - Pure Linear Algebra (Matrix Inversion).

28. **Slide 28: GP: The "Magic" Visual**
    - (Graphic: Shaded region that pinches at training points).
    - At training points: Uncertainty $\to$ 0 (if no noise).
    - Away from data: Uncertainty grows back to the prior.

29. **Slide 29: GP Advantages**
    - Non-parametric: No fixed number of weights (The data defines the model).
    - Exact uncertainty quantification.
    - Excellent for small datasets (Materials datasets!).

30. **Slide 30: GP Limitations**
    - Complexity $O(N^3)$: Inverting an $N \times N$ matrix is slow for $N > 10,000$.
    - Choosing the kernel is an art (though hyperparameters can be learned).

31. **Slide 31: Hyperparameter Optimization**
    - Learning the "length scale" $l$ and "variance" $\sigma$ from data.
    - Maximizing the Marginal Likelihood (The Evidence Approximation).

32. **Slide 32: Case Study: Alloy Discovery**
    - Predicting the hardness of a 5-element alloy.
    - Use GP to find regions where: (1) Hardness is high AND (2) Uncertainty is high.
    - This is Bayesian Optimization!

33. **Slide 33: Surrogate Modeling for Process Maps**
    - Mapping the "Safe Operating Region" for additive manufacturing.
    - GP provides a probabilistic boundary, not just a line.

34. **Slide 34: Trusting the GP**
    - If a new alloy is very different from the training set, the GP gives a huge error bar.
    - Warning: "Don't trust this prediction."

35. **Slide 35: Summary: Gaussian Processes**
    - The gold standard for UQ in small data regimes.
    - Defined by the Kernel.
    - Mesh-free and non-parametric.

## Section 5: Uncertainty in Deep Learning (Slides 36-42)

36. **Slide 36: When GPs are too slow...**
    - Deep learning on millions of images.
    - How do we get uncertainty from a Neural Network?

37. **Slide 37: The Problem with Standard NNs**
    - Standard Softmax/Linear outputs are "overconfident."
    - They will give a 99% probability to something they've never seen before.

38. **Slide 38: MC Dropout (Gal & Ghahramani 2016)**
    - Dropout is usually only for training.
    - **MC Dropout**: Keep dropout ON during testing.
    - Run the same input through the model 100 times.
    - If the 100 predictions are different, the model is uncertain!

39. **Slide 39: Deep Ensembles (Lakshminarayanan et al. 2017)**
    - Train 5 different models with different starting weights.
    - Average their predictions.
    - Disagreement between models = Epistemic Uncertainty.

40. **Slide 40: Mixture Density Networks (MDNs) (Neuer 6.4.4)**
    - NN predicts the Mean $\mu$ AND Variance $\sigma$ (and weight $\pi$) of a distribution.
    - Useful for multi-modal physics (e.g., A process can result in phase A or phase B).

41. **Slide 41: Stochastic Enrichment (Neuer 6.4.1)**
    - Training on "Cloudy" data: Instead of $x_i$, train on $x_i + \epsilon$.
    - Makes the model robust to sensor noise.

42. **Slide 42: Scaling the Trust**
    - Using MDN or Ensembles to flag "Impossible" process conditions in real-time.

## Section 6: Case Studies & Summary (Slides 43-50)

43. **Slide 43: Case Study: Bayesian Optimization in Materials**
    - Exploration vs. Exploitation.
    - Using the GP uncertainty to search for new superconductors.

44. **Slide 44: Case Study: Process Monitoring with MDNs**
    - Monitoring temperature in a furnace.
    - If the predicted variance $\sigma$ spikes, it indicates sensor failure or process instability.

45. **Slide 45: The "Human-in-the-Loop"**
    - ML suggests an experiment.
    - Scientist checks the uncertainty.
    - If uncertainty is high, the scientist performs the experiment to "teach" the model.

46. **Slide 46: Comparison: NNs vs GPs**
    - Small data: Use GP.
    - Big data / Images: Use NN + MC Dropout.

47. **Slide 47: Limitations of UQ**
    - "Garbage In, Garbage Out": If the prior is wrong, the uncertainty is wrong.
    - Bayesian methods can be computationally heavy.

48. **Slide 48: Integration: Putting it all together**
    - Unit 13 (PINNs) reduced epistemic uncertainty using physics.
    - Unit 12 (GPs) quantifies what's left.

49. **Slide 49: Take-Home Messages**
    - Point estimates are dangerous in engineering.
    - Epistemic uncertainty is your guide for discovery.
    - GPs are powerful, but scale poorly.
    - MC Dropout is a cheap way to get UQ in deep learning.

50. **Slide 50: References & Further Reading**
    - Rasmussen & Williams (2006): Gaussian Processes for Machine Learning.
    - Neuer (2024), Bishop (2006).
    - Next Unit: Automation in Microscopy.
