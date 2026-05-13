# Unit 8: Generalization, Robustness, and Process Windows - Slide Content (50 Slides)

## Part 1: Generalization and the Bias-Variance Tradeoff (10 slides)

**Slide 1: Title Slide**
- **Unit 8: Generalization, Robustness, and Process Windows**
- Machine Learning in Materials Processing & Characterization (ML-PC)
- Topic: Reliability, Evaluation, and Safe Operation regions in ML

**Slide 2: What is the Real Error?**
- We don't care how well the model performs on the **Training Data**.
- We care how it performs on the **Unseen Test Data**.
- **Generalization**: The ability of a model to apply its learned knowledge to new cases.

**Slide 3: Training vs. Testing Error (Sandfeld 12.2)**
- [Graph showing Training error monotonically decreasing while Testing error hits a minimum and then rises]
- Training error measures **fitting**.
- Testing error measures **generalization**.

**Slide 4: Underfitting: High Bias**
- Model is too simple to capture the underlying pattern.
- Example: Using a straight line for a quadratic relationship.
- High error on both training and test data.

**Slide 5: Overfitting: High Variance**
- Model is too complex and "memorizes" the noise in the training data.
- Example: A 10th-degree polynomial for 5 data points.
- Low training error, but high (and erratic) test error.

**Slide 6: The Bias-Variance Tradeoff**
- [The classic U-curve graph]
- Total Error = Bias² + Variance + Irreducible Noise.
- The goal: Find the "sweet spot" in model complexity.

**Slide 7: Why Overfitting Happens in Materials ML**
- High dimensionality + Small datasets (Ref. Unit 6).
- CNNs with millions of parameters trained on hundreds of images.
- The model learns instrument-specific artifacts rather than physics.

**Slide 8: Parsimony and Occam's Razor**
- "Among competing hypotheses, the one with the fewest assumptions should be selected."
- In ML: Prefer the simplest model that explains the data well.

**Slide 9: Regularization: Keeping it Simple**
- Adding a penalty for large weights ($L_1$, $L_2$).
- Forces the model to use only the most important features.

**Slide 10: Summary: Part 1**
- Generalization is the benchmark of success.
- Beware the "too good to be true" training accuracy.

---

## Part 2: Robustness and Noise (10 slides)

**Slide 11: Defining Robustness**
- A robust model produces stable outputs under perturbations.
- Stable predictions despite noisy inputs or shifts in the environment.

**Slide 12: Aleatory Uncertainty (Measurement Noise)**
- Random noise in the detector (Poisson/Gaussian).
- Robust models should be insensitive to these pixel-level fluctuations.

**Slide 13: Epistemic Uncertainty (Missing Knowledge)**
- The model hasn't seen this part of the material space yet.
- Robust models should say "I don't know" rather than making a wild guess.

**Slide 14: Distribution Shift**
- **Training**: Data from Microscope A.
- **Testing**: Data from Microscope B.
- Are the features (e.g., grain edges) the same? Or does the model depend on Microscope A's specific contrast?

**Slide 15: Outliers: The Robustness Test**
- [Comparison of Ordinary Least Squares (LSQ) vs. Mean Absolute Error (MAE)]
- LSQ is sensitive to outliers (errors are squared).
- Robust regression (Huber loss) ignores the extremes.

**Slide 16: Data Cleaning vs. Model Robustness**
- Should we delete outliers or use a model that handles them?
- **Scientific Caveat**: Is it an outlier (error) or a rare physical event (discovery)?

**Slide 17: Adversarial Examples in Microstructures**
- Can changing 1% of pixels (unnoticeable to humans) flip the model's prediction?
- Critical for automated quality control in manufacturing.

**Slide 18: Making CNNs Robust**
- Using Augmentation (Ref. Unit 6) to "teach" the model noise-invariance.
- Adding Gaussian noise during training.

**Slide 19: Physical Continuity**
- If I change the temperature by 1°C, the predicted yield strength should not jump by 100 MPa.
- Robustness as a "smoothness" requirement.

**Slide 20: Summary: Part 2**
- Robustness ensures that the model doesn't "break" when the lab environment changes.

---

## Part 3: Cross-Validation and Hyperparameter Tuning (12 slides)

**Slide 21: Cross-Validation (CV)**
- Reusing the limited data to get a more robust estimate of performance.
- Not just one split, but many.

**Slide 22: K-Fold Cross-Validation (Sandfeld 16.2)**
- [Diagram of 5-Fold CV: Data split into 5 parts, each part acts as test set once]
- Reduces the risk of a "lucky" or "unlucky" single split.

**Slide 23: Stratified K-Fold**
- Ensures each fold has the same proportion of classes as the original data.
- Crucial for imbalanced materials datasets (e.g., rare defects).

**Slide 24: Group-based Splitting (Ref. Unit 3)**
- **Never** put images from the same specimen in both training and testing folds.
- High correlation within one sample leads to "Leakage" and inflated accuracy.

**Slide 25: Parameters vs. Hyperparameters**
- **Parameters**: Weights ($w$) and biases ($b$). Optimized by the algorithm.
- **Hyperparameters**: Learning rate ($\eta$), number of layers, batch size. Set by the human.

**Slide 26: Hyperparameter Tuning**
- The search for the optimal "meta-settings" for the model.
- "Training the trainer."

**Slide 27: Grid Search**
- Brute-force: Trying every combination on a grid.
- [Visualizing a 2D grid of Learning Rate vs. Number of Layers]
- Exponentially expensive as the number of hyperparameters grows.

**Slide 28: Random Search**
- Sampling the space randomly.
- Often finds better solutions faster than grid search (Bergstra & Bengio, 2012).
- Not all hyperparameters are equally important.

**Slide 29: Bayesian Optimization**
- Building a "surrogate" model of the model performance.
- Intelligently choosing the next set of hyperparameters to try.
- Minimizes the number of expensive training runs.

**Slide 30: Automated ML (AutoML)**
- Using algorithms (like Hyperopt or Optuna) to handle the tuning automatically.
- Letting the computer find the best architecture.

**Slide 31: Practical Tip: The Validation Set**
- Train set (for weights).
- **Validation set** (for hyperparameter tuning).
- Test set (held back for final evaluation ONLY).

**Slide 32: Summary: Part 3**
- CV provides a reliable performance estimate.
- Hyperparameter tuning is the key to maximizing model potential.

---

## Part 4: Defining Process Windows (10 slides)

**Slide 33: What is a Process Window?**
- The operating region where the product meets all specifications.
- Example: Casting temperature vs. Cooling rate.

**Slide 34: The Traditional Process Map**
- [Example of an Ashby-style or AM process map (Power vs. Velocity)]
- Usually defined by trial-and-error or simple physics models.

**Slide 35: ML-driven Process Windows**
- Using a classifier to predict "Pass" or "Fail" for every point in the process space.
- Drawing the boundary where the model is 50/50.

**Slide 36: Probabilistic Windows**
- Instead of a sharp line, show probability contours.
- Where is the model 95% certain that the process is safe?
- Mapping uncertainty to "Safety Factors."

**Slide 37: Case Study: 3D Printing (AM)**
- Laser Power ($P$) vs. Scan Speed ($v$).
- ML identifies the "Golden Zone" between Lack-of-Fusion and Keyholing.

**Slide 38: Multi-Property Process Windows**
- Find the window that satisfies BOTH high strength AND high ductility.
- The intersection of multiple ML model "Safe Zones."

**Slide 39: Visualizing High-D Windows**
- When we have 5-10 process variables, we can't draw a simple 2D map.
- Using dimensionality reduction (PCA/t-SNE) or "Slices" through the window.

**Slide 40: Real-time Window Monitoring**
- If the current process state moves toward the edge of the window, the model triggers an alert.
- Autonomous process control.

**Slide 41: Summary: Part 4**
- ML transforms data into actionable "Safe Zones" for manufacturing.

---

## Part 5: Sensitivity Analysis (8 slides)

**Slide 42: What Drives the Model?**
- Sensitivity Analysis: Quantifying how much the output changes when inputs change.
- "Is the model looking at the right thing?"

**Slide 43: Local Sensitivity Analysis**
- Perturbing one variable slightly at a specific point.
- The "Gradient" of the model output.

**Slide 44: Global Sensitivity Analysis**
- How do variables interact across the entire process space?
- Sobol indices: Decomposing the variance of the output.

**Slide 45: Permutation Importance**
- Shuffle one feature's values and see how much accuracy drops.
- If accuracy plummets, that feature was critical.

**Slide 46: Feature Importance in CNNs**
- Saliency Maps / Grad-CAM.
- [Image of a microstructure with "important" areas highlighted in red]
- Is the model looking at the grain or the scale bar?

**Slide 47: SHAP (Shapley Additive Explanations)**
- A game-theoretic approach to explaining predictions.
- Fairly distributing the "credit" for a prediction among the input features.

**Slide 48: Stability vs. Sensitivity**
- High sensitivity to physical parameters is good (it means the model captures the physics).
- High sensitivity to noise is bad (it means the model is not robust).

**Slide 49: Summary: Top Takeaways**
1. **Generalization** is the only accuracy that matters.
2. **Robustness** protects against real-world "dirt."
3. **Cross-Validation** ensures we aren't fooled by luck.
4. **Process Windows** bridge the gap between ML and the factory floor.

**Slide 50: References & Further Reading**
- Sandfeld Ch 12, 16.
- Bergstra & Bengio (2012) on Random Search.
- SHAP documentation (Lundberg & Lee, 2017).
