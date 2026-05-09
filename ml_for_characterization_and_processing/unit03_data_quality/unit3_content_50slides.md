# Unit 3: Data Quality, Preprocessing, and Robust Validation (Content)

## 1. Introduction to Data Quality
1. **Slide 1**: Title: Unit 3 - Data Quality, Preprocessing, and Robust Validation. Subtitle: Ensuring the Integrity of ML in Materials Science.
2. **Slide 2**: Objective: Why care about data quality? "Garbage In, Garbage Out" (GIGO). Model performance is fundamentally limited by input quality.
3. **Slide 3**: The ML Workflow: Collection → Preprocessing → Modeling → Learning → Error Analysis → Results. Unit 3 focuses on Preprocessing and Error Analysis.
4. **Slide 4**: The Role of Preprocessing: Transforming raw, potentially messy data into a structured format suitable for algorithms.

## 2. Data Cleaning
5. **Slide 5**: Common Data Issues: Structural problems (typos, units), Duplicates, Irrelevant observations, Missing values (NaNs), Outliers.
6. **Slide 6**: Missing Values: Sources: Sensor failure, transmission drops, or "out-of-range" readings.
7. **Slide 7**: Handling NaNs: Source correction is best (e.g., hardware fix). If not possible, digital repair: Interpolation (linear, mean-based), or removal (risk of bias).
8. **Slide 8**: Numerical Markers: Using "impossible" values (e.g., -1000°C) to track NaNs without losing record count.
9. **Slide 9**: Outlier Detection: Point Outlier (global), Contextual (noise relative to neighbors), and Collective (group behaves differently).
10. **Slide 10**: To Remove or Not? Is it a sensor artifact? Or a "rare event" physically significant (e.g., crack initiation)?
11. **Slide 11**: Duplicate Tracking: Redundant data points can bias training (over-weighting certain conditions) and inflate validation scores (leakage).

## 3. Data Transformation and Scaling
12. **Slide 12**: Why Transform? To linearize trends, align features, and ensure equal weighting across variables.
13. **Slide 13**: Shifting and Centering: Subtracting the mean to center data at zero. Useful for covariance-based methods (e.g., PCA).
14. **Slide 14**: Linear Scaling and Normalization: Rescaling features to [0,1] or [-1,1]. Essential for algorithms using Euclidean distance (e.g., kNN).
15. **Slide 15**: Standardization (Z-score): Scaling by mean and standard deviation (mean 0, std 1). Robust to scale differences across features.
16. **Slide 16**: Log-Transforms: Useful for variables spanning several orders of magnitude (e.g., grain sizes, dislocation densities). Linearizes exponential trends.
17. **Slide 17**: Differentiation: $f'(x)$ highlights changes and removes constant offsets (e.g., baseline drift). Useful for anomaly detection.
18. **Slide 18**: Frequency Domain (FFT): Fast Fourier Transform. Switching from time space to frequency space to detect oscillations.
19. **Slide 19**: Time-Frequency Analysis (Wavelet): Localizing features that are both temporal and periodic. Useful for transient phenomena (e.g., acoustic emissions).
20. **Slide 20**: Triggering: Isolating relevant sequences from long time series (e.g., periodic loading cycles in fatigue tests).

## 4. Labeling and Uncertainty
21. **Slide 21**: Labeling Challenges: Often manual ("hand-labeling"). Subjective and time-consuming.
22. **Slide 22**: Inter-Annotator Variance: Different experts might provide different labels/masks for the same TEM image. No single "unique" truth.
23. **Slide 23**: Quantifying Uncertainty: Models shouldn't just output a label; they should output confidence (Probabilities via Softmax).
24. **Slide 24**: Bayesian Perspective: Treating model parameters as distributions. Prior (what we think before data) + Likelihood (data) = Posterior (updated knowledge).

## 5. Robust Validation and Model Selection
25. **Slide 25**: Overfitting vs. Underfitting: The Bias-Variance Tradeoff.
26. **Slide 26**: Underfitting: Model is too stiff (High Bias). Fails to capture the trend.
27. **Slide 27**: Overfitting: Model is too flexible (High Variance). Captures noise as if it were a trend. Fails to generalize.
28. **Slide 28**: Parsimony (Occam's Razor): Prefer the simplest model that explains the data. "Entities must not be multiplied beyond necessity."
29. **Slide 29**: Regularization: Lasso (L1) can force redundant parameters to zero, yielding simpler models.
30. **Slide 30**: Train-Test Split (Holdout): Dividing data into independent sets. Training sets model parameters; Test set evaluates performance.
31. **Slide 31**: The Risk of Holdout: Small datasets might lead to unrepresentative splits (e.g., all "hard" cases in the test set).
32. **Slide 32**: K-Fold Cross-Validation: Split data into $k$ folds. Iteratively use $k-1$ for training and 1 for testing. Average the results.
33. **Slide 33**: Leave-One-Out (LOOCV): $k$ equals total number of samples. Most robust but computationally expensive for large datasets.
34. **Slide 34**: Stratified Splitting: Ensuring each fold has the same class distribution as the original dataset (crucial for imbalanced data).
35. **Slide 35**: Data Leakage: When information from the test set "leaks" into training, giving overly optimistic results.
36. **Slide 36**: Temporal Leakage: In time series, you cannot use future data to predict the past. Validation must follow the arrow of time.
37. **Slide 37**: Spatial/Group Leakage: If multiple data points come from the same specimen, random splitting can leak information. Need "Group-based" splitting.

## 6. Error Measures
38. **Slide 38**: Measuring Success: How do we quantify "accuracy"? Different measures for different tasks.
39. **Slide 39**: Regression - MAE (L1): Mean Absolute Error. Direct units, less sensitive to outliers.
40. **Slide 40**: Regression - MSE (L2) & RMSE: Mean Squared Error. Penalizes large errors disproportionately. RMSE returns to original units.
41. **Slide 41**: Coefficient of Determination ($R^2$): Fraction of variance explained by the model. 1 = perfect fit.
42. **Slide 42**: Classification - Confusion Matrix: TP (True Pos), TN (True Neg), FP (False Pos/Type I), FN (False Neg/Type II).
43. **Slide 43**: Precision: "Of all predicted positive, how many were actually positive?" (Avoids FP).
44. **Slide 44**: Recall (Sensitivity): "Of all actually positive, how many did we find?" (Avoids FN).
45. **Slide 45**: F1-Score (Dice Coefficient): Harmonic mean of Precision and Recall. Balances both metrics.
46. **Slide 46**: Jaccard Similarity (IoU): Intersection over Union. Essential for image segmentation/object detection tasks.
47. **Slide 47**: Categorical Cross-Entropy: Loss function for training multi-class classification models. Penalizes incorrect high-confidence predictions.

## 7. Summary and Key Takeaways
48. **Slide 48**: Summary: Data quality starts with cleaning. Transformation and Scaling ensure model stability.
49. **Slide 49**: Summary: Robust validation (K-Fold, Group-based splitting) is the only guard against overfitting and leakage.
50. **Slide 50**: Takeaway: Always visualize your residuals and understand your error metrics. A high $R^2$ doesn't always mean a useful model.
