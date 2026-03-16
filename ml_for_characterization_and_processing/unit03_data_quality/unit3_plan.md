# Unit 3: Data Quality, Preprocessing, and Robust Validation

## Objective
The objective of this unit is to provide students with a comprehensive understanding of data preprocessing, cleaning, transformation, and validation strategies. The focus is on ensuring data quality and model robustness in the context of materials characterization and processing.

## Key Topics

### 1. Data Cleaning
- **Sources of Error**: Sensors, transmission, human error. "Garbage in, garbage out" (GIGO).
- **Missing Values (NaNs)**: Handling at source vs. digital repair. Interpolation (linear, mean-based) and numerical markers (e.g., -1000°C for temperature).
- **Outlier Detection**: Point/Global, Contextual (noise), and Collective outliers. Point vs. rare events.
- **Duplicate Tracking**: Identifying and removing duplicates to prevent bias.

### 2. Data Transformation and Scaling
- **Shifting and Centering**: Aligning peaks, mean-centering at zero.
- **Scaling and Normalization**: Linear scaling, min-max scaling [0,1], non-dimensionalization (removing units).
- **Standardization**: Z-score (mean 0, std 1). Importance for algorithms sensitive to magnitude (e.g., KNN, PCA).
- **Complex Transformations**:
    - **Differentiation**: Highlighting changes, removing constant offsets.
    - **Frequency Domain (FFT)**: Extracting periodic signals/oscillations.
    - **Time-Frequency (Wavelet)**: Localizing temporal features.
    - **Log-transform**: Linearizing exponential trends.

### 3. Labeling and Uncertainty
- **Labeling Challenges**: Inter-annotator variance (no "unique" ground truth). Hand-labeling for images (e.g., masks for TEM/SEM).
- **Uncertainty**: Probabilistic labels vs. hard labels. Using Softmax to quantify confidence.
- **Bayesian Perspective**: Prior vs. Posterior distributions for parameters/predictions.

### 4. Robust Validation and Model Selection
- **Overfitting vs. Underfitting**: The Bias-Variance Tradeoff. Occam's Razor (Parsimony) and Lasso regression.
- **Validation Strategies**:
    - **Train-Test Split (Holdout)**: Simple split, risks of unrepresentative sets.
    - **K-Fold Cross-Validation**: Iterative training/testing for stable performance estimates.
    - **Leave-One-Out (LOOCV)**: Special case for small datasets.
    - **Stratified Splitting**: Preserving class balance across folds.
- **Data Leakage**:
    - **Temporal Leakage**: Respecting the "arrow of time" in time series.
    - **Correlation/Spatial Leakage**: Group-based splitting (ensuring specimens/experiments are not split between train and test).

### 5. Error Measures
- **Regression**: MAE (L1), MSE (L2), RMSE, MSLE, $R^2$ (Coefficient of Determination).
- **Classification**: Confusion Matrix (TP, TN, FP, FN), Precision, Recall (Sensitivity), F1-Score (Dice), Jaccard Similarity (IoU).
- **Categorical Cross-Entropy**: Loss function for multi-class tasks.

## Materials Sources
- Sandfeld (2024) Ch 11.5, 16.2
- Neuer (2024) Ch 3
- McClarren (2021) Ch 1
