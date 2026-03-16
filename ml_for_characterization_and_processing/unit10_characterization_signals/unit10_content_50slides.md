# Unit 10: ML for Characterization Signals (50 Slide Draft)

## Section 1: Intro & Spectral Data Examples (Slides 1-5)

1. **Slide 1: Title Slide**
   - Title: Unit 10: Machine Learning for Characterization Signals
   - Course: ML for Characterization and Processing (SS26)
   - Prof. Dr. Philipp Pelz
   - Focus: 1D Signals, PCA, and Autoencoders

2. **Slide 2: Beyond Images: 1D Characterization Signals**
   - We've focused on spatial data (Microstructures).
   - But many instruments produce 1D "Spectra."
   - Examples: XRD, EELS, EDS, XPS, IR.

3. **Slide 3: The Nature of Characterization Signals**
   - High Dimensionality: A spectrum may have 1024 or 4096 energy channels.
   - Sparsity: Often only a few "Peaks" are present.
   - Backgrounds: Continuous signals (bremsstrahlung, plasmons).
   - Noise: Shot noise (Poisson) is dominant.

4. **Slide 4: The Need for Machine Learning**
   - Manual peak fitting is slow and subjective.
   - Multi-phase materials lead to overlapping peaks.
   - Large mapping datasets (e.g., $10^6$ EDS spectra) are impossible to analyze by hand.

5. **Slide 5: Our Goals for this Unit**
   - Compression: How to store large maps efficiently.
   - Denoising: Improving signal quality without increasing acquisition time.
   - Phase ID: Grouping similar spectra automatically.

## Section 2: PCA for Spectra (Slides 6-15)

6. **Slide 6: Principal Component Analysis (PCA) for Spectra**
   - (Ref: Sandfeld Ch 15.2).
   - The classical tool for spectral decomposition.
   - Mapping high-dimensional spectra $\mathbf{x}$ to a low-dimensional coefficient vector $\mathbf{c}$.

7. **Slide 7: PCA: The Basis Function View**
   - $\mathbf{x} \approx \sum_{k=1}^K c_k \mathbf{v}_k$.
   - $\mathbf{v}_k$ are the "Eigen-spectra" (Principal Components).
   - $c_k$ are the "Scores" (how much of each eigen-spectrum is present).

8. **Slide 8: Interpreting Eigen-spectra**
   - PC1: The mean signal (the overall shape).
   - PC2, PC3: Deviations from the mean (specific peaks or background shifts).
   - Higher PCs: Usually represent noise.

9. **Slide 9: Why it works: Intrinsic Dimensionality**
   - Even with 1024 channels, the material only has a few phases.
   - The data lies on a low-dimensional linear subspace.

10. **Slide 10: Scree Plots: Choosing the Number of Components**
    - (Ref: Sandfeld 15.3.2).
    - Plotting Explained Variance vs. PC Index.
    - The "Elbow" indicates where signal stops and noise begins.

11. **Slide 11: Denoising via Reconstruction**
    - (Ref: Sandfeld 15.3.3).
    - 1. Project noisy spectrum $\mathbf{x}_{noisy}$ onto $K$ components.
    - 2. Reconstruct $\hat{\mathbf{x}} = \mathbf{V}_K \mathbf{c}_K$.
    - 3. Result: High-frequency noise is filtered out.

12. **Slide 12: Example: Fast EDS Mapping**
    - (Graphic: Original noisy spectrum vs. PCA-denoised spectrum).
    - Allows for 10x faster scanning while maintaining chemical sensitivity.

13. **Slide 13: PCA Limitations**
    - Linear: Only works for linear combinations (e.g., Beer-Lambert law).
    - Fails for shifts in peak positions (non-linear variations).

14. **Slide 14: PCA vs. Physical Models**
    - PCs are orthogonal; physical phases are not.
    - Transition to NMF (Non-negative Matrix Factorization).

15. **Slide 15: Summary: PCA**
    - Fast, robust, and the industry standard for denoising.

## Section 2: Autoencoders for Signals (Slides 16-30)

16. **Slide 16: Autoencoders (AE): The Non-linear Generalization**
    - (Ref: Sandfeld 19.4).
    - A Neural Network that learns to reconstruct its input: $f(x) \approx x$.

17. **Slide 17: AE Architecture: The "Hourglass"**
    - Encoder: $h = f(x)$ (Compression).
    - Bottleneck: The low-dimensional latent space.
    - Decoder: $\hat{x} = g(h)$ (Reconstruction).

18. **Slide 18: Why the Bottleneck?**
    - Forced compression: The network *must* find the most important features.
    - Capturing the "essence" of the signal.

19. **Slide 19: Non-linearity: The Power of AE**
    - (Ref: Sandfeld 19.4.2).
    - While PCA uses straight lines, AEs learn curved manifolds.
    - Can handle peak shifts, widening, and complex background shapes.

20. **Slide 20: Training Objective: Reconstruction Loss**
    - $\mathcal{L} = \|x - \hat{x}\|^2$.
    - Simple but effective.

21. **Slide 21: Latent Space Properties**
    - Dimensions in latent space often correlate with physical properties (e.g., thickness, concentration).

22. **Slide 22: Denoising Autoencoders (DAE)**
    - (Ref: McClarren 8.3.2).
    - Training: Input = $x + noise$, Target = $x$.
    - The network learns to project "noisy" points back onto the "clean" manifold.

23. **Slide 23: DAE Example: EELS Spectra**
    - Removing artifacts and multiple scattering from energy loss signals.

24. **Slide 24: Convolutional Autoencoders (CAE)**
    - Using 1D convolutions to capture local peak shapes.
    - Better than fully-connected AEs for shift-invariance.

25. **Slide 25: Variational Autoencoders (VAE)**
    - (Advanced topic).
    - Learning a continuous probability distribution in latent space.
    - Better for generative tasks and uncertainty.

26. **Slide 26: Comparison: PCA vs. AE**
    - PCA: Fast, unique solution, linear.
    - AE: Flexible, non-linear, but requires more data and training.

27. **Slide 27: AE for Data Compression**
    - Storing only the latent codes for a large spectral map.
    - 100x compression ratios possible.

28. **Slide 28: Anomaly Detection with AE**
    - If a spectrum is "exotic" (not in training set), the reconstruction error will be high.
    - Automatically flags "Unexpected" material states.

29. **Slide 29: Feature Extraction for Downstream Tasks**
    - Using latent codes as inputs for a Classifier (Unit 14).

30. **Slide 30: Summary: Autoencoders**
    - Deep learning's answer to PCA.
    - Powerful for non-linear characterization data.

## Section 4: Denoising & Feature Extraction (Slides 31-40)

31. **Slide 31: Practical Signal Processing Workflow**
    - 1. Preprocessing (Background subtraction, normalization).
    - 2. Dimensionality Reduction (PCA or AE).
    - 3. Clustering/Classification.

32. **Slide 32: Normalization Strategies**
    - Peak-height normalization vs. Area normalization.
    - Why it matters for ML performance.

33. **Slide 33: Multi-Spectral Imaging (Hyperspectral Data)**
    - Every pixel in an image is a 1D spectrum.
    - Data Cube: $(x, y, E)$.

34. **Slide 34: Spatial-Spectral Autoencoders**
    - Using 3D convolutions to capture both local chemistry and local morphology.

35. **Slide 35: Learning the Peak Dictionary**
    - Using ML to identify "End-members" (Pure phases).

36. **Slide 36: Denoising with Prior Knowledge**
    - Combining Unit 13 (Physics) with Autoencoders.
    - Forcing the AE to produce physically valid peaks (e.g., non-negative, Lorentzian shape).

37. **Slide 37: Signal Separation**
    - Unmixing overlapping signals using latent space analysis.

38. **Slide 38: Transfer Learning for Signals**
    - Using a model trained on synthetic XRD spectra to analyze real experimental data.

39. **Slide 39: Robustness to Instrumental Shifts**
    - Calibrating the energy axis using ML alignment.

40. **Slide 40: Summary: Signal Processing**
    - Latent spaces provide the "Universal Language" of characterization.

## Section 5: Case Studies (Slides 41-48)

41. **Slide 41: Case Study: Automatic XRD Phase ID**
    - Input: Noisy XRD pattern.
    - ML: Classification of phases using latent space embeddings.

42. **Slide 42: Case Study: EELS Spectrum Imaging**
    - Identifying oxidation states of Iron at atomic scale.
    - Using AEs to differentiate between $Fe^{2+}$ and $Fe^{3+}$.

43. **Slide 43: Case Study: Large-Scale EDS Maps**
    - Processing 1 million spectra to find trace elements.
    - PCA-based denoising enables detection below the original noise floor.

44. **Slide 44: Clustering in Latent Space (Sandfeld 15.6)**
    - (Graphic: UMAP plot of spectra).
    - Natural clusters correspond to different material phases.

45. **Slide 45: Discovering New Phases**
    - Using Anomaly detection to find "Unknown" peaks in discovery data.

46. **Slide 46: Real-time Signal Monitoring**
    - Using the encoder to monitor a process (e.g., plasma state) at 100 Hz.

47. **Slide 47: Integration: Unit 1-14 Convergence**
    - Signal analysis is the bridge between processing (Unit 9) and characterization (Unit 11).

48. **Slide 48: Conclusion: The Value of Unsupervised Learning**
    - We don't need a label to find a pattern.

## Section 6: Summary & Outlook (Slides 49-50)

49. **Slide 49: Take-Home Messages**
    - Spectra are high-dimensional but inherently sparse.
    - PCA is the first line of defense for denoising.
    - Autoencoders unlock non-linear relationships and compression.
    - Latent spaces are the key to automated interpretation.

50. **Slide 50: References & Further Reading**
    - Sandfeld (2024), Neuer (2024), McClarren (2021).
    - Next Unit: Inverse Problems and Process Maps.
