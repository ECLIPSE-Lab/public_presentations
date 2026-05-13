# Unit 09 Plan: Inverse Problems and Process Maps

## Metadata
- **Course:** Machine Learning for Characterization and Processing (ML-PC)
- **Unit:** 09
- **Topic:** Inverse Problems and Process Maps
- **Duration:** 90 Minutes
- **Key References:** 
    - Sandfeld (2024), Ch 19.6 (Inverse PINNs)
    - Neuer (2024), Ch 6.5 (Process Corridors)
    - McClarren (2021), Ch 2.5 (Symbolic Discovery)
    - MFML Unit 9 (Inverse Problems)

## Learning Objectives
- Define the difference between forward and inverse problems in materials processing.
- Explain the concept of "Ill-posedness" and how ML regularizes inverse solutions.
- Understand the role of Physics-Informed Neural Networks (PINNs) in parameter discovery.
- Use sparse regression (LASSO) to discover governing equations from noisy data (SINDy).
- Create and interpret multi-dimensional "Process Maps" and "Process Corridors."

## Lecture Structure (90 Minutes)

### 1. Introduction: Forward vs. Inverse (15m)
- Forward: Processing $\to$ Property. (Easy, well-defined).
- Inverse: Property $\to$ Processing. (Hard, non-unique).
- Why inverse problems matter: Designing the processing route for a desired product.
- Regularization: How ML handles non-uniqueness.

### 2. Process Maps and Corridors (15m)
- (Ref: Neuer Ch 6.5)
- Visualizing the "Safe Operating Window."
- Process Corridors: Dealing with uncertainty and drift over time.
- Beyond 2D maps: High-dimensional feasibility regions.

### 3. Solving Inverse Problems with PINNs (25m)
- (Ref: Sandfeld Ch 19.6)
- Using the PDE residual as a constraint.
- Parameter Discovery: Learning material constants (e.g., diffusivity) directly from measurements.
- Case Study: Determining laser parameters in AM from melt pool shapes.

### 4. Governing Equation Discovery (20m)
- (Ref: McClarren Ch 2.5)
- Sparse Identification of Non-linear Dynamics (SINDy).
- Using LASSO to find the simplest law that explains the data.
- Transitioning from "Black-Box" to "Formula."

### 5. Implementation: Surrogate-based Inversion (10m)
- Training a fast forward model (surrogate).
- Using Gradient Descent on the *input* space of the surrogate to find the optimal process.

### 6. Summary and Conclusion (5m)

## 50-Slide Strategy
- Slides 1-5: Intro & Forward/Inverse Duality
- Slides 6-12: Ill-posedness & Regularization
- Slides 13-20: Process Maps & Corridors (Theory & Examples)
- Slides 21-35: Parameter Discovery with PINNs
- Slides 36-45: Equation Discovery (SINDy & McClarren Case Study)
- Slides 46-50: Synthesis & Design of Experiments
