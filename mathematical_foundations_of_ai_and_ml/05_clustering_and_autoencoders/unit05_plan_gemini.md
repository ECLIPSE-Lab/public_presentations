# Lecture Plan: Clustering & Autoencoders (Math 05)

## Overview
This lecture focuses exclusively on unsupervised learning, transitioning from traditional clustering methods to deep-learning-based non-linear dimensionality reduction.

## Part 1: Clustering Foundations (45 mins)
*   **Unsupervised Learning Intro:** Finding hidden structure when labels are missing.
*   **K-Means:** Optimization problem (minimizing variance/distances), iterative assignment-update cycle.
*   **K-Medoids:** Robustness to outliers by using actual data points as prototypes.
*   **Gaussian Mixture Models (GMMs):** Probabilistic clustering, "soft" cluster assignments, and a conceptual introduction to Expectation-Maximization (EM).

## Part 2: Autoencoders and Dimensionality Reduction (45 mins)
*   **Non-linear Dimensionality Reduction:** Briefly recap why linear methods like PCA fail on complex manifolds.
*   **The Encoder-Decoder Architecture:** Explaining the "information bottleneck" and reconstruction loss.
*   **Latent Spaces:** How Autoencoders serve as "non-linear PCA" by utilizing backpropagation to learn highly efficient continuous representations.
