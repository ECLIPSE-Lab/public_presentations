# Unit 6: Data Scarcity & Transfer Learning - Lecture Plan

## Objective
The objective of this unit is to equip students with strategies to overcome the lack of large labeled datasets in materials science. We will focus on Data Augmentation to artificially expand datasets and Transfer Learning to leverage knowledge from large-scale natural and synthetic image datasets.

## Slide Structure (approx. 50 slides)

### Part 1: The Small Data Challenge (8 slides)
- **The Bottleneck (3 slides):**
  - Why is materials data "small"? (Cost of TEM/SEM, expert labeling time).
  - The "Labeled Data vs. Raw Data" gap.
- **Consequences of Small Data (2 slides):**
  - Overfitting and poor generalization.
  - High variance in model performance.
- **The Strategy Map (3 slides):**
  - Augmentation, Transfer Learning, Synthetic Data, and Active Learning.

### Part 2: Data Augmentation (12 slides)
- **Concept (2 slides):**
  - Reusing existing images by applying transformations (Sandfeld 19.2.5).
- **Geometric Transformations (4 slides):**
  - Rotations, Flips, Cropping, Warping.
  - When is a transformation "physically illegal"? (e.g., non-parallel lines in crystals).
- **Intensity & Noise Transformations (3 slides):**
  - Color/Brightness jittering.
  - Adding Gaussian or Poisson noise to mimic detector artifacts.
- **Implementation (3 slides):**
  - On-the-fly augmentation vs. offline.
  - Library tools (Torchvision, Albumentations).

### Part 3: Transfer Learning (12 slides)
- **Concept: Knowledge Reuse (3 slides):**
  - Learning on Peas to count Lentils (Sandfeld 19.2.4).
  - Pretraining on ImageNet (14 million natural images).
- **Feature Extraction vs. Fine-Tuning (4 slides):**
  - Freezing the "Backbone" (early layers find edges/blobs).
  - Training the "Head" (final layers find phases/defects).
- **Fine-Tuning Strategies (3 slides):**
  - Differential learning rates (low for backbone, high for head).
  - Gradual unfreezing.
- **Success Story (2 slides):**
  - U-Net initialized with ImageNet weights for nanoparticle segmentation (Sandfeld 19.3.1).

### Part 4: Learning from Synthetic Data (10 slides)
- **Digital Twins for Training (4 slides):**
  - Using Voronoi tessellations to generate infinite grain microstructures (Sandfeld 19.3.2).
  - Perfect masks for free: The benefit of algorithmic generation.
- **Sim-to-Real Gap (3 slides):**
  - Adding "realism" to synthetic data (noise, blur, texture).
  - Domain Adaptation: Making synthetic look real.
- **Case Study: SEM Grain Segmentation (3 slides):**
  - Training on Voronoi, testing on real SEM (Sandfeld Fig 19.11).

### Part 5: Practical Workflow & Best Practices (8 slides)
- **The Fine-Tuning Recipe (3 slides):**
  - 1. Load pretrained model. 2. Freeze base. 3. Train head. 4. Unfreeze & Fine-tune.
- **Validation in the Small Data Regime (3 slides):**
  - Importance of K-Fold and Group-based splitting (Ref. Unit 3).
- **Summary & Takeaways (2 slides):**
  - Augment carefully, Transfer aggressively.

## Materials Sources
- **Sandfeld (2024):** Ch 19.2.4 (Transfer Learning), 19.2.5 (Augmentation), 19.3.1 (Nanoparticles), 19.3.2 (Synthetic Data).
- **McClarren (2021):** Ch 6 (CNNs and large datasets).
- **Neuer (2024):** Ch 4 (Supervised learning strategies).
