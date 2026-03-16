# Unit 6: Data Scarcity & Transfer Learning - Slide Content (50 Slides)

## Part 1: The Small Data Challenge (8 slides)

**Slide 1: Title Slide**
- **Unit 6: Data Scarcity & Transfer Learning**
- Machine Learning in Materials Processing & Characterization (ML-PC)
- Topic: Learning from Limited Labeled Data

**Slide 2: The "Big Data" Myth**
- While materials generate TBs of raw data (e.g., 4D-STEM), **labeled data** is sparse.
- In common ML (ImageNet), labels are cheap (Amazon Mechanical Turk).
- In materials science, labels are expensive (PhD experts, hours of annotation).

**Slide 3: Why is Materials Data Scarce?**
- High acquisition cost per sample.
- Limited access to unique facilities (Synchrotrons, specialized TEMs).
- Expert annotation time: Segmenting 100 grains in an SEM can take hours.

**Slide 4: Labeled Data vs. Raw Data Gap**
- We have millions of "background" images, but only hundreds of "labeled" examples.
- Standard Deep Learning (ResNet-50) is designed for 1M+ images.
- If we train from scratch on 100 images, we get **Overfitting**.

**Slide 5: Overfitting on Small Data**
- Model "memorizes" the specific noise/artifacts of those 100 images.
- Fails catastrophically on a new dataset from a different microscope.
- High variance in model performance.

**Slide 6: The "Small Data" Survival Kit**
1. **Data Augmentation**: Multiply data by "distorting" it.
2. **Transfer Learning**: Reuse knowledge from "Big Data" models.
3. **Synthetic Training**: Generate labels for free using physics/algorithms.

**Slide 7: Strategy Map**
- [Diagram showing how these techniques interact: Pretraining -> Augmentation -> Fine-tuning]

**Slide 8: Summary: Part 1**
- Materials science is the land of "Small Data."
- We must be clever with data reuse and knowledge transfer.

---

## Part 2: Data Augmentation (12 slides)

**Slide 9: Concept: Artificially Expanding the Dataset**
- "Reusing existing images by applying transformations." (Sandfeld 19.2.5).
- A form of **Oversampling**.

**Slide 10: Geometric Transformations**
- Flips (Horizontal, Vertical).
- Rotations (90°, 180°, 270° or arbitrary degrees).
- Scaling/Cropping.

**Slide 11: Invariance via Augmentation**
- By rotating images, we force the network to be rotation-invariant.
- Crucial for microstructures where "up" and "down" are often arbitrary.

**Slide 12: When Augmentation is "Illegal"**
- **Physical Reality Check**: Transformations shouldn't violate materials physics.
- Example: If a texture has a specific preferred orientation, arbitrary rotations might confuse the model.
- Example: Warping shouldn't break grain boundary topology.

**Slide 13: Intensity Transformations**
- Brightness/Contrast jittering.
- Mimicking different detector settings or lighting conditions.

**Slide 14: Adding "Physical" Noise**
- Gaussian noise (electronic noise).
- Poisson/Shot noise (counting statistics).
- Blur (approximating different focus/resolution).

**Slide 15: Why add noise?**
- To make the model robust to "dirty" experimental data.
- Forces the network to focus on structural motifs, not pixel-perfect patterns.

**Slide 16: More Complex Augmentations**
- Elastic transformations (simulating sample strain or drift).
- CutOut / Random Erasing (handling occlusions or artifacts).

**Slide 17: Albumentations / Torchvision**
- [Code snippet example using Albumentations for a pipeline]
- `Compose([HorizontalFlip(), RandomRotate90(), GaussianNoise()])`

**Slide 18: On-the-fly vs. Offline Augmentation**
- **Offline**: Generating new images on disk before training.
- **On-the-fly**: Transforming images in RAM during each training batch. (Preferred for diversity).

**Slide 19: The "Label Consistency" Rule**
- If you rotate the image, you must rotate the mask (labels) identically!
- [Visual example of a rotated grain image and its corresponding mask]

**Slide 20: Summary: Part 2**
- Augmentation is the easiest way to combat small datasets.
- Use it to enforce physical invariances (Rotation, Scale).

---

## Part 3: Transfer Learning (12 slides)

**Slide 21: Concept: Knowledge Reuse**
- "Learning on Peas to count Lentils." (Sandfeld 19.2.4).
- Reusing a model trained for Task A on Task B.

**Slide 22: The ImageNet Pretraining**
- ImageNet: 14M images, 1000 classes (dogs, cats, cars).
- Why do dog features help find grain boundaries?
- **Hierarchical Features**: Early layers learn edges, blobs, and textures common to ALL images.

**Slide 23: The "Backbone" and the "Head"**
- **Backbone**: The feature extractor (ResNet, VGG).
- **Head**: The classifier/regressor for the specific task.

**Slide 24: Strategy 1: Feature Extraction**
- **Freeze** the backbone.
- Only train the new head on your small materials dataset.
- Fast, low risk of overfitting.

**Slide 25: Strategy 2: Fine-Tuning**
- Initialize with pretrained weights.
- Train the **entire** network (or just the last few layers).
- Requires a bit more data but allows the backbone to adapt to "microstructure-specific" textures.

**Slide 26: Layer-wise Learning Rates**
- High LR for the head (learning new classes).
- Very low LR for the backbone (fine-tuning existing filters).

**Slide 27: Domain Gap: Natural vs. Scientific**
- Natural images (RGB) vs. Microscopy (Grayscale, 16-bit).
- Natural (Perspective) vs. Microscopy (Orthographic, Top-down).
- Some features might not transfer (e.g., shadows/perspective).

**Slide 28: Gradual Unfreezing**
- 1. Train head until convergence.
- 2. Unfreeze top backbone layer, train more.
- 3. Repeat. (Prevents "catastrophic forgetting" of low-level features).

**Slide 29: Success Case: Au Nanoparticles (Sandfeld 19.3.1)**
- U-Net initialized with ImageNet weights.
- Segmenting crystalline regions from amorphous backgrounds.
- High accuracy even with limited labeled TEM frames.

**Slide 30: Transfer from Physics?**
- Pretraining on simulated data (DFT, MD) before fine-tuning on real experiments.
- The next frontier of transfer learning in materials science.

**Slide 31: Transfer across Microscopes**
- Training on a high-end aberration-corrected TEM.
- Transferring to a lower-end "office" TEM.

**Slide 32: Summary: Part 3**
- Don't train from scratch! Always start with a pretrained backbone.
- Fine-tune carefully to bridge the domain gap.

---

## Part 4: Learning from Synthetic Data (10 slides)

**Slide 33: The "Infinite Data" Dream**
- If we can simulate the microstructure, we can generate infinite labeled data.
- **Perfect Masks for Free**: No expert needed.

**Slide 34: Generating Grain Microstructures**
- Voronoi Tessellations (Sandfeld 19.3.2).
- Parameters: Number of seeds, regularity, boundary thickness.

**Slide 35: From Geometry to Image**
- [Image showing a Voronoi diagram becoming a noisy SEM-like image]
- Adding realistic texture, contrast, and noise to synthetic geometries.

**Slide 36: Sim-to-Real Gap**
- Synthetic data is often "too clean."
- CNNs might learn synthetic-only features and fail on real SEMs.

**Slide 37: Domain Adaptation**
- Making synthetic images look more like real ones.
- Using **GANs** (Generative Adversarial Networks) or simple style transfer.

**Slide 38: Case Study: SEM Grain Segmentation**
- Sandfeld Fig 19.11.
- Model trained **only** on Voronoi synthetic data.
- Tested on real polycrystalline SEM images.

**Slide 39: Success of Synthetic Data**
- Nearly perfect segmentation of complex grain networks.
- Synthetic data captured the "topological truth" of the grains.

**Slide 40: Adaptive Data Generation**
- Train on 1000 synthetic images.
- Find the "hardest" real images (worst predictions).
- Generate synthetic data that mimics those hard cases. (Sandfeld 19.3.2).

**Slide 41: Procedural Generation for Spectra**
- Simulating XRD/EELS peaks with varying noise and background.
- Training models to "de-noise" or "peak-find" automatically.

**Slide 42: Summary: Part 4**
- Synthetic data is a powerful label-free alternative.
- Bridging the sim-to-real gap is the key technical challenge.

---

## Part 5: Practical Workflow & Best Practices (8 slides)

**Slide 43: The Fine-Tuning Recipe**
1. Select a pretrained architecture (e.g., ResNet-50).
2. Replace the final layer (the head) for your number of classes.
3. Freeze all layers except the head.
4. Train head with a standard LR ($10^{-3}$).
5. Unfreeze and train everything with a very small LR ($10^{-5}$).

**Slide 44: Validation in the Small Data Regime**
- **K-Fold CV** is mandatory (Unit 3).
- Be extremely wary of **Data Leakage** from augmentation (don't have an image and its rotation in both train and test!).

**Slide 45: Group-based Splitting Revisited**
- [Diagram showing a specimen being split correctly across folds]
- If you have 5 specimens, split by specimen, not by individual augmented crops.

**Slide 46: Active Learning (Bonus concept)**
- Labeling only the most "informative" images (where the model is least confident).
- Maximizes the value of every expert hour.

**Slide 47: The "Gold Standard" Dataset**
- Even with TL/Augmentation, you need a small, high-quality "Gold Standard" test set.
- This is the absolute benchmark for your model.

**Slide 48: When to Stop?**
- Use **Early Stopping** based on the validation loss.
- Small datasets are prone to sudden overfitting late in training.

**Slide 49: Summary: Top Takeaways**
1. **Augment** your data to enforce physical invariances.
2. **Transfer** knowledge from large datasets (ImageNet) or synthetic models.
3. **Synthetic** data can provide infinite labels if generated carefully.
4. **Validate** rigorously to avoid the "Small Data Trap."

**Slide 50: References & Further Reading**
- Sandfeld Ch 19, McClarren Ch 6, Sytwu et al. (2019) for U-Net.
