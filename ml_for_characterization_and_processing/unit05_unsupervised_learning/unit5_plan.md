# Unit 5: Convolutional Neural Networks for Microstructure Analysis

## Learning Objectives
1. Explain the "Parameter Explosion" problem in Multi-layer Perceptrons (MLPs) for high-resolution images.
2. Define the convolution operation, kernels, stride, and padding.
3. Understand local connectivity and weight sharing for translation invariance.
4. Distinguish between Max and Average Pooling and their roles in downsampling.
5. Summarize the intuition behind key CNN architectures (LeNet, AlexNet, ResNet).
6. Apply CNN concepts to materials science case studies: phase segmentation and defect detection.

## Structure (50 Slides)
1. **The Image Problem (Slides 1-8)**
    - Limitations of MLPs (McClarren 6.1, Sandfeld 19.1.1).
    - Parameter explosion with 1024x1024 images.
    - Loss of spatial structure in flattened vectors.
2. **The Convolution Layer (Slides 9-20)**
    - Definition of discrete convolution (Sandfeld 19.1.2, McClarren 6.1).
    - Kernel/Filter intuition (edge detection example - McClarren Fig 6.5).
    - Stride and Padding (Same vs. Valid).
3. **Architectural Principles (Slides 21-30)**
    - Local connectivity and weight sharing (Sandfeld 19.1.2).
    - Translation invariance vs. Equivariance.
    - Receptive field concept (Sandfeld 19.1.4).
4. **The Pooling Layer (Slides 31-38)**
    - Max vs. Average pooling (McClarren 6.2, Sandfeld 19.1.3).
    - Hierarchical feature maps (Edges -> Motifs -> Phases).
5. **Key Architectures (Slides 39-44)**
    - LeNet (Handwritten digits - Sandfeld 19.1.6, McClarren 6.4).
    - AlexNet (The DL revolution).
    - ResNet (Skip connections - Sandfeld 19.1.6).
6. **Materials Science Case Studies (Slides 45-50)**
    - Phase segmentation of Au nanoparticles (TEM - Sandfeld 19.3.1).
    - Synthetic data training for SEM grain microstructures (Sandfeld 19.3.2).
    - Property prediction from Ising microstructures (Sandfeld 18.8).

## Key Terms
- Convolution, Kernel/Filter, Stride, Padding, Pooling, Translation Invariance, Skip Connections, U-Net.
