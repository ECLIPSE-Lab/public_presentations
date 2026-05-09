# Unit 5 Content: CNNs for Microstructure Analysis (50 Slides)

## Section 1: The Image Problem (Slides 1-8)
1. **Title Slide**: Convolutional Neural Networks for Microstructure Analysis.
2. **Microscopy in Materials Science**: Diverse data (TEM, SEM, AFM) often represented as high-resolution images.
3. **The Standard MLP Approach**: Review of Unit 4 - Flattening the image into a 1D vector.
4. **Parameter Explosion Example (1/3)**: For a 64x64 pixel image (4,096 inputs), a hidden layer of 512 units = 2.1 million weights.
5. **Parameter Explosion Example (2/3)**: For a 1024x1024 pixel image (1,048,576 inputs), the same hidden layer = 537 million weights (~4 GB memory).
6. **Parameter Explosion Example (3/3)**: Visualizing the massive connectivity matrix (Sandfeld 19.1.1).
7. **Loss of Spatial Structure**: Nearby pixels are highly correlated; MLPs treat them as independent after flattening (Sandfeld 19.1.1).
8. **The Translation Invariance Challenge**: If a grain moves 5 pixels, an MLP sees it as a completely new input pattern.

## Section 2: The Convolution Layer (Slides 9-20)
9. **Introduction to Convolutions**: Inspiration from classical image processing filters (McClarren 6.1).
10. **The Discrete Convolution Formula**: $(I * K)_{m,n} = \sum \sum I_{m-i, n-j} K_{i,j}$ (Sandfeld 19.1.2).
11. **Visualizing the "Sliding Window"**: Animation/Diagram of a 3x3 kernel over a 5x5 image (McClarren Fig 6.4).
12. **The Feature Map**: How the output of a convolution represents detected features (Sandfeld 19.1.2).
13. **Kernels as Feature Detectors (1/2)**: Edge detection example using the Laplacian filter (McClarren 6.1.2, Fig 6.5).
14. **Kernels as Feature Detectors (2/2)**: Horizontal vs. Vertical edge kernels.
15. **Stride**: Controlling the step size of the window (Sandfeld 19.1.2).
16. **Stride Effects**: Reducing the spatial resolution of the feature map.
17. **Padding (1/2)**: Why images shrink after convolution (Sandfeld 19.1.2).
18. **Padding (2/2)**: "Same" padding (McClarren 6.1.1).
19. **Activation Functions in CNNs**: Applying ReLU/tanh to the convolved output (Sandfeld 18.2).
20. **Convolution vs. MLP Parameters**: Drastic reduction through local connectivity.

## Section 3: Architectural Principles (Slides 21-30)
21. **Local Connectivity**: Each neuron only "sees" a small local patch (Sandfeld 19.1.2).
22. **Weight Sharing**: Using the same kernel across the entire image (Sandfeld 19.1.2).
23. **Translation Invariance**: If an edge is learned, it can be detected anywhere (McClarren 6.1).
24. **Translation Equivariance**: The feature map moves with the input.
25. **Multi-Channel Convolution**: Handling RGB (3 channels) or multi-layered feature maps (McClarren 6.1.3, Fig 6.7).
26. **Visualizing Multi-Channel Output**: Sandfeld Fig 19.3.
27. **The Concept of Depth in CNNs**: Number of filters per layer.
28. **Receptive Field (1/2)**: The region of input that influences a specific neuron (Sandfeld 19.1.4).
29. **Receptive Field (2/2)**: How stacking layers increases the receptive field (Sandfeld 19.1.4).
30. **Effective Kernel Size**: Formula and intuition (Sandfeld Eq 19.3).

## Section 4: The Pooling Layer (Slides 31-38)
31. **Why Pool?**: Downsampling and noise robustness (Sandfeld 19.1.3, McClarren 6.2).
32. **Max Pooling**: Taking the maximum in a window (McClarren Eq 6.24).
33. **Intuition for Max Pooling**: Preserving sharp signals/peaks (McClarren 6.2).
34. **Average Pooling**: Taking the mean (McClarren Eq 6.23).
35. **Pooling and Shift Invariance**: Why pooling makes the network less sensitive to exact pixel locations (McClarren 6.2).
36. **Visualizing Pooling Effects**: McClarren Fig 6.6.
37. **Hierarchical Feature Representation (1/2)**: From edges to motifs to objects (Sandfeld 19.1.2).
38. **Hierarchical Feature Representation (2/2)**: Visualizing learned filters in deep layers (Sandfeld Fig 19.4 - AlexNet).

## Section 5: Key Architectures (Slides 39-44)
39. **LeNet-5 (1995)**: The ancestor of modern CNNs for digit recognition (Sandfeld 19.1.6, Fig 19.7).
40. **AlexNet (2012)**: The breakthrough in ImageNet - GPUs, ReLU, Dropout (Sandfeld 19.1.6, Fig 19.8).
41. **Deepening Networks**: The challenge of vanishing gradients (Sandfeld 19.1.6).
42. **ResNet (2015)**: Skip connections to enable 100+ layers (Sandfeld 19.1.6).
43. **The Residual Block**: $y = F(x) + x$ (Sandfeld Fig 19.9).
44. **U-Net**: Encoder-Decoder for semantic segmentation (Sandfeld 19.3.1).

## Section 6: Materials Science Case Studies (Slides 45-50)
45. **Phase Segmentation (TEM)**: Crystalline Au nanoparticles vs. amorphous background (Sandfeld 19.3.1, Fig 19.10).
46. **Synthetic Training Data**: Overcoming the label bottleneck with Voronoi grain microstructures (Sandfeld 19.3.2).
47. **Transferring Synthetic to Real**: Grain boundary segmentation in SEM (Sandfeld Fig 19.11).
48. **Property Prediction**: Ising model microstructures for temperature classification (Sandfeld 18.8, Fig 18.9).
49. **Defect Detection Intuition**: Highlighting dislocations/pores as specific features.
50. **Summary & Future Outlook**: CNNs as the backbone of autonomous microscopy.
