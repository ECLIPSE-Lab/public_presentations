# Deep Kernel Learning - Lecture Materials

This directory contains teaching materials for Deep Kernel Learning, based on the seminal paper by Wilson et al. (2016).

## Files

### Presentation
- `deep_kernel_learning.qmd` - Main Quarto RevealJS presentation
- `template.qmd` - Template that includes the presentation
- `ref.bib` - Bibliography with all references

### Code
- `deep_kernel_demo.py` - Python demonstration script

### Original Paper
- `deepkernelarxiv.tex` - Original LaTeX paper (Wilson et al., 2016)

## Key Concepts Covered

### 1. The Big Question
- MacKay's question about combining neural networks and Gaussian processes
- Historical context of the neural network vs GP debate

### 2. Deep Kernel Learning Architecture
- Mathematical formulation: $k(\mathbf{x}_i, \mathbf{x}_j | \boldsymbol{\theta}) \rightarrow k(g(\mathbf{x}_i, \mathbf{w}), g(\mathbf{x}_j, \mathbf{w}) | \boldsymbol{\theta}, \mathbf{w})$
- Deep feature transformation with neural networks
- Base kernel composition (RBF vs Spectral Mixture)

### 3. Scalability with KISS-GP
- Linear scaling: $\mathcal{O}(n)$ training, $\mathcal{O}(1)$ prediction
- Kronecker and Toeplitz algebra for fast matrix operations
- Inducing point approximations

### 4. Experimental Results
- UCI regression datasets (16 datasets, up to 2M samples)
- Face orientation extraction
- Step function recovery
- Performance comparisons with standalone DNNs and GPs

## Running the Presentation

1. **Build the presentation:**
   ```bash
   quarto render template.qmd --to revealjs
   ```

2. **Run the demonstration:**
   ```bash
   python deep_kernel_demo.py
   ```

## Learning Objectives

After this lecture, students should understand:

1. **Why combine neural networks and Gaussian processes?**
   - Neural networks: Automatic representation discovery, inductive biases
   - Gaussian processes: Non-parametric flexibility, uncertainty quantification
   - Deep kernels: Best of both worlds

2. **How Deep Kernel Learning works:**
   - Transform inputs through deep architecture
   - Apply expressive base kernels (RBF, Spectral Mixture)
   - Joint learning through GP marginal likelihood

3. **Scalability challenges and solutions:**
   - Standard GPs: $\mathcal{O}(n^3)$ complexity
   - KISS-GP: $\mathcal{O}(n)$ scaling
   - Practical implications for large datasets

4. **When to use Deep Kernel Learning:**
   - High-dimensional structured data
   - Need for uncertainty quantification
   - Large-scale applications
   - When interpretability matters

## Key Insights

### Performance
- DKL consistently outperforms standalone DNNs and GPs
- Spectral mixture kernels provide significant improvements over RBF
- Minimal computational overhead (~10% additional cost)

### Scalability
- Linear scaling enables learning from large datasets
- KISS-GP maintains accuracy while providing speed
- Practical for real-world applications

### Interpretability
- Learned representations reveal meaningful structure
- Metric learning overcomes Euclidean distance limitations
- Uncertainty quantification for decision making

## References

The main reference is:
- Wilson, A. G., Hu, Z., Salakhutdinov, R., & Xing, E. P. (2016). Deep kernel learning. *Artificial intelligence and statistics*, 370-378.

Additional references are provided in `ref.bib` for further reading.

## Discussion Questions

1. How does DKL compare to other kernel learning approaches?
2. What are the computational trade-offs between different approaches?
3. When would you choose DKL over standalone DNNs or GPs?
4. How does the choice of base kernel affect performance?
5. What are the limitations of the current approach?

## Extensions

Potential areas for further exploration:
- Multi-task learning with deep kernels
- Temporal/spatial data applications
- High-dimensional classification
- Active learning and Bayesian optimization
- Integration with modern deep learning frameworks 