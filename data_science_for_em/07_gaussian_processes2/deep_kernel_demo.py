#!/usr/bin/env python3
"""
Deep Kernel Learning Demonstration
==================================

This script demonstrates key concepts from Deep Kernel Learning:
1. Basic kernel functions (RBF vs Spectral Mixture)
2. Deep feature transformation
3. Kernel composition
4. Scalability considerations

Based on Wilson et al. (2016) "Deep Kernel Learning"
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import time

def rbf_kernel(x1, x2, length_scale=1.0):
    """
    RBF (Squared Exponential) kernel function
    """
    return np.exp(-0.5 * np.sum((x1 - x2)**2) / length_scale**2)

def spectral_mixture_kernel(x1, x2, weights, frequencies, length_scales):
    """
    Simplified Spectral Mixture kernel with Q components
    """
    result = 0
    for q in range(len(weights)):
        # Simplified version: RBF component + cosine component
        rbf_part = np.exp(-0.5 * np.sum((x1 - x2)**2) / length_scales[q]**2)
        cos_part = np.cos(2 * np.pi * frequencies[q] * np.sum(x1 - x2))
        result += weights[q] * rbf_part * cos_part
    return result

def deep_feature_transform(X, network_weights=None):
    """
    Simple deep feature transformation using a neural network
    """
    if network_weights is None:
        # Simple transformation: non-linear mapping
        return np.tanh(X @ np.random.randn(X.shape[1], 2))
    else:
        # Use provided weights (simplified)
        return np.tanh(X @ network_weights)

def deep_kernel(x1, x2, base_kernel='rbf', network_weights=None, **kernel_params):
    """
    Deep kernel: compose base kernel with deep feature transformation
    """
    # Transform inputs through deep network
    g1 = deep_feature_transform(x1.reshape(1, -1), network_weights).flatten()
    g2 = deep_feature_transform(x2.reshape(1, -1), network_weights).flatten()
    
    # Apply base kernel to transformed features
    if base_kernel == 'rbf':
        return rbf_kernel(g1, g2, kernel_params.get('length_scale', 1.0))
    elif base_kernel == 'spectral_mixture':
        return spectral_mixture_kernel(
            g1, g2, 
            kernel_params.get('weights', [1.0]),
            kernel_params.get('frequencies', [1.0]),
            kernel_params.get('length_scales', [1.0])
        )
    else:
        raise ValueError(f"Unknown base kernel: {base_kernel}")

def generate_step_function_data(n_samples=1000, noise=0.1):
    """
    Generate step function data with discontinuities
    """
    X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
    y = np.where(X < -2, -1.0, 
                 np.where(X < 0, 1.0,
                          np.where(X < 2, -1.0, 1.0)))
    y = y.astype(float)  # Ensure y is float type
    y += noise * np.random.randn(n_samples, 1)
    return X, y.flatten()

def compare_kernels():
    """
    Compare different kernel approaches on step function data
    """
    print("=== Deep Kernel Learning Demonstration ===\n")
    
    # Generate data
    X, y = generate_step_function_data(n_samples=200, noise=0.05)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target: Step function with discontinuities\n")
    
    # Test different approaches
    approaches = {
        'Standard RBF': {'kernel': 'rbf', 'deep': False},
        'Deep RBF': {'kernel': 'rbf', 'deep': True},
        'Deep Spectral Mixture': {'kernel': 'spectral_mixture', 'deep': True}
    }
    
    results = {}
    
    for name, config in approaches.items():
        print(f"Testing {name}...")
        start_time = time.time()
        
        if config['deep']:
            # Deep kernel approach
            network_weights = np.random.randn(X.shape[1], 2) * 0.1
            
            if config['kernel'] == 'spectral_mixture':
                kernel_params = {
                    'weights': [0.5, 0.5],
                    'frequencies': [1.0, 2.0],
                    'length_scales': [0.5, 1.0]
                }
            else:
                kernel_params = {'length_scale': 1.0}
            
            # Create custom kernel matrix
            n = X.shape[0]
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = deep_kernel(X[i], X[j], config['kernel'], 
                                        network_weights, **kernel_params)
            
            # Add noise
            K += 0.01 * np.eye(n)
            
            # Solve GP equations
            try:
                alpha = np.linalg.solve(K, y)
                y_pred = K @ alpha
                mse = np.mean((y - y_pred)**2)
            except np.linalg.LinAlgError:
                mse = float('inf')
                
        else:
            # Standard GP approach
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, random_state=42)
            gp.fit(X, y)
            y_pred = gp.predict(X)
            mse = np.mean((y - y_pred)**2)
        
        elapsed_time = time.time() - start_time
        results[name] = {'mse': mse, 'time': elapsed_time}
        
        print(f"  MSE: {mse:.4f}")
        print(f"  Time: {elapsed_time:.3f}s\n")
    
    return results, X, y

def scalability_analysis():
    """
    Demonstrate scalability considerations
    """
    print("=== Scalability Analysis ===\n")
    
    sizes = [50, 100, 200, 500, 1000]
    times_standard = []
    times_deep = []
    
    for n in sizes:
        print(f"Testing with {n} samples...")
        
        # Generate data
        X = np.random.randn(n, 1)
        y = np.sin(X.flatten()) + 0.1 * np.random.randn(n)
        
        # Standard GP timing
        start_time = time.time()
        kernel = RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01)
        gp.fit(X, y)
        elapsed_standard = time.time() - start_time
        times_standard.append(elapsed_standard)
        
        # Deep kernel timing (simplified)
        start_time = time.time()
        network_weights = np.random.randn(X.shape[1], 2) * 0.1
        K = np.zeros((n, n))
        for i in range(min(n, 50)):  # Limit for demonstration
            for j in range(min(n, 50)):
                K[i, j] = deep_kernel(X[i], X[j], 'rbf', network_weights)
        elapsed_deep = time.time() - start_time
        times_deep.append(elapsed_deep)
        
        print(f"  Standard GP: {elapsed_standard:.3f}s")
        print(f"  Deep Kernel: {elapsed_deep:.3f}s")
    
    # Plot scalability
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times_standard, 'o-', label='Standard GP', linewidth=2)
    plt.plot(sizes, times_deep, 's-', label='Deep Kernel', linewidth=2)
    plt.xlabel('Number of samples')
    plt.ylabel('Training time (s)')
    plt.title('Scalability Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Theoretical complexity
    plt.subplot(1, 2, 2)
    n_theory = np.array(sizes)
    plt.plot(n_theory, (n_theory/100)**3, '--', label='O(n³) - Standard GP', linewidth=2)
    plt.plot(n_theory, n_theory/100, '--', label='O(n) - KISS-GP', linewidth=2)
    plt.xlabel('Number of samples')
    plt.ylabel('Relative complexity')
    plt.title('Theoretical Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_kernels():
    """
    Visualize different kernel functions
    """
    print("=== Kernel Visualization ===\n")
    
    # Create test points
    x_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    x_ref = np.array([[0.0]])  # Reference point
    
    # Compute kernel values
    k_rbf = [rbf_kernel(x, x_ref.flatten()) for x in x_test]
    
    # Spectral mixture kernel
    weights = [0.6, 0.4]
    frequencies = [1.0, 2.0]
    length_scales = [0.8, 1.2]
    k_sm = [spectral_mixture_kernel(x.flatten(), x_ref.flatten(), 
                                  weights, frequencies, length_scales) 
            for x in x_test]
    
    # Deep kernels
    network_weights = np.random.randn(1, 2) * 0.5
    k_deep_rbf = [deep_kernel(x.flatten(), x_ref.flatten(), 'rbf', network_weights) 
                  for x in x_test]
    k_deep_sm = [deep_kernel(x.flatten(), x_ref.flatten(), 'spectral_mixture', 
                           network_weights, weights=weights, 
                           frequencies=frequencies, length_scales=length_scales) 
                 for x in x_test]
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x_test, k_rbf, 'b-', linewidth=2, label='RBF Kernel')
    plt.xlabel('x')
    plt.ylabel('k(x, 0)')
    plt.title('Standard RBF Kernel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(x_test, k_sm, 'r-', linewidth=2, label='Spectral Mixture')
    plt.xlabel('x')
    plt.ylabel('k(x, 0)')
    plt.title('Spectral Mixture Kernel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(x_test, k_deep_rbf, 'g-', linewidth=2, label='Deep RBF')
    plt.xlabel('x')
    plt.ylabel('k(x, 0)')
    plt.title('Deep RBF Kernel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(x_test, k_deep_sm, 'm-', linewidth=2, label='Deep Spectral Mixture')
    plt.xlabel('x')
    plt.ylabel('k(x, 0)')
    plt.title('Deep Spectral Mixture Kernel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Deep Kernel Learning Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    results, X, y = compare_kernels()
    visualize_kernels()
    scalability_analysis()
    
    print("\n=== Summary ===")
    print("Key insights from Deep Kernel Learning:")
    print("1. Deep kernels combine neural network flexibility with GP interpretability")
    print("2. Spectral mixture kernels are more expressive than RBF kernels")
    print("3. Scalability is crucial for practical applications")
    print("4. KISS-GP enables linear scaling while maintaining accuracy")
    print("\nFor more details, see Wilson et al. (2016) 'Deep Kernel Learning'") 