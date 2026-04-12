import matplotlib.pyplot as plt
import numpy as np
import os

def draw_cv_schematic(output_path='images/cross_validation.png'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 1]})
    fig.patch.set_facecolor('#ffffff')
    
    # Colors
    fold_colors = ['#4a90e2', '#e94e77', '#50e3c2', '#f5a623', '#9013fe']
    group_colors = ['#d1e8ff', '#ffdbe4', '#d2f9f0', '#ffeccc', '#eadaff']
    
    # 1. Standard 5-fold CV (Random)
    ax = axes[0]
    ax.axis('off')
    ax.set_title("Standard $k$-fold CV (Random split)", loc='left', fontsize=14, weight='bold', pad=15)
    
    # Let's say we have 25 samples
    n_samples = 25
    x = np.linspace(0, 10, n_samples)
    
    # Create blocks for standard CV
    for i in range(5):
        # We will show the 5 folds as rows of training/validation splits
        y_pos = 4 - i
        ax.text(-0.5, y_pos + 0.5, f"Fold {i+1}", va='center', ha='right', fontsize=11, weight='bold')
        for j in range(5):
            is_val = (j == i)
            color = fold_colors[i] if is_val else '#dddddd'
            rect = plt.Rectangle((j*2, y_pos), 1.9, 0.9, facecolor=color, edgecolor='none')
            ax.add_patch(rect)
            
            # Add text inside
            label = "Val" if is_val else "Train"
            text_color = "white" if is_val else "#555555"
            ax.text(j*2 + 0.95, y_pos + 0.45, label, va='center', ha='center', color=text_color, fontsize=10, weight='bold')
            
    ax.set_xlim(-1.5, 10)
    ax.set_ylim(-0.5, 5.5)

    # 2. Grouped CV (Grouped split)
    ax = axes[1]
    ax.axis('off')
    ax.set_title("Grouped / Blocked CV (Preserving families e.g. alloy type)", loc='left', fontsize=14, weight='bold', pad=15)
    
    for i in range(5):
        y_pos = 4 - i
        ax.text(-0.5, y_pos + 0.5, f"Fold {i+1}", va='center', ha='right', fontsize=11, weight='bold')
        for j in range(5):
            is_val = (j == i)
            # Differentiate blocks to show they are groups
            bg_color = group_colors[j] # groups are column-wise
            
            color = fold_colors[i] if is_val else bg_color
            border_color = 'white' if is_val else '#999999'
            
            rect = plt.Rectangle((j*2, y_pos), 1.9, 0.9, facecolor=color, edgecolor=border_color, lw=1.5)
            ax.add_patch(rect)
            
            # Add group name inside
            group_name = f"Alloy {chr(65+j)}"
            label = "Val" if is_val else "Train"
            text_color = "white" if is_val else "#333333"
            
            # Two lines of text: status and group
            ax.text(j*2 + 0.95, y_pos + 0.6, label, va='center', ha='center', color=text_color, fontsize=10, weight='bold')
            ax.text(j*2 + 0.95, y_pos + 0.3, group_name, va='center', ha='center', color=text_color, fontsize=8)

    ax.set_xlim(-1.5, 10)
    ax.set_ylim(-0.5, 5.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    draw_cv_schematic()
