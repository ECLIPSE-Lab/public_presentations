from PIL import Image
import numpy as np

img = Image.open('umap_mnist_coil20_comparison.png').convert('L')
data = np.array(img)
# Invert so text/plots are positive, white background is 0
if data.mean() > 127: # assuming white background
    data = 255 - data

# Sum across columns to get row-wise activity
row_sums = data.sum(axis=1)

# Find where the image is active
active_rows = np.where(row_sums > 1000)[0]
if len(active_rows) > 0:
    first_active = active_rows[0]
    last_active = active_rows[-1]
    
    # Look for a gap (large white space) near the bottom, which might separate the caption
    # Let's search from the bottom up to the middle
    gap_found = False
    for r in range(last_active, first_active, -1):
        if row_sums[r] < 1000 and row_sums[r-1] < 1000 and row_sums[r-2] < 1000 and row_sums[r-3] < 1000 and row_sums[r-4] < 1000 and row_sums[r-5] < 1000:
            # We found a gap of at least 6 pixels
            # The plot is above this gap, the caption is below
            print(f"Gap found at row {r}, cropping bottom part.")
            img_cropped = img.crop((0, first_active, img.width, r))
            img_cropped.save('umap_mnist_coil20_comparison_cropped.png')
            gap_found = True
            break
    
    if not gap_found:
        print("No gap found, just trimming whitespace")
        img_cropped = img.crop((0, first_active, img.width, last_active))
        img_cropped.save('umap_mnist_coil20_comparison_cropped.png')
