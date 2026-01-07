import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()

# Extract data and targets
X = digits.data
y = digits.target

# Basic dataset information
print("Dataset structure:")
print(f"- Number of samples: {X.shape[0]}")
print(f"- Number of features per sample: {X.shape[1]}")
print(f"- Image dimensions: 8x8 pixels (flattened to {X.shape[1]} features)")
print(f"- Data type: {X.dtype}")
print(f"- Target shape: {y.shape}")
print(f"- Unique digit classes: {np.unique(y)}")
print(f"- Class distribution: {np.bincount(y)}")

# Basic statistics
print("\nBasic statistics:")
print(f"- Pixel value range: {X.min()} to {X.max()}")
print(f"- Mean pixel value: {X.mean():.2f}")
print(f"- Standard deviation: {X.std():.2f}")
print(f"- Median pixel value: {np.median(X):.2f}")

# Visualize sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("Sample Images from Digits Dataset", fontsize=16)

for i, ax in enumerate(axes.flat):
    # Reshape flattened image back to 8x8
    image = X[i].reshape(8, 8)
    ax.imshow(image, cmap="gray", interpolation="nearest")
    ax.set_title(f"Digit: {y[i]}", fontsize=12)
    ax.axis("off")

plt.tight_layout()
import os

results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)
plt.savefig(
    os.path.join(results_dir, "digits_samples.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# Preprocessing demonstration
print("\nPreprocessing:")
# Normalize pixel values to [0, 1] range
X_normalized = X / 16.0
print(
    f"- After normalization: range [{X_normalized.min():.3f}, {X_normalized.max():.3f}]"
)

# Flatten is already done, but show how to reshape
print("- Images are already flattened for ML algorithms")
print("- To reconstruct: use .reshape(8, 8) as shown in visualization")

# Additional insights
print("\nDataset insights:")
print("- Each image is 8x8 grayscale (64 pixels)")
print("- Pixel values range from 0 (black) to 16 (white)")
print("- Balanced dataset with ~180 samples per digit")
print("- Total: 1797 handwritten digit images")
print("- Perfect for classification tasks")
