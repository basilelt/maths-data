import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from multi_class_logistic import MultiClassLogisticRegression
import pickle
import os

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(int)

# Take a subset for faster testing
X = X[:10000]  # 10k samples for faster testing
y = y[:10000]

print(f"MNIST subset: {X.shape[0]} samples, {X.shape[1]} features")

# Normalize pixel values to [0, 255] range like digits dataset
X = X  # MNIST is already 0-255, digits is 0-16, but we'll rescale

# Downsample MNIST 28x28 images to 8x8 to match digits dataset
print("Downsampling MNIST images from 28x28 to 8x8...")
X_downsampled = np.zeros((X.shape[0], 64))  # 8*8 = 64

for i in range(X.shape[0]):
    img_28x28 = X[i].reshape(28, 28)
    # Simple downsampling by averaging 3.5x3.5 blocks (28/8 = 3.5)
    # Use block averaging for better quality
    img_8x8 = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            r_start, r_end = r * 3, (r + 1) * 3 + (
                1 if r < 7 else 0
            )  # Handle uneven division
            c_start, c_end = c * 3, (c + 1) * 3 + (1 if c < 7 else 0)
            img_8x8[r, c] = np.mean(img_28x28[r_start:r_end, c_start:c_end])
    X_downsampled[i] = img_8x8.flatten()

X = X_downsampled

# Scale to match digits dataset range (approximately 0-16)
X = X * (16.0 / 255.0)

print(f"After downsampling: {X.shape[0]} samples, {X.shape[1]} features")
print(".2f")

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Load the trained model from digits dataset
models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
model_path = os.path.join(models_dir, "trained_model.pkl")

if not os.path.exists(model_path):
    print(f"Error: Trained model not found at {model_path}")
    print("Please run train_digits.py first to train the model.")
    exit(1)

with open(model_path, "rb") as f:
    model = pickle.load(f)

print(f"Loaded trained model from '{model_path}'")

# Test the model on MNIST
print("Testing model on MNIST dataset...")

# Make predictions
y_test_pred = model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracy
test_accuracy = accuracy_score(y_test_labels, y_test_pred)
print(".4f")

# Detailed classification report
print("\nClassification Report on MNIST:")
print(classification_report(y_test_labels, y_test_pred))

print("\nNote: The model was trained on 8x8 digits dataset and tested on 28x28 MNIST.")
print("This demonstrates the model's generalization capability (or lack thereof).")
print("Expected performance should be significantly lower than on the digits dataset.")

# Save results
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

results_file = os.path.join(results_dir, "mnist_test_results.txt")
with open(results_file, "w") as f:
    f.write("MNIST Test Results\n")
    f.write("==================\n\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test_labels, y_test_pred))

print(f"\nResults saved to '{results_file}'")
