import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from multi_class_logistic import MultiClassLogisticRegression, GradientDescent
from functools import partial
import pickle

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split into train/test sets (80/20) - same as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)
y_test_labels = np.argmax(y_test, axis=1)

# Load the trained model
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Loaded trained model from 'models/trained_model.pkl'")

# ============================================================================
# 1. PARAMETER INFLUENCE ANALYSIS
# ============================================================================


def train_with_tracking(learning_rate, max_iter, X_train, y_train):
    """Train model and track loss convergence"""
    model = MultiClassLogisticRegression(learning_rate=learning_rate, max_iter=max_iter)

    # Modify to track loss during training
    n_samples, n_features = X_train.shape
    n_classes = y_train.shape[1]

    # Initialize parameters
    W = np.random.randn(n_classes, n_features) * 0.01
    b = np.zeros(n_classes)
    params = np.concatenate([W.flatten(), b.flatten()])

    # Create gradient function
    grad_func = partial(model._gradient, X=X_train, y=y_train)

    # Track loss over iterations
    loss_history = []
    current_params = params.copy()

    gd = GradientDescent(grad_func, learning_rate, max_iter)

    # Override descent to track loss
    step = learning_rate
    for i in range(max_iter):
        current_gradient = grad_func(current_params)
        current_params = current_params - step * current_gradient

        # Calculate current loss
        W_current = current_params[: n_features * n_classes].reshape(
            n_classes, n_features
        )
        b_current = current_params[n_features * n_classes :]
        z = X_train @ W_current.T + b_current
        probs = model._softmax(z)
        loss = model._cross_entropy_loss(y_train, probs)
        loss_history.append(loss)

    # Set final parameters
    model.W = current_params[: n_features * n_classes].reshape(n_classes, n_features)
    model.b = current_params[n_features * n_classes :]

    return model, loss_history


# Test different learning rates and max iterations
learning_rates = [0.001, 0.01, 0.1, 0.5]
max_iters = [100, 500, 1000, 2000]

results = {}
convergence_data = {}

for lr in learning_rates:
    for max_iter in max_iters:
        print(f"Training with lr={lr}, max_iter={max_iter}")
        model_temp, loss_history = train_with_tracking(lr, max_iter, X_train, y_train)

        # Calculate accuracies
        y_train_pred = model_temp.predict(X_train)
        y_test_pred = model_temp.predict(X_test)
        train_acc = accuracy_score(np.argmax(y_train, axis=1), y_train_pred)
        test_acc = accuracy_score(y_test_labels, y_test_pred)

        key = f"lr_{lr}_iter_{max_iter}"
        results[key] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "lr": lr,
            "max_iter": max_iter,
        }
        convergence_data[key] = loss_history

# Plot convergence for different parameter combinations
plt.figure(figsize=(15, 10))

# Plot convergence curves
plt.subplot(2, 2, 1)
for lr in learning_rates:
    for max_iter in max_iters:
        key = f"lr_{lr}_iter_{max_iter}"
        loss_history = convergence_data[key]
        plt.plot(loss_history, label=f"lr={lr}, iter={max_iter}", alpha=0.7)

plt.xlabel("Iteration")
plt.ylabel("Cross-Entropy Loss")
plt.title("Convergence Curves")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.yscale("log")

# Plot final accuracies vs learning rate
plt.subplot(2, 2, 2)
for max_iter in max_iters:
    lrs = []
    train_accs = []
    test_accs = []
    for lr in learning_rates:
        key = f"lr_{lr}_iter_{max_iter}"
        lrs.append(lr)
        train_accs.append(results[key]["train_acc"])
        test_accs.append(results[key]["test_acc"])

    plt.plot(lrs, train_accs, "o-", label=f"Train (iter={max_iter})", alpha=0.7)
    plt.plot(lrs, test_accs, "s-", label=f"Test (iter={max_iter})", alpha=0.7)

plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Learning Rate")
plt.legend()
plt.xscale("log")

# Plot final accuracies vs max iterations
plt.subplot(2, 2, 3)
for lr in learning_rates:
    iters = []
    train_accs = []
    test_accs = []
    for max_iter in max_iters:
        key = f"lr_{lr}_iter_{max_iter}"
        iters.append(max_iter)
        train_accs.append(results[key]["train_acc"])
        test_accs.append(results[key]["test_acc"])

    plt.plot(iters, train_accs, "o-", label=f"Train (lr={lr})", alpha=0.7)
    plt.plot(iters, test_accs, "s-", label=f"Test (lr={lr})", alpha=0.7)

plt.xlabel("Max Iterations")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Max Iterations")
plt.legend()

# Plot test accuracy heatmap
plt.subplot(2, 2, 4)
lr_labels = [str(lr) for lr in learning_rates]
iter_labels = [str(iter) for iter in max_iters]
accuracy_matrix = np.zeros((len(learning_rates), len(max_iters)))

for i, lr in enumerate(learning_rates):
    for j, max_iter in enumerate(max_iters):
        key = f"lr_{lr}_iter_{max_iter}"
        accuracy_matrix[i, j] = results[key]["test_acc"]

sns.heatmap(
    accuracy_matrix,
    annot=True,
    fmt=".3f",
    xticklabels=iter_labels,
    yticklabels=lr_labels,
    cmap="viridis",
)
plt.xlabel("Max Iterations")
plt.ylabel("Learning Rate")
plt.title("Test Accuracy Heatmap")

plt.tight_layout()
plt.savefig("results/parameter_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

print(
    "Parameter influence analysis completed. Plot saved as 'results/parameter_analysis.png'"
)

# ============================================================================
# 2. COEFFICIENT INTERPRETATION
# ============================================================================

# Visualize learned weights as 8x8 images for each class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(10):
    # Reshape weights for class i to 8x8
    weights_image = model.W[i].reshape(8, 8)
    axes[i].imshow(weights_image, cmap="RdYlBu_r", interpolation="nearest")
    axes[i].set_title(f"Class {i} Weights")
    axes[i].axis("off")

plt.colorbar(
    axes[0].imshow(model.W[0].reshape(8, 8), cmap="RdYlBu_r"),
    ax=axes,
    orientation="horizontal",
    fraction=0.02,
    pad=0.04,
)
plt.suptitle("Learned Weights Visualization (8x8 pixels per class)")
plt.tight_layout()
plt.savefig("results/coefficient_visualization.png", dpi=300, bbox_inches="tight")
plt.show()

print(
    "Coefficient interpretation completed. Plot saved as 'results/coefficient_visualization.png'"
)

# ============================================================================
# 3. ERROR ANALYSIS
# ============================================================================

# Make predictions on test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test_labels, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Find misclassified samples
misclassified_indices = np.where(y_test_pred != y_test_labels)[0]
print(f"Number of misclassified samples: {len(misclassified_indices)}")

# Display some misclassified samples (up to 10)
num_to_show = min(10, len(misclassified_indices))
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(num_to_show):
    idx = misclassified_indices[i]
    image = X_test[idx].reshape(8, 8)
    true_label = y_test_labels[idx]
    pred_label = y_test_pred[idx]

    axes[i].imshow(image, cmap="gray", interpolation="nearest")
    axes[i].set_title(f"True: {true_label}, Pred: {pred_label}")
    axes[i].axis("off")

# Hide unused subplots
for i in range(num_to_show, 10):
    axes[i].axis("off")

plt.suptitle("Misclassified Test Samples")
plt.tight_layout()
plt.savefig("results/misclassified_samples.png", dpi=300, bbox_inches="tight")
plt.show()

print("Error analysis completed. Plot saved as 'results/misclassified_samples.png'")

# Print some statistics about misclassifications
from collections import Counter

misclassified_true = y_test_labels[misclassified_indices]
misclassified_pred = y_test_pred[misclassified_indices]

print("\nMisclassification Statistics:")
print("True label -> Predicted label (count):")
confusion_pairs = list(zip(misclassified_true, misclassified_pred))
pair_counts = Counter(confusion_pairs)
for (true, pred), count in sorted(pair_counts.items()):
    print(f"  {true} -> {pred}: {count}")

print("\nMost common misclassifications:")
most_common = pair_counts.most_common(5)
for (true, pred), count in most_common:
    print(f"  True {true} classified as {pred}: {count} times")

print("\nPossible reasons for misclassifications:")
print("1. Similar digit shapes (e.g., 1 vs 7, 3 vs 8, 4 vs 9)")
print("2. Poor handwriting quality in the original images")
print("3. Overlapping or ambiguous pixel patterns")
print("4. Limited model capacity for complex patterns")
print("5. Insufficient training data for certain digit variations")
