import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from multi_class_logistic import MultiClassLogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split into train/test sets (80/20) - same as both models
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# One-hot encode labels for custom model
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train_raw.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test_raw.reshape(-1, 1))

print("Training Custom Multi-Class Logistic Regression...")
# Train custom model
custom_model = MultiClassLogisticRegression(learning_rate=0.01, max_iter=1000)
custom_model.fit(X_train, y_train_onehot)

# Predictions for custom model
y_train_pred_custom = custom_model.predict(X_train)
y_test_pred_custom = custom_model.predict(X_test)

print("Training Scikit-Learn Logistic Regression...")
# Train sklearn model
sklearn_model = LogisticRegression(max_iter=1000, random_state=42)
sklearn_model.fit(X_train, y_train_raw)

# Predictions for sklearn model
y_train_pred_sklearn = sklearn_model.predict(X_train)
y_test_pred_sklearn = sklearn_model.predict(X_test)


# Function to print detailed metrics
def print_metrics(y_true, y_pred, model_name, dataset_type):
    print(f"\n{model_name} - {dataset_type} Metrics:")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1-Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("\nPer-class metrics:")
    print(classification_report(y_true, y_pred, digits=4))


# Print metrics for both models
print_metrics(y_train_raw, y_train_pred_custom, "Custom Model", "Training")
print_metrics(y_test_raw, y_test_pred_custom, "Custom Model", "Test")
print_metrics(y_train_raw, y_train_pred_sklearn, "Scikit-Learn Model", "Training")
print_metrics(y_test_raw, y_test_pred_sklearn, "Scikit-Learn Model", "Test")


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name, dataset_type):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=digits.target_names,
        yticklabels=digits.target_names,
    )
    plt.title(f"Confusion Matrix - {model_name} ({dataset_type})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(
        f'results/confusion_matrix_{model_name.lower().replace(" ", "_")}_{dataset_type.lower()}.png',
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# Plot confusion matrices
plot_confusion_matrix(y_test_raw, y_test_pred_custom, "Custom Model", "Test")
plot_confusion_matrix(y_test_raw, y_test_pred_sklearn, "Scikit-Learn Model", "Test")

# Summary comparison
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON SUMMARY")
print("=" * 60)

test_accuracy_custom = accuracy_score(y_test_raw, y_test_pred_custom)
test_accuracy_sklearn = accuracy_score(y_test_raw, y_test_pred_sklearn)

print(".4f")
print(".4f")
print(".4f")

if test_accuracy_custom > test_accuracy_sklearn:
    print("Custom model performs better on test accuracy.")
elif test_accuracy_sklearn > test_accuracy_custom:
    print("Scikit-learn model performs better on test accuracy.")
else:
    print("Both models have identical test accuracy.")

print("\nConfusion matrices saved as PNG files.")
