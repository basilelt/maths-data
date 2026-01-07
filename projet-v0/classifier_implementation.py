"""
Authors: Basile LE THIEC, Lilian NOACCO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


class LogisticRegressionFromScratch:
    def __init__(
        self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.0, random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.classes = None
        self.loss_history = []

    def softmax(self, z):
        # Subtract max for numerical stability (prevents overflow)
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = len(y_true)

        # One-hot encode y_true
        y_one_hot = np.eye(len(self.classes))[y_true]

        # Clip predictions to avoid log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # Cross-entropy loss
        loss = -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=1))

        # Add L2 regularization term
        if self.lambda_reg > 0:
            loss += (self.lambda_reg / (2 * m)) * np.sum(self.weights**2)

        return loss

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        # Initialize weights and bias
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        m = X.shape[0]

        # Map y to class indices (0 to n_classes-1)
        y_mapped = np.array([np.where(self.classes == yi)[0][0] for yi in y])

        # Gradient descent iterations
        for iteration in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(z)

            # One-hot encode
            y_one_hot = np.eye(n_classes)[y_mapped]

            # Compute loss
            loss = self.cross_entropy_loss(y_pred, y_mapped)
            self.loss_history.append(loss)

            # Backward pass - compute gradients
            error = (y_pred - y_one_hot) / m
            dw = np.dot(X.T, error)
            db = np.sum(error, axis=0, keepdims=True)

            # Add regularization gradient
            if self.lambda_reg > 0:
                dw += (self.lambda_reg / m) * self.weights

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print progress
            if (iteration + 1) % 100 == 0:
                print(
                    f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.4f}"
                )

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)

    def predict(self, X):
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes[class_indices]


def main():
    print("=" * 60)
    print("Multiclass Logistic Regression Classifier Project")
    print("=" * 60)

    # Load the digits dataset
    print("\n1. LOADING DATASET")
    print("-" * 60)
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Image size: 8x8 = 64 features")
    print(f"Pixel value range: {X.min()} to {X.max()}")
    print(f"Dataset distribution:\n{pd.Series(y).value_counts().sort_index()}")

    # Split and scale data
    print("\n2. DATA PREPROCESSING")
    print("-" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    print(f"Scaled data mean: {X_train_scaled.mean():.6f}")
    print(f"Scaled data std: {X_train_scaled.std():.6f}")

    # Train custom implementation
    print("\n3. TRAINING CUSTOM IMPLEMENTATION")
    print("-" * 60)
    model_custom = LogisticRegressionFromScratch(
        learning_rate=0.1, n_iterations=500, lambda_reg=0.001, random_state=42
    )
    model_custom.fit(X_train_scaled, y_train)

    y_train_pred_custom = model_custom.predict(X_train_scaled)
    y_test_pred_custom = model_custom.predict(X_test_scaled)

    train_acc_custom = accuracy_score(y_train, y_train_pred_custom)
    test_acc_custom = accuracy_score(y_test, y_test_pred_custom)

    print(f"\nCustom Implementation Results:")
    print(f"Training Accuracy: {train_acc_custom:.4f}")
    print(f"Test Accuracy: {test_acc_custom:.4f}")

    # Train scikit-learn implementation
    print("\n4. TRAINING SCIKIT-LEARN IMPLEMENTATION")
    print("-" * 60)
    model_sklearn = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=42, C=1.0
    )
    model_sklearn.fit(X_train_scaled, y_train)

    y_train_pred_sklearn = model_sklearn.predict(X_train_scaled)
    y_test_pred_sklearn = model_sklearn.predict(X_test_scaled)

    train_acc_sklearn = accuracy_score(y_train, y_train_pred_sklearn)
    test_acc_sklearn = accuracy_score(y_test, y_test_pred_sklearn)

    print(f"\nScikit-Learn Implementation Results:")
    print(f"Training Accuracy: {train_acc_sklearn:.4f}")
    print(f"Test Accuracy: {test_acc_sklearn:.4f}")

    # Comparison
    print("\n5. COMPARISON RESULTS")
    print("-" * 60)
    print(f"Test Accuracy Difference: {abs(test_acc_custom - test_acc_sklearn):.4f}")
    print(f"Custom model test errors: {np.sum(y_test != y_test_pred_custom)}/360")
    print(f"Sklearn model test errors: {np.sum(y_test != y_test_pred_sklearn)}/360")

    # Detailed metrics
    print("\n6. DETAILED METRICS")
    print("-" * 60)
    print("\nCustom Implementation - Classification Report:")
    print(classification_report(y_test, y_test_pred_custom))

    print("\nScikit-Learn Implementation - Classification Report:")
    print(classification_report(y_test, y_test_pred_sklearn))

    # Save results
    results_summary = pd.DataFrame(
        {
            "Metric": [
                "Training Accuracy",
                "Test Accuracy",
                "Number of Test Errors",
                "Error Rate (%)",
            ],
            "Custom Implementation": [
                f"{train_acc_custom:.4f}",
                f"{test_acc_custom:.4f}",
                f"{np.sum(y_test != y_test_pred_custom)}",
                f"{np.sum(y_test != y_test_pred_custom)/len(y_test)*100:.2f}%",
            ],
            "Scikit-Learn Implementation": [
                f"{train_acc_sklearn:.4f}",
                f"{test_acc_sklearn:.4f}",
                f"{np.sum(y_test != y_test_pred_sklearn)}",
                f"{np.sum(y_test != y_test_pred_sklearn)/len(y_test)*100:.2f}%",
            ],
        }
    )

    results_summary.to_csv("results_comparison.csv", index=False)
    print("\n✓ Results saved to 'results_comparison.csv'")

    return model_custom, model_sklearn, X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    model_custom, model_sklearn, X_train_scaled, X_test_scaled, y_train, y_test = main()
