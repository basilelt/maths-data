# Projet de classification
# Basile et Lilian

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
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.0, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.classes = None
        self.loss_history = []

    def softmax(self, z):
        # on retire le max pour la stabilite num
        z_max = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = len(y_true)

        # one hot encoding
        y_one_hot = np.eye(len(self.classes))[y_true]

        # clip pour pas avoir log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=1))

        # terme de regul L2
        if self.lambda_reg > 0:
            loss += (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)

        return loss

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        # init des poids
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        m = X.shape[0]

        # mapping des classes 0 a n-1
        y_mapped = np.array([np.where(self.classes == yi)[0][0] for yi in y])

        # boucle d'entrainement
        for i in range(self.n_iterations):
            # forward prop
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(z)

            y_one_hot = np.eye(n_classes)[y_mapped]

            loss = self.cross_entropy_loss(y_pred, y_mapped)
            self.loss_history.append(loss)

            # backward prop / gradients
            error = (y_pred - y_one_hot) / m
            dw = np.dot(X.T, error)
            db = np.sum(error, axis=0, keepdims=True)

            # gradient de la reg
            if self.lambda_reg > 0:
                dw += (self.lambda_reg / m) * self.weights

            # update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if (i + 1) % 100 == 0:
                print(f"iter {i + 1}, loss: {loss:.4f}")

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)

    def predict(self, X):
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes[class_indices]


def main():
    print("Projet Classification Logistique - from scratch")
    print("---------------------------------------------")

    # chargement dataset
    print("\nLoading data...")
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    print(f"Shape: {X.shape}")
    print(f"Nb classes: {len(np.unique(y))}")

    # split et scaling
    print("\nPreprocessing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Notre modele
    print("\nTraining custom model...")
    model_custom = LogisticRegressionFromScratch(
        learning_rate=0.1, n_iterations=500, lambda_reg=0.001, random_state=42
    )
    model_custom.fit(X_train_scaled, y_train)

    y_train_pred_custom = model_custom.predict(X_train_scaled)
    y_test_pred_custom = model_custom.predict(X_test_scaled)

    train_acc_custom = accuracy_score(y_train, y_train_pred_custom)
    test_acc_custom = accuracy_score(y_test, y_test_pred_custom)

    print(f"Train acc: {train_acc_custom:.4f}")
    print(f"Test acc: {test_acc_custom:.4f}")

    # Sklearn pour comparer
    print("\nTraining sklearn...")
    model_sklearn = LogisticRegression(
        max_iter=1000, solver="lbfgs", random_state=42, C=1.0
    )
    model_sklearn.fit(X_train_scaled, y_train)

    y_test_pred_sklearn = model_sklearn.predict(X_test_scaled)
    test_acc_sklearn = accuracy_score(y_test, y_test_pred_sklearn)

    print(f"Sklearn Test acc: {test_acc_sklearn:.4f}")

    # Resultats
    print("\nComparaison:")
    print(f"Diff: {abs(test_acc_custom - test_acc_sklearn):.4f}")

    # metrics
    print("\nReport Custom:")
    print(classification_report(y_test, y_test_pred_custom))

    # save csv
    results = pd.DataFrame({
        "Modele": ["Custom", "Sklearn"],
        "Test Acc": [test_acc_custom, test_acc_sklearn]
    })

    results.to_csv("results.csv", index=False)
    print("saved to results.csv")

    return model_custom, model_sklearn


if __name__ == "__main__":
    main()
