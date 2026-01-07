import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from multi_class_logistic import MultiClassLogisticRegression
import pickle

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split into train/test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# Initialize and train the model
model = MultiClassLogisticRegression(learning_rate=0.01, max_iter=1000)
model.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert one-hot back to labels for accuracy calculation
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracies
train_accuracy = accuracy_score(y_train_labels, y_train_pred)
test_accuracy = accuracy_score(y_test_labels, y_test_pred)

# Print accuracies
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the trained model
import os

models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "trained_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved as '{model_path}'")
