import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from multi_class_logistic import MultiClassLogisticRegression

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

# Convert one-hot back to labels for accuracy calculation
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("Training and evaluating models on digits dataset...")
print("=" * 50)

# Train non-regularized model
print("Training non-regularized model...")
model_no_reg = MultiClassLogisticRegression(
    learning_rate=0.01, max_iter=1000, reg_lambda=0.0
)
model_no_reg.fit(X_train, y_train)

y_train_pred_no_reg = model_no_reg.predict(X_train)
y_test_pred_no_reg = model_no_reg.predict(X_test)

train_acc_no_reg = accuracy_score(y_train_labels, y_train_pred_no_reg)
test_acc_no_reg = accuracy_score(y_test_labels, y_test_pred_no_reg)

print(f"Non-regularized Model - Training Accuracy: {train_acc_no_reg:.4f}")
print(f"Non-regularized Model - Test Accuracy: {test_acc_no_reg:.4f}")

# Train regularized model
print("\nTraining regularized model (lambda=0.01)...")
model_reg = MultiClassLogisticRegression(
    learning_rate=0.01, max_iter=1000, reg_lambda=0.01
)
model_reg.fit(X_train, y_train)

y_train_pred_reg = model_reg.predict(X_train)
y_test_pred_reg = model_reg.predict(X_test)

train_acc_reg = accuracy_score(y_train_labels, y_train_pred_reg)
test_acc_reg = accuracy_score(y_test_labels, y_test_pred_reg)

print(f"Regularized Model - Training Accuracy: {train_acc_reg:.4f}")
print(f"Regularized Model - Test Accuracy: {test_acc_reg:.4f}")

# Comparison
print("\n" + "=" * 50)
print("COMPARISON:")
print("=" * 50)
print(f"Training Accuracy Difference: {train_acc_reg - train_acc_no_reg:.4f}")
print(f"Test Accuracy Difference: {test_acc_reg - test_acc_no_reg:.4f}")

if test_acc_reg > test_acc_no_reg:
    print("Regularized model performs better on test set")
elif test_acc_reg < test_acc_no_reg:
    print("Non-regularized model performs better on test set")
else:
    print("Both models perform equally on test set")

# Detailed classification reports
print("\nNon-regularized Model Classification Report:")
print(classification_report(y_test_labels, y_test_pred_no_reg))

print("\nRegularized Model Classification Report:")
print(classification_report(y_test_labels, y_test_pred_reg))
