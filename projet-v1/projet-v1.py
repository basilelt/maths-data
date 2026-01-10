# Basile LE THIEC & Lilian NOACCO
# 2A Alt IR

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Traitement initial des données
def get_data():
    """Loads and preprocesses the digits dataset."""
    digits = load_digits()
    X = digits.data
    y = digits.target

    # One-hot encode labels for the custom model
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot

# 2. Implémentation from scratch
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, alpha=0.1):
        """
        Initialise le modèle de régression logistique.
        alpha: float, paramètre de régularisation L2.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha  # Paramètre de régularisation
        self.W = None
        self.b = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = y.shape[1]
        
        # Initialize parameters
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        # Gradient descent
        for _ in range(self.max_iter):
            z = X @ self.W.T + self.b
            probs = self._softmax(z)
            
            # Gradient calculation with L2 regularization
            grad_W = (1 / n_samples) * (probs - y).T @ X + self.alpha * self.W
            grad_b = (1 / n_samples) * np.sum(probs - y, axis=0)
            
            # Update parameters
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b

    def predict(self, X):
        z = X @ self.W.T + self.b
        probs = self._softmax(z)
        return np.argmax(probs, axis=1)

# --- Main script execution ---

# Get data
X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = get_data()

# Train and evaluate the "from scratch" model
print("Training custom model...")
custom_model = CustomLogisticRegression()
custom_model.fit(X_train, y_train_onehot)
y_pred_custom = custom_model.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print("Custom model training finished.")

# 3. Implémentation directe scikit-learn
print("\nTraining scikit-learn model...")
sklearn_model = LogisticRegression(max_iter=1000)
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print("Scikit-learn model training finished.")

# 4. Comparaison des résultats
print("\n--- Results Comparison ---")
print(f"Custom Model Test Accuracy:   {accuracy_custom:.4f}")
print(f"Scikit-learn Model Test Accuracy: {accuracy_sklearn:.4f}")

# 5. BONUS: Implémentation avec Régularisation L2
print("\n--- Bonus: L2 Regularization ---")
print("Training custom model with L2 regularization...")
# Utilisation d'un alpha pour la régularisation
regularized_model = CustomLogisticRegression(alpha=0.01)
regularized_model.fit(X_train, y_train_onehot)
y_pred_regularized = regularized_model.predict(X_test)
accuracy_regularized = accuracy_score(y_test, y_pred_regularized)
print(f"Regularized Custom Model Test Accuracy: {accuracy_regularized:.4f}")


# 6. Analyse des résultats
print("\n--- Results Analysis ---")

# Interprétation des coefficients
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
   # Reshape the coefficients for the current class into an 8x8 image
   image = custom_model.W[i].reshape(8, 8)
   ax.imshow(image, cmap='viridis')
   ax.set_title(f"Class {i}")
   ax.axis('off')
plt.suptitle("Learned Coefficients for Each Class")
plt.show()

# Analyse des erreurs
misclassified_indices = np.where(y_test != y_pred_custom)[0]
if len(misclassified_indices) > 0:
   print(f"\nFound {len(misclassified_indices)} misclassified images.")
   # Display up to 5 misclassified images
   num_images_to_show = min(len(misclassified_indices), 5)
   fig, axes = plt.subplots(1, num_images_to_show, figsize=(10, 3))
   for i in range(num_images_to_show):
       idx = misclassified_indices[i]
       image = X_test[idx].reshape(8, 8)
       # Handle case where axes is not an array for a single subplot
       if num_images_to_show == 1:
           ax = axes
       else:
           ax = axes[i]
       ax.imshow(image, cmap='gray')
       ax.set_title(f"Pred: {y_pred_custom[idx]}, True: {y_test[idx]}")
       ax.axis('off')
   plt.suptitle("Misclassified Images")
   plt.show()
else:
   print("\nNo misclassified images to display.")

# 7. Prise de recul
print("\n--- Reflection ---")
print("""
Influence des paramètres:
- Le 'learning_rate' contrôle la taille des pas lors de la descente de gradient. Un taux trop élevé peut entraîner une divergence, tandis qu'un taux trop bas peut ralentir la convergence.
- 'max_iter' est le nombre maximum d'itérations. Un nombre plus élevé permet au modèle de converger, mais peut aussi conduire à un surapprentissage si non contrôlé.

Difficultés et limites:
- L'implémentation de la fonction softmax a nécessité une attention particulière pour éviter les problèmes de stabilité numérique (overflow) en soustrayant la valeur maximale de 'z'.
- Le modèle de régression logistique est un modèle linéaire, il peut donc avoir du mal à séparer des classes qui ne sont pas linéairement séparables dans l'espace des caractéristiques.
- Le choix des hyperparamètres (learning_rate, max_iter) est crucial et nécessite souvent une validation croisée pour trouver les valeurs optimales.
""")