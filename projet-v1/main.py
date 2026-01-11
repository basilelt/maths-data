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
    """Charge et prétraite le jeu de données digits"""
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Encodage one-hot des étiquettes pour le modèle personnalisé
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    # Division des données
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
        Initialise le modèle de régression logistique
        alpha: float, paramètre de régularisation L2
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

        # Initialisation des paramètres
        self.W = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        # Descente de gradient
        for _ in range(self.max_iter):
            z = X @ self.W.T + self.b
            probs = self._softmax(z)

            # Calcul du gradient avec régularisation L2
            grad_W = (1 / n_samples) * (probs - y).T @ X + self.alpha * self.W
            grad_b = (1 / n_samples) * np.sum(probs - y, axis=0)

            # Mise à jour des paramètres
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b

    def predict(self, X):
        z = X @ self.W.T + self.b
        probs = self._softmax(z)
        return np.argmax(probs, axis=1)


## Script principal

# Obtenir les données
X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = get_data()

# Entraîner et évaluer le modèle "from scratch"
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
print("\n### Results Comparison ###")
print(f"Custom Model Test Accuracy:   {accuracy_custom:.4f}")
print(f"Scikit-learn Model Test Accuracy: {accuracy_sklearn:.4f}")

# 5. BONUS: Implémentation avec Régularisation L2
print("\n### Bonus: L2 Regularization ###")
print("Training custom model with L2 regularization...")
# Utilisation d'un alpha pour la régularisation
regularized_model = CustomLogisticRegression(alpha=0.01)
regularized_model.fit(X_train, y_train_onehot)
y_pred_regularized = regularized_model.predict(X_test)
accuracy_regularized = accuracy_score(y_test, y_pred_regularized)
print(f"Regularized Custom Model Test Accuracy: {accuracy_regularized:.4f}")


# 6. Analyse des résultats
print("\n### Results Analysis ###")

# Interprétation des coefficients
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    # Remodeler les coefficients pour la classe actuelle en une image 8x8
    image = custom_model.W[i].reshape(8, 8)
    ax.imshow(image, cmap="viridis")
    ax.set_title(f"Classe {i}")
    ax.axis("off")
plt.suptitle("Coefficients appris pour chaque classe")
plt.savefig("coefficients.png")
plt.close()

# Analyse des erreurs
misclassified_indices = np.where(y_test != y_pred_custom)[0]
if len(misclassified_indices) > 0:
    print(f"\nFound {len(misclassified_indices)} misclassified images.")
    # Afficher jusqu'à 5 images mal classifiées
    num_images_to_show = min(len(misclassified_indices), 5)
    fig, axes = plt.subplots(1, num_images_to_show, figsize=(10, 3))
    for i in range(num_images_to_show):
        idx = misclassified_indices[i]
        image = X_test[idx].reshape(8, 8)
        # Gérer le cas où axes n'est pas un tableau pour un seul sous-graphique
        if num_images_to_show == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Préd: {y_pred_custom[idx]}, Vrai: {y_test[idx]}")
        ax.axis("off")
    plt.suptitle("Images mal classifiées")
    plt.savefig("misclassified.png")
    plt.close()
else:
    print("\nNo misclassified images to display.")

# 7. Prise de recul
print("\nReflection")
print(
    """
Influence des paramètres:
- Le 'learning_rate' contrôle la taille des pas lors de la descente de gradient. Un taux trop élevé peut entraîner une divergence, tandis qu'un taux trop bas peut ralentir la convergence.
- 'max_iter' est le nombre maximum d'itérations. Un nombre plus élevé permet au modèle de converger, mais peut aussi conduire à un surapprentissage si non contrôlé.

Difficultés et limites:
- L'implémentation de la fonction softmax a nécessité une attention particulière pour éviter les problèmes de stabilité numérique (overflow) en soustrayant la valeur maximale de 'z'.
- Le modèle de régression logistique est un modèle linéaire, il peut donc avoir du mal à séparer des classes qui ne sont pas linéairement séparables dans l'espace des caractéristiques.
- Le choix des hyperparamètres (learning_rate, max_iter) est crucial et nécessite souvent une validation croisée pour trouver les valeurs optimales.
"""
)
