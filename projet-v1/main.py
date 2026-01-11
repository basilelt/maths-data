# Basile LE THIEC & Lilian NOACCO
# 2A Alt IR

import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from my_descent import GradientDescent
from tqdm import tqdm


# 1. Traitement initial des données
def get_data():
    """Charge et prétraite le jeu de données digits"""
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Encodage one-hot des étiquettes pour le modèle personnalisé
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot


# 2. Implémentation from scratch
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, alpha=0.0):
        """
        Initialise le modèle de régression logistique
        alpha: float, paramètre de régularisation L2 (0 pour sans régularisation)
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

        # Taille de theta
        theta_size = n_classes * n_features + n_classes
        theta = np.zeros(theta_size)

        def gradient(theta):
            W = theta[: n_classes * n_features].reshape(n_classes, n_features)
            b = theta[n_classes * n_features :]
            z = X @ W.T + b
            probs = self._softmax(z)
            grad_W = (1 / n_samples) * (probs - y).T @ X + self.alpha * W
            grad_b = (1 / n_samples) * np.sum(probs - y, axis=0)
            return np.concatenate([grad_W.flatten(), grad_b])

        # Utilisation de la classe GradientDescent
        descent_obj = GradientDescent(gradient, self.learning_rate, self.max_iter)
        optimal_theta, _ = descent_obj.descent(theta, taux_erreur=1e-6)

        # Reshape des paramètres optimaux
        self.W = optimal_theta[: n_classes * n_features].reshape(n_classes, n_features)
        self.b = optimal_theta[n_classes * n_features :]

    def predict(self, X):
        z = X @ self.W.T + self.b
        probs = self._softmax(z)
        return np.argmax(probs, axis=1)


# Fonction pour le réglage des hyperparamètres par validation croisée
def tune_hyperparameters(X, y_onehot, param_grid):
    best_params = None
    best_score = 0
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_labels = np.argmax(y_onehot, axis=1)

    total_combinations = len(param_grid["learning_rate"]) * len(param_grid["alpha"])

    with tqdm(total=total_combinations * 3, desc="Hyperparameter Tuning") as pbar:
        for lr in param_grid["learning_rate"]:
            for alpha in param_grid["alpha"]:
                scores = []
                for train_idx, val_idx in kf.split(X, y_labels):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y_onehot[train_idx], y_onehot[val_idx]
                    model = CustomLogisticRegression(
                        learning_rate=lr, alpha=alpha, max_iter=200
                    )
                    model.fit(X_train_fold, y_train_fold)
                    pred = model.predict(X_val_fold)
                    acc = accuracy_score(np.argmax(y_val_fold, axis=1), pred)
                    scores.append(acc)
                    pbar.update(1)  # Update progress for each fold
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {"learning_rate": lr, "alpha": alpha}
    print(f"Meilleurs paramètres trouvés : {best_params}, Score CV : {best_score:.4f}")
    return best_params, best_score


## Script principal

# Obtenir les données
X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = get_data()

# Réglage des hyperparamètres par validation croisée
param_grid = {"learning_rate": [0.001, 0.01, 0.1], "alpha": [0.0, 0.01, 0.1]}
best_params, best_cv_score = tune_hyperparameters(X_train, y_train_onehot, param_grid)
print(f"Meilleurs paramètres de la CV : {best_params}, Score CV : {best_cv_score:.4f}")

# Entraîner et évaluer le modèle "from scratch" avec les meilleurs paramètres
print("Formation du modèle personnalisé...")
custom_model = CustomLogisticRegression(
    learning_rate=best_params["learning_rate"], alpha=best_params["alpha"]
)
custom_model.fit(X_train, y_train_onehot)
y_pred_custom = custom_model.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print("Formation du modèle personnalisé terminée.")

# 3. Implémentation directe scikit-learn
print("\nFormation du modèle scikit-learn...")
sklearn_model = LogisticRegression(max_iter=1000)
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print("Formation du modèle scikit-learn terminée.")

# 4. Comparaison des résultats
print("\n### Comparaison des résultats ###")
print(f"Précision du test du modèle personnalisé :   {accuracy_custom:.4f}")
print(f"Précision du test du modèle scikit-learn : {accuracy_sklearn:.4f}")

# Matrices de confusion
print("\nMatrice de confusion du modèle personnalisé :")
print(confusion_matrix(y_test, y_pred_custom))
print("\nMatrice de confusion du modèle scikit-learn :")
print(confusion_matrix(y_test, y_pred_sklearn))

# Plot confusion matrices
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_custom, ax=ax, cmap="Blues")
plt.title("Matrice de Confusion - Modèle Personnalisé")
plt.savefig("confusion_matrix_custom.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_sklearn, ax=ax, cmap="Blues")
plt.title("Matrice de Confusion - Modèle Scikit-learn")
plt.savefig("confusion_matrix_sklearn.png")
plt.close()

# 5. BONUS: Implémentation avec Régularisation L2
print("\n### Bonus : Régularisation L2 ###")
print("Formation du modèle personnalisé avec régularisation L2...")
# Utilisation du meilleur alpha trouvé
regularized_model = CustomLogisticRegression(
    learning_rate=best_params["learning_rate"], alpha=best_params["alpha"]
)
regularized_model.fit(X_train, y_train_onehot)
y_pred_regularized = regularized_model.predict(X_test)
accuracy_regularized = accuracy_score(y_test, y_pred_regularized)
print(
    f"Précision du test du modèle personnalisé régularisé : {accuracy_regularized:.4f}"
)


# 6. Analyse des résultats
print("\n### Analyse des résultats ###")

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
    print(f"\nTrouvé {len(misclassified_indices)} images mal classifiées.")
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
    print("\nAucune image mal classifiée à afficher.")

# 8. BONUS: Test sur MNIST
print("\n### Bonus : Test sur MNIST ###")
print("Chargement du jeu de données MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X_mnist = mnist.data.astype(np.float32) / 255.0  # Normalisation
y_mnist = mnist.target.astype(int)

# Utiliser un sous-ensemble pour la rapidité
X_mnist_small, _, y_mnist_small, _ = train_test_split(
    X_mnist, y_mnist, train_size=10000, random_state=42, stratify=y_mnist
)
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(
    X_mnist_small, y_mnist_small, test_size=0.2, random_state=42, stratify=y_mnist_small
)

# Encodage one-hot
encoder_mnist = OneHotEncoder(sparse_output=False)
y_train_mnist_onehot = encoder_mnist.fit_transform(y_train_mnist.reshape(-1, 1))
y_test_mnist_onehot = encoder_mnist.transform(y_test_mnist.reshape(-1, 1))

print("Formation du modèle personnalisé sur MNIST...")
mnist_model = CustomLogisticRegression(
    learning_rate=best_params["learning_rate"],
    alpha=best_params["alpha"],
    max_iter=2000,
)  # Augmenter max_iter pour MNIST
mnist_model.fit(X_train_mnist, y_train_mnist_onehot)
y_pred_mnist = mnist_model.predict(X_test_mnist)
accuracy_mnist = accuracy_score(y_test_mnist, y_pred_mnist)
print(f"Précision du test MNIST : {accuracy_mnist:.4f}")

# 7. Prise de recul
print("\n### Réflexion ###")
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
