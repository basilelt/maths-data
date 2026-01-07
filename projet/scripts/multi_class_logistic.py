import numpy as np
from functools import partial


class GradientDescent:
    def __init__(
        self, gradient, learning_rate: float = 0.01, max_iterations: int = 1000
    ):
        """
        Initialise l'objet GradientDescent avec les paramètres nécessaires.

        Paramètres :
        - gradient : La fonction gradient de la fonction de coût.
        - learning_rate : Taux d'apprentissage (pas) pour la mise à jour des paramètres.
        - max_iterations : Nombre maximal d'itérations de l'algorithme de descente.
        """
        self.gradient = gradient  # tuple de vecteurs (df/dx, df/dy, ...)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def descent(self, initial_point) -> int:
        """
        Effectue l'algorithme de descente de gradient.

        Paramètres :
        - initial_point : Le point de départ de l'algorithme.

        Retourne :
        - Le point optimal trouvé par l'algorithme.
        """
        step = self.learning_rate
        current_point = initial_point
        for i in range(self.max_iterations):
            current_gradient = self.gradient(current_point)
            current_point = self.update(current_point, current_gradient)

            # step = step / (i + 2)  # linearly decreasing
            # step = step / (i + 2) ** 2  # quadratically decreasing
            # step = step * np.exp(-self.beta * (i + 1))  # exponentially decreasing
            # step = step / (self.alpha * (i + 1) + 1)  # keras linearly decreasing

        return current_point

    def update(self, point, gradient_value):
        """
        Met à jour le point en utilisant le gradient et le taux d'apprentissage.

        Paramètres :
        - point : Le point à mettre à jour.
        - gradient_value : Le gradient de la fonction de coût au point donné.

        Retourne :
        - Le nouveau point après la mise à jour.
        """
        return point - self.learning_rate * gradient_value


class MultiClassLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, reg_lambda=0.0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.W = None
        self.b = None

    def _softmax(self, z):
        # z is (n_samples, n_classes)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _cross_entropy_loss(self, y_true, y_pred):
        # y_true is one-hot (n_samples, n_classes), y_pred is probabilities
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def _gradient(self, params, X, y):
        # params: flattened [W.flatten(), b.flatten()]
        n_samples, n_features = X.shape
        n_classes = y.shape[1]
        W = params[: n_features * n_classes].reshape(n_classes, n_features)
        b = params[n_features * n_classes :].reshape(n_classes)

        # z = X @ W.T + b  # b should be (n_classes,) for broadcasting
        z = X @ W.T + b
        softmax_probs = self._softmax(z)

        # Gradient w.r.t. W: (1/n_samples) * (softmax - y).T @ X + reg_lambda * W
        grad_W = (1 / n_samples) * (softmax_probs - y).T @ X + self.reg_lambda * W
        # Gradient w.r.t. b: (1/n_samples) * sum(softmax - y, axis=0)
        grad_b = (1 / n_samples) * np.sum(softmax_probs - y, axis=0)

        return np.concatenate([grad_W.flatten(), grad_b.flatten()])

    def fit(self, X, y):
        # y should be one-hot encoded, shape (n_samples, n_classes)
        n_samples, n_features = X.shape
        n_classes = y.shape[1]

        # Initialize parameters
        self.W = np.random.randn(n_classes, n_features) * 0.01
        self.b = np.zeros(n_classes)

        initial_params = np.concatenate([self.W.flatten(), self.b.flatten()])

        # Create gradient function with X and y
        grad_func = partial(self._gradient, X=X, y=y)

        # Gradient descent
        gd = GradientDescent(grad_func, self.learning_rate, self.max_iter)
        optimal_params = gd.descent(initial_params)

        # Reshape back
        self.W = optimal_params[: n_features * n_classes].reshape(n_classes, n_features)
        self.b = optimal_params[n_features * n_classes :]

    def predict_proba(self, X):
        z = X @ self.W.T + self.b
        return self._softmax(z)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
