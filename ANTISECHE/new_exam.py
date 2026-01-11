# -*- coding: utf-8 -*-
"""
@author: JDION
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import Activation
from keras.utils import get_custom_objects
from keras.models import Sequential
from keras.layers import Dense

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


### EXERCICE 1
# question 1 sur feuille
# question 2
def heaviside(x):
    return tf.where(x >= 0, 1.0, 0.0)


# Fonction d'activation personnalisée
get_custom_objects().update({"heaviside": Activation(heaviside)})

# Architecture du réseau
modele = Sequential()

# Couches de neurones
modele.add(Dense(2, input_dim=2, activation="heaviside"))  # à compléter
modele.add(Dense(1, activation="heaviside"))  # à compléter


# Couche 1
coeff = np.array([[-1, 3], [2, 1]])  # à compléter
biais = np.array([0, 0])  # à compléter
poids = [coeff, biais]
modele.layers[0].set_weights(poids)

# Couche 2
coeff = np.array([[1], [1]])  # à compléter
biais = np.array([-2])  # à compléter
poids = [coeff, biais]
modele.layers[1].set_weights(poids)

# Affichage
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x_grid, y_grid = np.meshgrid(x, y)
xy_grid = np.c_[x_grid.ravel(), y_grid.ravel()]

# Prédiction des valeurs sur la grille
z_grid = modele.predict(xy_grid).reshape(x_grid.shape)

# Visualisation des résultats en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x_grid, y_grid, z_grid, cmap="viridis")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
ax.set_title("Sortie du réseau sur [-5, 5]^2")
plt.show()

### EXERCICE 3


def read_data():
    housing_data = np.genfromtxt("housing.csv", delimiter=",")
    return housing_data


housing_data = read_data()
print(housing_data)

X = housing_data[:, :-1]  # All columns except the last
y = housing_data[:, -1]  # Last column is the target (Price)

# 1. Standardization
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# median
med_inc_first_group = X_standardized[0, 0]
print(f"Median : {med_inc_first_group}")

# 2. Regression lineaire sans regularization
X_train, X_test, y_train, y_test = train_test_split(
    X_standardized, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

print("Linear Regression Coefficients:", linear_model.coef_)
print("Linear Regression Intercept:", linear_model.intercept_)


# 3. Predire le prix avec variables explicatives
y_pred = linear_model.predict(X_test)

first_prediction = y_pred[0]
print(f"Predicted Price for the first test group: {first_prediction}")

# 4 Regularization with Lasso
# (a) Lasso Regression for a specific regularization parameter
lasso_model = Lasso(alpha=0.1, max_iter=10000)
lasso_model.fit(X_train, y_train)

# Print coefficients for Lasso
print("Lasso Regression Coefficients:", lasso_model.coef_)
print("Lasso Regression Intercept:", lasso_model.intercept_)

# (b) Regularization path
alphas = np.logspace(-4, 0, 100)  # Test a range of alpha values
coefficients = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    coefficients.append(lasso.coef_)

# Plot the regularization path
plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    plt.plot(alphas, [coef[i] for coef in coefficients], label=f"Feature {i}")

plt.xscale("log")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("Coefficient Value")
plt.title("Regularization Path (Lasso)")
plt.legend()
plt.show()

# (c) Identify the feature with the most significant effect
# The feature with the largest absolute coefficient (non-zero) has the most impact
most_important_feature_index = np.argmax(np.abs(lasso_model.coef_))
print(
    f"The feature with the most significant effect: Feature {most_important_feature_index}"
)


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 4. Regularization with Lasso
# (a) Lasso Regression for a specific regularization parameter
lasso_model = Lasso(alpha=0.1, max_iter=10000)
lasso_model.fit(X_train, y_train)

# Print coefficients for Lasso
print("Lasso Regression Coefficients:", lasso_model.coef_)
print("Lasso Regression Intercept:", lasso_model.intercept_)

# (b) Regularization path
alphas = np.logspace(-4, 0, 100)  # Test a range of alpha values
coefficients = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    coefficients.append(lasso.coef_)

# Plot the regularization path
plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    plt.plot(alphas, [coef[i] for coef in coefficients], label=f"Feature {i}")

plt.xscale("log")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("Coefficient Value")
plt.title("Regularization Path (Lasso)")
plt.legend()
plt.show()

# (c) Identify the feature with the most significant effect
# The feature with the largest absolute coefficient (non-zero) has the most impact
most_important_feature_index = np.argmax(np.abs(lasso_model.coef_))
print(
    f"The feature with the most significant effect: Feature {most_important_feature_index}"
)
