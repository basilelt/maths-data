# -*- coding: utf-8 -*-
"""
@author: JDION
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

### EXERCICE 1


def heaviside(x):
    return tf.where(x >= 0, 1.0, 0.0)


# Fonction d'activation personnalisée
get_custom_objects().update({"heaviside": Activation(heaviside)})

# Architecture du réseau
modele = Sequential()

# Couches de neurones
modele.add(Dense(None, input_dim=None, activation="heaviside"))  # à compléter
modele.add(Dense(None, activation="heaviside"))  # à compléter

# Couche 1
coeff = np.array(None)  # à compléter
biais = np.array(None)  # à compléter
poids = [coeff, biais]
modele.layers[0].set_weights(poids)

# Couche 2
coeff = np.array(None)  # à compléter
biais = np.array(None)  # à compléter
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

## Q1
X = housing_data[:, :-1]  # Toutes les colonnes sauf la dernière
y = housing_data[:, -1]  # Seulement la dernière colonne
scale = StandardScaler()
X_scaled = scale.fit_transform(X)
valeur_q1 = X_scaled[0, 0]
print("\nQ1. Revenue médian standardisé (1er groupe) :", valeur_q1)


## Q2
reg = LinearRegression()
reg.fit(X_scaled, y)

coefs_str = ", ".join(["{:.2f}".format(c) for c in reg.coef_])
print(f"\nQ2. Solution OLS :")
print(f"    Intercept : {reg.intercept_:.2f}")
print(f"    Coefficients : [{coefs_str}]")

## Q3
x_g1 = X_scaled[0].reshape(1, -1)
pred_g1 = reg.predict(x_g1)[0]
vrai_prix = y[0]
