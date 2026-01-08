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

### EXERCICE 1


def heaviside(x):
    return tf.where(x >= 0, 1.0, 0.0)


# Architecture du réseau
modele = Sequential()

# Couches de neurones
modele.add(Dense(2, input_dim=2, activation=heaviside))  # à compléter
modele.add(Dense(1, activation=heaviside))  # à compléter

# Couche 1
coeff = np.array([[-1.0, 2.0], [3.0, 1.0]])  # à compléter
biais = np.array([0.0, 0.0])  # à compléter
poids = [coeff, biais]
modele.layers[0].set_weights(poids)

# Couche 2
coeff = np.array([[1.0], [1.0]])  # à compléter
biais = np.array([-2.0])  # à compléter
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

features_names = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

X = housing_data[:, :-1]  # Toutes les colonnes sauf la dernière
y = housing_data[:, -1]  # Seulement la dernière colonne
N = len(y)

## Q1
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
print(f"\nQ3. Prédiction pour le 1er groupe :")
print(f"    Prix prédit : {pred_g1:.2f}")
print(f"    Prix réel   : {vrai_prix:.2f}")
print(
    "    Commentaire : L'écart (résidu) est normal car le modèle ne passe pas exactement par tous les points."
)

## Q4a
alpha = 0.1  # Valeur classique pour voir l'effet
lasso_reg = Lasso(alpha=alpha)
lasso_reg.fit(X_scaled, y)
coefs_lasso_str = ", ".join(["{:.2f}".format(c) for c in lasso_reg.coef_])
print(f"\nQ4(a). Coefficients Lasso (alpha={alpha}) : [{coefs_lasso_str}]")
print("       -> Notez que plusieurs coefficients sont tombés à 0.00.")

## Q4b
n_lambdas = 100
lambdas = np.logspace(-4, 1, n_lambdas)
coefs_path = []

for lam in lambdas:
    l = Lasso(alpha=lam)
    l.fit(X_scaled, y)
    coefs_path.append(l.coef_)

# Tracé du graphique
plt.figure(figsize=(10, 6))
plt.plot(lambdas, coefs_path)
plt.xscale("log")  # Echelle logarithmique indispensable
plt.xlabel("Paramètre de régularisation (Alpha/Lambda)")
plt.ylabel("Valeur des Coefficients")
plt.title("Chemin de régularisation LASSO")
plt.axis("tight")
plt.grid(True)
plt.legend(
    features_names, loc="upper right", fontsize="small"
)  # Légende pour identifier les courbes
plt.savefig("lasso_path.png")

print("\nQ4(b). Ordre de grandeur pertinent :")
print("       Entre 10^-2 et 10^-1. C'est là que les variables inutiles s'annulent")
print("       mais que la variable principale reste forte.")
