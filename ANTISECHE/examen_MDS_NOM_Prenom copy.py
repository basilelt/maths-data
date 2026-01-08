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
# ATTENTION: La fonction est passée directement, SANS guillemets
modele.add(Dense(2, input_dim=2, activation = heaviside))
modele.add(Dense(1, activation = heaviside)) 

# Couche 1
coeff = np.array([[-1.0, 2.0], [ 3.0, 1.0]]) 
biais = np.array([0.0, 0.0])    
poids = [coeff,biais]
modele.layers[0].set_weights(poids)

# Couche 2
coeff = np.array([[1.0], [1.0]]) 
biais = np.array([-2.0]) 
poids = [coeff,biais]
modele.layers[1].set_weights(poids)

# Affichage (le reste est inchangé)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x_grid, y_grid = np.meshgrid(x, y)
xy_grid = np.c_[x_grid.ravel(), y_grid.ravel()]

# Prédiction des valeurs sur la grille
z_grid = modele.predict(xy_grid).reshape(x_grid.shape)

# Visualisation des résultats en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Sortie du réseau sur [-5, 5]^2')
plt.show()

#EX3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- FONCTION UTILITAIRE (Style TD) ---
def read_data(filename="housing.csv"):
    """
    Lecture des données avec numpy, comme vu en cours.
    Le fichier housing.csv contient 8 variables explicatives et 1 cible (Price).
    """
    # housing.csv n'a pas d'en-tête, on charge tout directement
    data = np.genfromtxt(filename, delimiter=",")
    return data

# CHARGEMENT ET PRÉPARATION
print("--- CHARGEMENT DES DONNÉES ---")
housing_data = read_data()

# Séparation des variables (Slicing Numpy)
# X : Toutes les colonnes sauf la dernière (Variables explicatives)
# y : La dernière colonne (Variable cible : Price)
X = housing_data[:, :-1]
y = housing_data[:, -1]
N = len(y)

# Noms des variables (pour l'interprétation finale)
features_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                  "Population", "AveOccup", "Latitude", "Longitude"]

# QUESTION 1 : STANDARDISATION
# Pourquoi ? Pour que toutes les variables soient à la même échelle (moyenne 0, écart-type 1).
# C'est indispensable avant une régularisation (Lasso/Ridge) pour ne pas fausser les pénalités.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# "Donner la valeur du revenu médian (MedInc) du premier groupe"
# MedInc est la colonne 0. Premier groupe est la ligne 0.
valeur_q1 = X_scaled[0, 0]
print(f"\nQ1. Revenu médian standardisé (1er groupe) : {valeur_q1:.2f}")



# QUESTION 2 : RÉGRESSION LINÉAIRE (OLS)
# On utilise LinearRegression sans paramètres pour faire des Moindres Carrés Ordinaires.

reg = LinearRegression()
reg.fit(X_scaled, y)

# Affichage formaté comme dans le TD4 pour plus de lisibilité
coefs_str = ', '.join(['{:.2f}'.format(c) for c in reg.coef_])
print(f"\nQ2. Solution OLS :")
print(f"    Intercept : {reg.intercept_:.2f}")
print(f"    Coefficients : [{coefs_str}]")


# QUESTION 3 : PRÉDICTION
# On teste le modèle sur le premier individu (groupe 1).

# Attention : reshape(1, -1) est obligatoire pour prédire sur une seule ligne avec sklearn
x_g1 = X_scaled[0].reshape(1, -1)
pred_g1 = reg.predict(x_g1)[0]
vrai_prix = y[0]

print(f"\nQ3. Prédiction pour le 1er groupe :")
print(f"    Prix prédit : {pred_g1:.2f}")
print(f"    Prix réel   : {vrai_prix:.2f}")
print("    Commentaire : L'écart (résidu) est normal car le modèle ne passe pas exactement par tous les points.")


# QUESTION 4 : RÉGULARISATION LASSO
# Le Lasso ajoute une pénalité L1 qui force certains coefficients à devenir NULS.
# C'est utile pour la SÉLECTION DE VARIABLES.

# (a) Solution pour un alpha donné
alpha_choisi = 0.1 # Valeur classique pour voir l'effet
lasso_reg = Lasso(alpha=alpha_choisi)
lasso_reg.fit(X_scaled, y)

coefs_lasso_str = ', '.join(['{:.2f}'.format(c) for c in lasso_reg.coef_])
print(f"\nQ4(a). Coefficients Lasso (alpha={alpha_choisi}) : [{coefs_lasso_str}]")
print("       -> Notez que plusieurs coefficients sont tombés à 0.00.")

# (b) Chemin de régularisation (Boucle manuelle comme dans TD4)
# On fait varier alpha (lambda) de très petit à grand pour voir l'évolution des poids.
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
plt.xscale('log') # Echelle logarithmique indispensable
plt.xlabel('Paramètre de régularisation (Alpha/Lambda)')
plt.ylabel('Valeur des Coefficients')
plt.title('Chemin de régularisation LASSO')
plt.axis('tight')
plt.grid(True)
plt.legend(features_names, loc='upper right', fontsize='small') # Légende pour identifier les courbes
plt.show()

print("\nQ4(b). Ordre de grandeur pertinent :")
print("       Entre 10^-2 et 10^-1. C'est là que les variables inutiles s'annulent")
print("       mais que la variable principale reste forte.")

# (c) Variable la plus influente
# C'est celle qui a le coefficient le plus élevé en valeur absolue, 
# ou celle qui s'annule en dernier sur le graphe Lasso.
idx_max = np.argmax(np.abs(reg.coef_))
nom_variable_max = features_names[idx_max]
coef_max = reg.coef_[idx_max]

print(f"\nQ4(c). Variable explicative ayant le plus d'effet : {nom_variable_max}")
print(f"       Coefficient associé : {coef_max:.2f}")
print("       Justification : C'est elle qui a le plus fort poids dans la décision du prix.")





# BONUS : RÉGULARISATION RIDGE (L2)
from sklearn.linear_model import Ridge

print("\n--- BONUS : RIDGE REGRESSION ---")

# 1. Ajustement simple avec Ridge
# Ridge réduit les coefficients sans les annuler (contrairement au Lasso)
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_scaled, y)

coefs_ridge_str = ', '.join(['{:.2f}'.format(c) for c in ridge_reg.coef_])
print(f"Ridge (alpha=1.0) Coeffs : [{coefs_ridge_str}]")

# 2. Chemin de régularisation Ridge
# On prend une plage plus large (jusqu'à 10^5) car Ridge décroît plus lentement
lambdas_ridge = np.logspace(-4, 5, 100) 
coefs_ridge_path = []

for lam in lambdas_ridge:
    r = Ridge(alpha=lam)
    r.fit(X_scaled, y)
    coefs_ridge_path.append(r.coef_)

# Tracé du graphique Ridge
plt.figure(figsize=(10, 6))
plt.plot(lambdas_ridge, coefs_ridge_path)
plt.xscale('log')
plt.title('Chemin de régularisation RIDGE (L2)')
plt.xlabel('Alpha (Lambda)')
plt.ylabel('Coefficients')
# On remet la légende pour comparer les couleurs avec le Lasso
plt.legend(features_names, loc='upper right', fontsize='small')
plt.grid(True)
plt.show()

print("Commentaire Ridge : Les courbes convergent vers 0 ensemble et plus doucement.")
print("Il n'y a pas de sélection brutale de variables comme avec le Lasso.")