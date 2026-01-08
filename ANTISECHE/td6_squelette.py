# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:21:13 2024

@author: JDION
"""
 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from my_descent import GradientDescent

### Q4 

# Architecture du réseau
modele = Sequential()

# Couches de neurones
modele.add(Dense(2, input_dim=1, activation = "relu")) # 1ère couche à compléter
modele.add(Dense(1, activation = 'linear')) # 2ème couche à compléter

# Couche 0
poids = np.array([[-3, 1]]) # poids de la 1ère couche à compléter
biais = np.array([3, -1]) # biais de la 1ère couche à compléter
param = [poids,biais]
modele.layers[0].set_weights(param)

# Couche 1
poids = np.array([[1], [1]])  # poids de la 2ème couche à compléter
biais = np.array([0])  # biais de la 2ème couche à compléter
param = [poids,biais]
modele.layers[1].set_weights(param)

# Affichage
#X = None # entrée du réseau à compléter
X = np.linspace(0, 2, 1000).reshape(-1, 1)
Y_pred = modele.predict(X)
plt.plot(X, Y_pred, color='red', label='Sortie du réseau')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()

"""
### Q5

# Lecture des données

def read_data():
    x = np.genfromtxt('x_train.csv', delimiter=',')
    y = np.genfromtxt('y_train.csv', delimiter=',')
    return x, y

x_train, y_train = read_data()

## Entraînement du réseau

# Architecture du réseau
modele2 = Sequential()

# Couches de neurones
# à compléter

# Entraînement du réseau par descente de gradient
modele2.compile(None) # à compléter
history = modele2.fit(None) # à compléter

# Affichage
X = None # entrée du réseau à compléter
Y_pred = modele2.predict(X)
plt.plot(X, modele.predict(X), color='red', label='Sortie du réseau parfait')
plt.plot(X, Y_pred, color='blue', label='Sortie du réseau entraîné')
plt.scatter(x_train, y_train, label = "Données d'entraînement")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()

# Affichage de l'erreur au fil des époques
plt.plot(history.history['loss'])
plt.xlabel('époques')
plt.ylabel('erreur')
plt.legend()
plt.grid()
plt.show()


### Q7c
"""