import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data 
y = iris.target 

#Séparation Entraînement/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print(f"Taille de l'échantillon d'apprentissage: {X_train.shape[0]} ")
print(f"Taille de l'échantillon de test: {X_test.shape[0]} ")
print(f"Données d'entrée: {X_train.shape[1]} caractéristiques")
print(" Petal length, Petal width")

