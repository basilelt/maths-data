# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:22:15 2025
@author: BLETHIEC
"""

import numpy as np
from descente_stochastique import GradientDescent
import pandas as pd
import random as rd
import matplotlib.pyplot as plt


def F(a, b, x):
    return 1 / (1 + np.exp(-(a * x + b)))


def grad_ce(data, a):
    grad = np.array([0.0, 0.0])
    for x, y in data:
        p = F(a[0], a[1], x)
        grad[0] += x * (p - y)  # dérivée par rapport à a
        grad[1] += p - y  # dérivée par rapport à b
    return grad


df = pd.read_csv("titanic2.csv", sep=";")
print(df.head())

# Visualisation des données
plt.scatter(df["Age"], df["Survived"])
plt.xlabel("Age")
plt.ylabel("Survived")
plt.title("Titanic Survival by Age")
plt.show()

initial_point = [0, 0]
initial_point[0], initial_point[1] = rd.randint(1, 90), rd.randint(0, 1)

gd = GradientDescent(
    gradient=grad_ce,
    learning_rate=0.01,
    max_iterations=1000,
    epsilon=1e-6,
    batch_size=1,
)

gd.descent(initial_point=initial_point, data=df.to_numpy())
