# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:47:15 2024
@author: JDION, BLETHIEC
"""

import numpy as np
from my_descent_td1 import GradientDescent


### EXERCICE 5
# Q2a
def f5(x):
    return x**2 + 1


def gradf5(x):
    return 2 * x


delta = 0.2
k = 10
ipoint = 2
gd = GradientDescent(gradf5, learning_rate=delta, max_iterations=k)
epoint = gd.descent(initial_point=(ipoint))
print("Exercice 5.2.a:")
print(f"Point optimal pour f(x) = x^2 + 1:\n{epoint}\n")

# Q2b
delta = 0.01
k = 1000
ipoint = -1.5
gd = GradientDescent(gradf5, learning_rate=delta, max_iterations=k)
epoint = gd.descent(initial_point=(ipoint))
print("Exercice 5.2.b:")
print(f"Point optimal pour f(x) = x^2 + 1:\n{epoint}\n")

# Q2c
print(f"Exercice 5.2.c:")
k = 1000
for delta in [0.9, 1.1, 0.05]:
    gd = GradientDescent(gradf5, learning_rate=delta, max_iterations=k)
    epoint = gd.descent(initial_point=(ipoint))
    print(f"Point optimal pour f(x) = x^2 + 1 avec delta = {delta}:\n{epoint}")


### EXERCICE 6
def f6(x):
    return x**4 + -2 * (x**3) + 4


def gradf6(x):
    return 4 * (x**3) - 6 * (x**2)


print("\n\nExercice 6:")
delta = 0.05
k = 1000
ipoint = -1
gd = GradientDescent(gradf6, learning_rate=delta, max_iterations=k)
epoint = gd.descent(initial_point=(ipoint))
print(f"Point optimal pour f(x) = x^4 - 2x^3 + 4:\n{epoint}\n")
print(f"Son image par f est :\n {f6(epoint)}\n")


### EXERCICE 7
# Q1
print("Exercice 7.1:")



# Q2a

# Q2b

# Q2c


### EXERCICE 8
data = np.array(((4, 1), (7, 3), (8, 3), (10, 6), (12, 7)))
# TODO


### EXERCICE 9
data = np.array(((1, 0, 0), (0, 1, 5), (2, 1, 1), (1, 2, 0), (2, 2, 3)))
# TODO
