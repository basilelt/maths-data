# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:37:36 2022

@author: JDION
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing

from my_descent import GradientDescent

### EXERCICE 1 : REGRESSION QUADRATIQUE

def read_data(filename="data_quad.csv"):
    data = np.genfromtxt(filename, delimiter=',')
    x, y = data[0], data[1]
    return x, y

x, y = read_data()
plt.plot(x, y,'o', label = 'Données')
plt.legend()
plt.show()

# Q1

d = 2
poly = PolynomialFeatures(degree=d, include_bias=False) 
X = poly.fit_transform(x.reshape(-1, 1))
N = len(x)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Q2

y2 = y.copy()
for i in range(int(N / 5)):
    y2[i * 5] = 0


# Q3 

# résolution avec descente de gradient

def E(a):
    E = 0
    for i in range(len(x)):
        E += abs(y2[i] - (a[0] + a[1] * X_scaled[i][0] + a[2] * X_scaled[i][1]))
    return E

def gradE(a, sample):
    gradE = np.array((0, 0, 0), dtype = float)
    for i in range(len(sample)):
        gradE += -np.array((1, sample[i][0][0], sample[i][0][1])) * (sample[i][1] - (a[0] + a[1] * sample[i][0][0] + a[2] * sample[i][0][1])) / abs(sample[i][1] - (a[0] + a[1] * sample[i][0][0] + a[2] * sample[i][0][1]))
    return gradE

delta = 0.001

gd = GradientDescent(gradE, learning_rate=delta, max_iterations=10000, batch_size=len(x))

a = np.array([0, 0, -1])
result = gd.descent(a, data = np.array(list(zip(X_scaled, y2)), dtype=object))
num_epochs = gd.num_iterations

coefficients = ', '.join(['{:.2f}'.format(c) for c in result])
print("Descente de gradient partant de a = {} avec un pas de {}, \
      résultat a = [{}], E(a) = {:.2f}, {} époques"\
          .format(a, delta, coefficients, E(result), num_epochs))
    
def plot_regression(result, x, y, poly, scaler):
    t = np.linspace(0, 1, 200).reshape(-1, 1)
    Phi = poly.transform(t)              
    Phi_scaled = scaler.transform(Phi)
    y_hat = result[0] + Phi_scaled @ result[1:]
        
    plt.figure()
    plt.plot(x, y, 'o', label='Données')
    plt.plot(t, y_hat, 'r', linewidth=2, label='Ajustement degré 2')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ajustement polynomial')
    plt.show()
    
plot_regression(result, x, y2, poly, scaler)
