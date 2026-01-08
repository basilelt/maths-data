# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:47:15 2024

@author: JDION
"""

import numpy as np

from my_descent_td1 import GradientDescent


### EXERCICE 5
def f(x):
    return x.pow(2) +1
def gradf(x):
    return 2*x
delta = 0.2
k=10
gd= GradientDescent(gradf,learning_rate=delta,max_iterations=k)
#Q2a
a=2
result = gd.descent(a)
print("descante de gradient partant de {} en {} iterations avec un pas de {},\n  \
resultat a={},f(a)={}".format(a,k,delta,result,f(result)))

#Q2b
a=-1.5
result = gd.descent(a)
print("descante de gradient partant de {} en {} iterations avec un pas de {},\n  \
resultat a={},f(a)={}".format(a,k,delta,result,f(result)))

#Q2c
deltas=[0.9,1.1,0.05]
for d in deltas:
    gd=GradientDescent(gradf,learning_rate=d,max_iterations=k)
    a=2
    result=gd.descent(a)
    print("descante de gradient partant de {} en {} iterations avec un pas de {},\n  \
resultat a={},f(a)={}".format(a,k,d,result,f(result)))


### EXERCICE 6
print("------EXERCICE6-------")
def f(x):
    return x**4 - 2*x**3 +4
def gradf(x):
    return 4*x**3 -6*x**2
delta = 0.05
k=1000
gd= GradientDescent(gradf,learning_rate=delta,max_iterations=k)
#Q2a
a=-1
result = gd.descent(a)
print("descante de gradient partant de {} en {} iterations avec un pas de {},\n  \
resultat a={},f(a)={}".format(a,k,delta,result,f(result)))


### EXERCICE 7 


### EXERCICE 8

data = np.array(((4, 1), (7, 3), (8, 3), (10, 6), (12, 7)))


### EXERCICE 9

data = np.array(((1, 0, 0), (0, 1, 5), (2, 1, 1), (1, 2, 0), (2, 2, 3)))




