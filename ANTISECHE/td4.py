import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.special import eval_legendre
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


#Visualisation des données 
def read_data(filename="data.csv"):
    data = np.genfromtxt(filename,delimiter=",")
    x,y = data[0],data[1]
    return x,y
x,y= read_data()
plt.plot(x,y,'o',label='Données')
plt.legend()
plt.show()

#Q2 & Q4
d=50
poly=PolynomialFeatures(degree=d, include_bias=False)
X=poly.fit_transform(x.reshape(-1,1))
N=len(X)

scaler= StandardScaler()
X_scaled = scaler.fit_transform(X)

reg=LinearRegression().fit(X_scaled,y)
error = mean_squared_error(y,reg.predict(X_scaled))*N

coefficients = ','.join (['{:.2f}'.format(c) for c in reg.coef_])

def plot_regression(x,y,reg,poly,scaler,d):
    # Changement 1 : adaptation à la plage réelle des x
    t=np.linspace(-1, 1, 1000).reshape(-1,1)
    
    Phi = poly.transform(t)
    Phi_scaled= scaler.transform(Phi)
    y_hat= reg.predict( Phi_scaled)

    plt.figure()
    plt.scatter(x,y,label="Données")
    plt.plot(t,y_hat,'r',linewidth=2,label=f"Ajustemnet degré{d}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ajustement polynomiale')
    plt.ylim(-2,2)
    plt.legend()
    plt.show()

plot_regression(x,y,reg,poly,scaler,d)

#Ex3

reg = Ridge(alpha = 0.01).fit(X_scaled,y)
error = mean_squared_error(y, reg.predict(X_scaled))*N

coefficients = ','.join (['{:.2f}'.format(c) for c in reg.coef_])
print(f'Ajustement RIDGE : intercept = {reg.intercept_:.4f}, coef = [{coefficients}], erreur = {error:.2f}')
plot_regression(x,y,reg,poly,scaler,d)

n_lambdas = 200
lambdas = np.logspace(-5,5,n_lambdas)
coefs = []

for lam in lambdas :
    reg = Ridge(alpha = lam, max_iter = int(1e7)).fit(X_scaled,y)
    result = np.concatenate(([reg.intercept_], reg.coef_))
    coefs.append(result)

plt.figure()
plt.plot(lambdas, coefs)
plt.xscale('log')
plt.xlabel('lambda')
plt.ylabel('coefficients')
plt.title('chemin de régulation Ridge ')
plt.show()


from sklearn.linear_model import Lasso
reg=Lasso(alpha=0.001).fit(X_scaled,y)
error = mean_squared_error(y,reg.predict(X_scaled))*N
coefficients = ','.join (['{:.2f}'.format(c) for c in reg.coef_])
print(f'Ajustement LASSO : intercept = {reg.intercept_:.4f}, coef = [{coefficients}], erreur = {error:.2f}')
plot_regression(x,y,reg,poly,scaler,d)

n_lambdas = 200
lambdas = np.logspace(-5,5,n_lambdas)
coefs = []

for lam in lambdas :
    reg = Lasso(alpha = lam, max_iter = int(1e7)).fit(X_scaled,y)
    result = np.concatenate(([reg.intercept_], reg.coef_))
    coefs.append(result)

plt.figure()
plt.plot(lambdas, coefs)
plt.xscale('log')
plt.xlabel('lambda')
plt.ylabel('coefficients')
plt.title('chemin de régulation Lasso')
plt.show()

#Ex4

housing=fetch_california_housing()
x,y= housing.data,housing.target
scaler= StandardScaler()
X_scaled = scaler.fit_transform(x)
n_lambdas = 100
lambdas = np.logspace(-5,5,n_lambdas)
coefs = []

for lam in lambdas :
    reg = Lasso(alpha = lam, max_iter = int(1e7)).fit(X_scaled,y)
    result = np.concatenate(([reg.intercept_], reg.coef_))
    coefs.append(result)

plt.figure()
labels=['biais','MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
plt.plot(lambdas, coefs,label=labels)
plt.xscale('log')
plt.xlabel('lambda')
plt.ylabel('coefficients')
plt.title('chemin de régulation Lasso')
plt.show()