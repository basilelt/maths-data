

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def read_data_numpy(filename):
    """ Pour fichiers sans header (ex: housing.csv - Exam Ex3) """
    return np.genfromtxt(filename, delimiter=",")

def read_data_pandas(filename):
    """ Pour fichiers avec header (ex: titanic.csv - TD5) """
    # Comme dans td5_ex1.py
    df = pd.read_csv(filename, sep=None, engine='python')
    df.columns = df.columns.str.strip().str.lower()
    return df

# À utiliser si : "Prédire une valeur continue", "Lasso", "Ridge", "Polynomial"

print("\n--- BLOC 1 : REGRESSION ---")

# 1. Chargement & Préparation 
housing = read_data_numpy('housing.csv') 
X = housing[:, :-1]
y = housing[:, -1]
N = len(y)

# (Simulation données pour que le code tourne)
#X = np.random.rand(50, 8); y = 3*X[:,0] + np.random.normal(0, 0.1, 50); N=50

# 2. Standardisation (OBLIGATOIRE avant Lasso/Ridge)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. OLS (Linear Regression)
reg = LinearRegression().fit(X_scaled, y)
# Affichage formaté (Style TD4)
coefs_fmt = ','.join(['{:.2f}'.format(c) for c in reg.coef_])
print(f'Ajustement OLS : intercept = {reg.intercept_:.2f}, coef = [{coefs_fmt}]')

# 4. LASSO (Sélection de variables)
# a) Solution ponctuelle
lasso = Lasso(alpha=0.1).fit(X_scaled, y)
coefs_l = ','.join(['{:.2f}'.format(c) for c in lasso.coef_])
print(f'Lasso (alpha=0.1) : coef = [{coefs_l}]')

# b) Chemin de régularisation (Boucle TD4)
lambdas = np.logspace(-4, 1, 100)
coefs_path = []

for lam in lambdas:
    l = Lasso(alpha=lam).fit(X_scaled, y)
    coefs_path.append(l.coef_)

plt.figure(figsize=(8,4))
plt.plot(lambdas, coefs_path)
plt.xscale('log')
plt.title('Chemin Lasso')
plt.xlabel('lambda')
plt.ylabel('coefficients')
plt.grid(True)
plt.show() 

# À utiliser si : "Prédire Survie/Classe 0 ou 1", "Titanic", "Iris"

print("\n--- BLOC 2 : LOGISTIQUE ---")

# 1. Chargement (Style TD5_ex1)
df = read_data_pandas('titanic2.csv')
df = df.dropna(subset=['age', 'survived'])
X = df[['age']].values # ou plusieurs colonnes
y = df['survived'].values

# (Simulation)
#X_log = np.random.rand(100, 2); y_log = (X_log[:,0] + X_log[:,1] > 1).astype(int)

# 2. Split Train/Test (Style TD5_ex2)
X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

# 3. Normalisation
scaler_log = StandardScaler()
X_train_s = scaler_log.fit_transform(X_train)
X_test_s = scaler_log.transform(X_test)

# 4. Modèle
clf = LogisticRegression()
clf.fit(X_train_s, y_train)

# 5. Scores
y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy LogReg: {acc:.2f}")
# print(f"Coefs: {clf.coef_}")


# À utiliser si : "Réseau logique", "Keras", "ReLU", "Heaviside"

print("\n--- BLOC 3 : NEURAL NET ---")

def heaviside(x):
    return tf.where(x >= 0, 1.0, 0.0)

# 1. Définition Manuelle (Exam Ex1)
model = Sequential()
# Input_dim = nb entrées. Activation 'relu', 'linear' ou heaviside
model.add(Dense(2, input_dim=2, activation=heaviside)) 
model.add(Dense(1, activation=heaviside)) 

# 2. Forcer les poids (TD6 / Exam)
# Couche 1 : [[wx1, wx2], [wy1, wy2]]
W1 = np.array([[-1.0, 2.0], [ 3.0, 1.0]]) 
b1 = np.array([0.0, 0.0])
model.layers[0].set_weights([W1, b1])

# Couche 2
W2 = np.array([[1.0], [1.0]])
b2 = np.array([-2.0])
model.layers[1].set_weights([W2, b2])

# 3. Test / Affichage courbe (Style TD6)
X_grid = np.array([[0,0], [1,1], [10,10]]) # Points test
Y_pred = model.predict(X_grid, verbose=0)
print(f"Preds NN: \n{Y_pred}")


# À utiliser si : "Coder la descente", "Trouver le minimum de f(x)"

print("\n--- BLOC 4 : GRADIENT DESCENT ---")

class GradientDescent:
    # Copie exacte de my_descent.py
    def __init__(self, gradient, learning_rate, max_iterations):
        self.gradient = gradient
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def descent(self, initial_point):
        # Version simplifiée TD1
        point = initial_point
        for i in range(self.max_iterations):
            grad = self.gradient(point)
            point = point - self.learning_rate * grad
        return point

# Exemple TD1 (f(x) = x^2 + 1)
def gradf(x): return 2*x

gd = GradientDescent(gradf, learning_rate=0.2, max_iterations=10)
res = gd.descent(initial_point=2)
print(f"Minimum trouvé (partant de 2) : {res:.4f}")
