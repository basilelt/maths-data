import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
import numpy as np

#Chargement et préparation des données
titanic = pd.read_csv('titanic2.csv', sep=None, engine='python')
titanic.columns = titanic.columns.str.strip().str.lower()
data = titanic.dropna(subset=['age', 'survived']).copy()

#Normalisation del'age uniquement
scaler = StandardScaler()
data['age_normalized'] = scaler.fit_transform(data[['age']])

#Création du modèle de Régression Linéaire
model = LinearRegression()
X = data[['age_normalized']] 
y = data['survived']        
model.fit(X, y)           

#Calcul de la droite de régression
#On crée des points entre le min et le max de l'age normalisé pour tracer la droite
x_range = np.linspace(data['age_normalized'].min(), data['age_normalized'].max(), 100).reshape(-1, 1)
y_pred = model.predict(x_range)

#Visualisation
plt.figure(figsize=(10, 6))

#Points réels
plt.scatter(data['age_normalized'], data['survived'], alpha=0.2, color='blue', label='Données réelles')

plt.plot(x_range, y_pred, color='red', linewidth=3, label='Droite de régression linéaire')
plt.title("Survie en fonction de l'age")
plt.xlabel("Âge normalisé")
plt.ylabel("Prédiction de survie")
plt.grid(True)
plt.legend()
plt.show()

