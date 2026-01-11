# Projet Régression Logistique pour la Classification de Chiffres Manuscrits

Ce projet implémente une régression logistique multi-classes from scratch pour classifier les chiffres manuscrits du dataset digits de scikit-learn.

## Structure du Projet

- `src/` : Code source Python
  - `main.py` : Script principal contenant l'implémentation from scratch et les comparaisons
  - `my_descent.py` : Classe GradientDescent pour l'optimisation

- `docs/` : Documentation
  - `grille_evaluation_2A_projet_mds.csv` : Grille d'évaluation du projet
  - `sujet-projet.pdf` : Sujet du projet

- `reports/` : Rapport LaTeX
  - `rapport.tex` : Source du rapport
  - `rapport.pdf` : Rapport compilé

- `results/` : Résultats générés
  - `confusion_matrix_custom.png` : Matrice de confusion du modèle custom
  - `confusion_matrix_sklearn.png` : Matrice de confusion de scikit-learn
  - `coefficients.png` : Visualisation des coefficients appris
  - `misclassified.png` : Exemples d'images mal classifiées

- `requirements.txt` : Dépendances Python
- `.python-version` : Version Python (3.13)

## Installation

1. Créer un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

Exécuter le script principal depuis la racine du projet :
```bash
python src/main.py
```

Le script effectuera :
- Chargement et prétraitement des données
- Réglage des hyperparamètres par validation croisée
- Entraînement du modèle custom et comparaison avec scikit-learn
- Génération des visualisations dans `results/`
- Test bonus sur MNIST

## Rapport

Le rapport détaillé se trouve dans `reports/rapport.pdf`. Pour le recompiler :
```bash
cd reports
pdflatex rapport.tex
```

## Auteurs

- Basile LE THIEC
- Lilian NOACCO

2A Alt IR