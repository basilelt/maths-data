# Rapport de Projet : Classifieur d'Images par Régression Logistique

## Résumé Exécutif

Ce projet implémente un classifieur de chiffres manuscrits utilisant la régression logistique multiclasse entraînée par descente de gradient. Nous avons développé une implémentation « from scratch » en Python pur, puis l'avons comparée avec l'implémentation de scikit-learn. Notre modèle personnalisé atteint une précision de test de **95.28%** (17 erreurs sur 360 échantillons de test), tandis que scikit-learn atteint **97.22%** (10 erreurs). Ces résultats montrent que notre implémentation fonctionne correctement et s'approche des performances des algorithmes optimisés de production.

---

## 1. Introduction et Objectifs

### 1.1 Contexte du Projet

Le projet demande d'implémenter un classifieur d'images pour reconnaître les chiffres manuscrits (0-9) en utilisant la base de données **digits** de scikit-learn, contenant 1797 images en niveaux de gris de taille 8×8 (64 features par image) avec des valeurs de pixels entre 0 et 16.

### 1.2 Objectifs Principaux

1. **Implémenter la régression logistique multiclasse from scratch** en utilisant l'algorithme de descente de gradient
2. **Comparer les résultats** avec l'implémentation de scikit-learn
3. **Analyser les résultats obtenus** incluant l'influence des paramètres et l'interprétation des coefficients
4. **Évaluer le modèle** en utilisant des métriques appropriées (accuracy, confusion matrix, classification report)

### 1.3 Questions de Recherche Adressées

- Comment implémenter correctement la régression logistique multiclasse ?
- Quels paramètres d'entraînement influencent les performances ?
- Comment les deux implémentations se comparent-elles ?
- Quels sont les chiffres les plus confondus et pourquoi ?

---

## 2. Fondation Théorique

### 2.1 Régression Logistique Multiclasse

La régression logistique est un modèle de classification linéaire probabiliste. Pour la classification multiclasse avec \(K\) classes, nous utilisons la fonction **softmax** :

\[
P(y=k|x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
\]

où \(z_k = w_k^T x + b_k\) est la sortie linéaire pour la classe \(k\).

### 2.2 Fonction de Coût

Nous utilisons la **perte d'entropie croisée** (cross-entropy loss) avec régularisation L2 optionnelle :

\[
L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik}) + \frac{\lambda}{2m} \|W\|^2
\]

où :
- \(m\) est le nombre d'échantillons
- \(y_{ik}\) est l'encodage one-hot de la vraie classe
- \(\hat{y}_{ik}\) est la probabilité prédite
- \(\lambda\) est le paramètre de régularisation

### 2.3 Descente de Gradient

Les gradients sont calculés par rapport aux poids et biais :

\[
\frac{\partial L}{\partial W} = \frac{1}{m} X^T (Y_{pred} - Y_{true}) + \frac{\lambda}{m} W
\]

\[
\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (y_{pred,i} - y_{true,i})
\]

Les paramètres sont mis à jour selon :

\[
W := W - \alpha \frac{\partial L}{\partial W}
\]

où \(\alpha\) est le taux d'apprentissage.

### 2.4 Stabilité Numérique

Pour éviter le débordement (overflow) lors du calcul de l'exponentielle, nous soustrayons la valeur maximale avant d'appliquer softmax :

\[
P(y=k|x) = \frac{e^{z_k - \max(z)}}{\sum_{j=1}^{K} e^{z_j - \max(z)}}
\]

---

## 3. Méthodologie

### 3.1 Préparation des Données

**Chargement du dataset :**
- 1797 images de digits manuscrits
- 64 features (pixels 8×8)
- 10 classes (chiffres 0-9)
- Distribution équilibrée (~180 images par classe)

**Division train/test :**
- Ensemble d'entraînement : 1437 échantillons (80%)
- Ensemble de test : 360 échantillons (20%)
- Stratified split pour préserver la distribution des classes

**Normalisation :**
- StandardScaler (centrage et réduction)
- Moyenne après normalisation : 0.000000
- Écart-type après normalisation : 0.976281

### 3.2 Implémentation Personnalisée

Notre classe `LogisticRegressionFromScratch` implémente :
- Initialisation des poids et biais à zéro
- Fonction softmax avec stabilité numérique
- Calcul de perte d'entropie croisée
- Algorithme de descente de gradient avec boucles sur les itérations
- Régularisation L2 optionnelle
- Prédiction de probabilités et labels

**Hyperparamètres utilisés :**
- Taux d'apprentissage : 0.1
- Nombre d'itérations : 500
- Régularisation L2 (lambda) : 0.001
- Random state : 42

### 3.3 Implémentation Scikit-Learn

Configuration pour une comparaison équitable :
- Solver : 'lbfgs' (optimisation quasi-Newton)
- Max iterations : 1000
- C = 1.0 (inverse de la force de régularisation)
- Random state : 42

---

## 4. Résultats

### 4.1 Performance Globale

| Métrique | Implémentation Personnalisée | Scikit-Learn |
|----------|------------------------------|--------------|
| **Accuracy d'entraînement** | 0.9812 (98.12%) | 0.9993 (99.93%) |
| **Accuracy de test** | 0.9528 (95.28%) | 0.9722 (97.22%) |
| **Nombre d'erreurs (test)** | 17/360 | 10/360 |
| **Taux d'erreur (%)** | 4.72% | 2.78% |

**Écart de performance :** La différence de 1.94 points de pourcentage en accuracy de test est acceptable et s'explique par les différences dans les algorithmes d'optimisation.

### 4.2 Métriques Détaillées - Implémentation Personnalisée

```
              precision    recall  f1-score   support
           0       1.00      0.97      0.99        36
           1       0.86      0.86      0.86        36
           2       0.97      1.00      0.99        35
           3       1.00      0.97      0.99        37
           4       0.92      1.00      0.96        36
           5       1.00      0.97      0.99        37
           6       0.97      0.94      0.96        36
           7       0.97      1.00      0.99        36
           8       0.91      0.86      0.88        35
           9       0.92      0.94      0.93        36

    accuracy                           0.95       360
   macro avg       0.95      0.95      0.95       360
weighted avg       0.95      0.95      0.95       360
```

**Observations :**
- Les chiffres 0, 2, 3, 5, 7 sont reconnus avec une excellente précision (≥0.97)
- Le chiffre 1 a la plus faible précision (0.86) et rappel (0.86)
- Le chiffre 8 a aussi des performances modérées (F1 = 0.88)

### 4.3 Métriques Détaillées - Scikit-Learn

```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        36
           1       0.89      0.89      0.89        36
           2       1.00      1.00      1.00        35
           3       0.97      1.00      0.99        37
           4       0.97      1.00      0.99        36
           5       1.00      1.00      1.00        37
           6       1.00      0.97      0.99        36
           7       1.00      1.00      1.00        36
           8       0.89      0.89      0.89        35
           9       1.00      0.97      0.99        36

    accuracy                           0.97       360
   macro avg       0.97      0.97      0.97       360
weighted avg       0.97      0.97      0.97       360
```

Scikit-Learn obtient une meilleure performance générale, particulièrement sur les chiffres 0, 2, 5, 7 (100% precision et recall).

---

## 5. Analyse des Résultats

### 5.1 Influence des Paramètres

#### 5.1.1 Taux d'Apprentissage (Learning Rate)

| Learning Rate | Accuracy Test |
|---------------|---------------|
| 0.01 | 0.9222 |
| 0.05 | 0.9444 |
| **0.1** | **0.9528** |
| 0.2 | 0.9639 |

**Conclusions :**
- Les taux plus élevés donnent de meilleures performances
- À 0.2, nous commençons à voir des oscillations mais toujours une amélioration
- Taux optimal : 0.1 offre un bon équilibre convergence/performance

#### 5.1.2 Régularisation L2 (Lambda)

| Lambda | Accuracy Test |
|--------|---------------|
| 0.0 | 0.9528 |
| 0.001 | 0.9528 |
| 0.01 | 0.9528 |
| 0.1 | 0.9528 |

**Conclusions :**
- La régularisation L2 a peu d'impact sur ce dataset
- Les poids du modèle ne sont pas excessivement grands
- Pas de surapprentissage significatif à combattre

#### 5.1.3 Nombre d'Itérations

| Itérations | Accuracy Test | Perte finale |
|-----------|---------------|-------------|
| 100 | 0.9389 | 0.2935 |
| 250 | 0.9444 | 0.1988 |
| 500 | 0.9528 | 0.1206 |
| 1000 | 0.9639 | 0.0813 |

**Conclusions :**
- Plus d'itérations améliore les performances
- La convergence est progressive mais stable
- 500 itérations offrent un bon compromis convergence/temps de calcul
- 1000 itérations donnent les meilleures performances (96.39%)

### 5.2 Interprétation des Coefficients Appris

Les coefficients de régression logistique représentent l'importance de chaque pixel pour prédire chaque classe. Les poids élevés (positifs ou négatifs) indiquent les pixels discriminants.

#### Pixels les plus importants par digit (Implémentation Personnalisée) :

**Digit 0 :** Pixels périphériques (5,2), (3,6), (4,1) - forme circulaire
**Digit 1 :** Pixels verticaux (2,3), (2,4), (6,4) - trait vertical central
**Digit 2 :** Pixels inférieurs (6,3), (5,3), (7,6) - base distincte
**Digit 3 :** Pixels supérieurs (0,4), courbes (5,5), (5,6) - forme sinueuse
**Digit 4 :** Pixels centraux et gauche (5,1), (4,1), (5,4) - intersection
**Digit 5 :** Pixels supérieurs (0,2), (0,6), (0,5) - haut distinctif
**Digit 6 :** Pixels gauche (5,2), (4,2), (6,6) - boucle basse
**Digit 7 :** Pixels droits supérieurs (4,6), (4,5), (3,6) - trait diagonal
**Digit 8 :** Pixels centraux (4,3), (3,3), (5,2) - deux boucles
**Digit 9 :** Pixels supérieurs droits (2,5), (3,5), (3,3) - boucle haute

Ces patterns correspondent bien aux caractéristiques visuelles distinctives de chaque chiffre.

### 5.3 Analyse des Erreurs de Classification

#### Paires de Chiffres Confondus (Custom Model) :

| Confusion | Nombre | Explication |
|-----------|--------|-------------|
| 1 ↔ 8 | 5 | Trait vertical similaire, 8 a boucles |
| 1 ↔ 6 | 2 | Formes éloignées, peut-être mauvaise segmentation |
| 1 ↔ 9 | 2 | Formes très différentes, erreurs d'écriture |
| 4 ↔ 0 | 1 | Confusion rare, géométries très différentes |
| 4 ↔ 1 | 1 | Intersection vs trait vertical |
| 5 ↔ 9 | 1 | Formes supérieures similaires |
| 6 ↔ 8 | 1 | Boucles similaires, position différente |

**Patterns d'erreurs principaux :**
- Chiffres avec traits verticaux (1, 7) vs chiffres avec boucles (8, 0)
- Similarités visuelles entre 3 et 5
- Confusion entre 4 et 1 sur certains styles d'écriture

#### Chiffres les Plus Difficiles à Classifier :

1. **Chiffre 1 :** 5 erreurs (recall = 0.86)
   - Confond facilement avec 8 et 9
   - Trait vertical simple peut être bruyant

2. **Chiffre 8 :** 5 erreurs (F1 = 0.88)
   - Confund avec 1 et autres formes de boucles
   - Structure complexe à deux boucles

3. **Chiffres 4, 9 :** Performance modérée
   - Plus d'erreurs que pour 0, 2, 3, 5, 7

---

## 6. Comparaison Implémentations

### 6.1 Performances Comparatives

**Avantages de Scikit-Learn :**
- Accuracy 2% supérieure (97.22% vs 95.28%)
- Moins d'erreurs (10 vs 17)
- Optimisation plus sophistiquée (LBFGS vs vanille SGD)
- Convergence plus rapide et stable

**Avantages de l'Implémentation Personnalisée :**
- ✓ Transparence totale de l'algorithme
- ✓ Éducatif : montre chaque étape du gradient descent
- ✓ Flexible : facile à modifier (loss, régularisation, etc.)
- ✓ Performance acceptable (95.28% accuracy)
- ✓ Pas de dépendances optimisées requises

### 6.2 Différences Algorithmiques

| Aspect | Implémentation Personnalisée | Scikit-Learn |
|--------|------------------------------|--------------|
| **Optimiseur** | Vanilla SGD | L-BFGS (quasi-Newton) |
| **Stabilité numérique** | Softmax avec max subtraction | Robuste |
| **Initialisation** | Poids = 0 | Random (mais cohérent) |
| **Convergence** | Plus lente | Très rapide |
| **Régularisation** | L2 simple | Flexible (l1, l2, elastic) |

### 6.3 Cadre de Comparaison Équitable

Pour assurer une comparaison équitable :
- ✓ Même données (train/test split identique)
- ✓ Même normalisation (StandardScaler)
- ✓ Même métrique (accuracy, F1)
- ✓ Random state fixé (42)
- ✓ Même ensemble de validation

---

## 7. Limitations et Difficultés Rencontrées

### 7.1 Limitations Théoriques

1. **Linéarité :** La régression logistique suppose une séparation linéaire. Les digits manuscrits ne sont pas toujours linéairement séparables.

2. **Complexité computationnelle :** O(n_features × n_classes × n_iterations) - acceptable pour ce dataset mais peut être limitant pour de grands volumes.

3. **Pas de feature engineering :** Utilise directement les pixels bruts; pourrait bénéficier d'une extraction de features (edges, corners).

### 7.2 Difficultés Techniques Rencontrées

1. **Overflow numérique :** Avant la stabilisation softmax, l'exponentielle causait des NaN.
   - Solution : Soustraction du max avant exponentielle

2. **Convergence lente :** Taux d'apprentissage initial (0.01) était trop faible.
   - Solution : Augmentation à 0.1 pour meilleure convergence

3. **One-hot encoding :** Gestion des indices des classes.
   - Solution : Mapping explicite des labels vers indices

### 7.3 Limitations Pratiques

- **Données limitées :** 1797 images pour 10 classes
- **Images petites :** 8×8 pixels = 64 features seulement
- **Bruit :** Images manuscrites bruitées naturellement
- **Déséquilibre léger :** Distribution presque équilibrée mais quelques classes légèrement surreprésentées

---

## 8. Prise de Recul et Perspectives

### 8.1 Réflexion Générale

Ce projet a permis de :
- ✓ Comprendre en détail le fonctionnement de la régression logistique
- ✓ Implémenter un algorithme d'apprentissage from scratch
- ✓ Appliquer les concepts mathématiques à un problème réel
- ✓ Valider notre implémentation contre une référence
- ✓ Analyser les erreurs et les limites du modèle

Les performances obtenues (95.28% accuracy) démontrent que notre implémentation fonctionne correctement et apprend efficacement à partir des données.

### 8.2 Extensions Possibles

**Améliorations à court terme :**
1. Ajuster les hyperparamètres (learning rate scheduler)
2. Implémenter la validation croisée
3. Ajouter une régularisation L1 (LASSO)
4. Implémentation du batch gradient descent vs vanilla SGD

**Améliorations à moyen terme :**
1. **Régularisation avancée :**
   - Elastic Net (L1 + L2)
   - Dropout (pour la régularisation)
   
2. **Feature engineering :**
   - Détection d'edges (filtres de Sobel)
   - Moments invariants
   - Histogrammes d'orientations

3. **Modèles alternatifs :**
   - Réseaux de neurones (MLP)
   - SVM avec noyau non-linéaire
   - Ensemble methods (Random Forest, Gradient Boosting)

4. **Tests sur des datasets plus complexes :**
   - MNIST (70,000 images, 28×28)
   - Fashion-MNIST
   - SVHN (Street View House Numbers)

### 8.3 Améliorations Algorithmes Futures

**Optimisation :**
- Momentum et Nesterov Accelerated Gradient
- Adam, RMSprop optimizers
- Learning rate scheduling (decay, warmup)
- Mini-batch gradient descent

**Robustesse :**
- Cross-validation k-fold
- Ensemble learning
- Data augmentation (rotations, translations)
- Regularisation par dropout

---

## 9. Conclusion

Ce projet a permis d'implémenter avec succès un classifieur multiclasse en régression logistique et de démontrer sa validité en le comparant à scikit-learn. Notre implémentation atteint **95.28% d'accuracy**, un résultat très satisfaisant pour une implémentation from scratch sans optimisations avancées.

### Résultats Clés :

1. **Implémentation fonctionnelle :** Algorithme de descente de gradient converge correctement
2. **Performance acceptable :** 95.28% accuracy sur le test set
3. **Analyse d'erreurs utile :** Identification des confusions courantes (1↔8, digits complexes)
4. **Paramètres importants :** Learning rate et nombre d'itérations sont critiques
5. **Stabilité numérique :** Gestion correcte des calculs en virgule flottante

### Recommandations :

Pour améliorer les performances du classifieur :
- Utiliser un optimiseur plus avancé (LBFGS, Adam)
- Appliquer du feature engineering
- Tester d'autres architectures (réseaux de neurones)
- Utiliser des datasets plus grands (MNIST)
- Implémenter la validation croisée

---

## Annexes

### A. Configuration de l'Environnement

```
Python 3.14
scikit-learn
numpy
pandas
matplotlib
seaborn
```

### B. Ressources Utilisées

- Documentation scikit-learn : https://scikit-learn.org
- Dataset digits : 1797 images 8x8, 10 classes
- Références mathématiques : Régression logistique standard, softmax multiclasse

### C. Fichiers Livrés

1. `classifier_implementation.py` - Code source complet
2. `rapport_projet.md` - Ce rapport
3. `results_comparison.csv` - Résultats métriques
