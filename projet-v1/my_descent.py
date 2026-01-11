# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:45:29 2024

@author: JDION
"""

"""
Création de la classe GradientDescent utilisée durant l'intégralité de ce cours
"""
import numpy as np


class GradientDescent:

    def __init__(self, gradient, learning_rate, max_iterations):
        """
        Initialise l'objet GradientDescent avec les paramètres nécessaires.

        Paramètres :
        - gradient : La fonction gradient de la fonction de coût.
        - learning_rate : Taux d'apprentissage (pas) pour la mise à jour des paramètres.
        - max_iterations : Nombre maximal d'itérations de l'algorithme de descente.
        """

        self.gradient = gradient
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def descent(self, initial_point, taux_erreur):
        """
        Effectue l'algorithme de descente de gradient.

        Paramètres :
        - initial_point : Le point de départ de l'algorithme (array 1D).
        - taux_erreur : Seuil de tolérance sur la norme du gradient.

        Retourne :
        - Le point optimal trouvé par l'algorithme.
        - Le nombre d'itérations effectuées.
        """
        nb_iter = 0
        for i in range(self.max_iterations):
            grad = self.gradient(initial_point)
            if np.linalg.norm(grad) <= taux_erreur:
                return (initial_point, nb_iter)
            else:
                initial_point = self.update(initial_point, grad)
                nb_iter += 1

        return (initial_point, nb_iter)

    def update(self, point, gradient_value):
        """
        Met à jour le point en utilisant le gradient et le taux d'apprentissage.

        Paramètres :
        - point : Le point à mettre à jour.
        - gradient_value : Le gradient de la fonction de coût au point donné.

        Retourne :
        - Le nouveau point après la mise à jour.
        """
        new_point = point - self.learning_rate * gradient_value

        return new_point
