# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:22:15 2025
@author: BLETHIEC
"""

import numpy as np
from descente_stochastique import GradientDescent
import pandas as pd


df = pd.read_csv('titanic2.csv')

def grad_cout(data):
    