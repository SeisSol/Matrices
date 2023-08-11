import numpy as np
import os
import pickle

from quad_rules.QuadRule import QuadRule


class JaskowiecSukumar(QuadRule):
    def __init__(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        fn = "JaskowiecSukumar.pkl"
        with open(os.path.join(directory, fn), "rb") as f:
            self.data = pickle.load(f)
