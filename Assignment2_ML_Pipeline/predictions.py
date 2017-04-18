"""
Module for predictions.
"""

import random
import numpy as np

class Zero_Predictor:
    """
    Class for predictor that always predicts 0.
    """
    def __init__(self):
        self.probability = 0

    def predict(self, X = None):
        """
        Predicts 0 once or for len of input.
        """
        if X is not None:
            predictions = [0 for i in range(len(X))]
            return np.asarray(predictions)
        else:
            return 0

    def score(self, X, y):
        """
        Making predictions and calculating score on X.
        In:
            - X: features
            - y: true target
        """
        predictions = self.predict(X)
        true_value = np.asarray(y)

        score = 1 - sum(np.absolute(predictions - true_value)) / len(predictions)

        return score


class Average_Predictor:
    """
    Class for predictor that makes predictions according
    to average of target vector: how well would we do if
    we randomly predict according to mean of training data?
    """
    def __init__(self):
        self.probability = 0

    def fit(self, target):
        """
        Function to 'train' the model, i.e. determine the
        mean of the target variable.
        """
        self.probability = target.mean()

    def predict(self, X = None):
        """
        Function to make predictions by creating random
        number and seeing if generated number is greater than
        mean of target variable.
        """
        if X is not None:
            predictions = [1 if random.random() < self.probability else 0 for i in range(len(X))]
            return np.asarray(predictions)
        else:
            if random.random() < self.probability:
                return 1
            else:
                return 0

    def score(self, X, y):
        """
        Making predictions and calculating score on X.
        In:
            - X: features
            - y: true target
        """
        predictions = self.predict(X)
        true_value = np.asarray(y)

        score = 1 - sum(np.absolute(predictions - true_value)) / len(predictions)

        return score
