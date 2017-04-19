"""
Module for predictions.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

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

    def score(self, X, y, score='accuracy'):
        """
        Making predictions and calculating score on X.
        In:
            - X: features
            - y: true target
            - score: metric to use for evaluation.
                    options: accuracy, precision
        """
        predictions = self.predict(X)
        true_value = np.asarray(y)

        if score == "accuracy":
            return 1 - sum(np.absolute(predictions - true_value)) / len(predictions)

        elif score == "precision":
            return 0


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

    def score(self, X, y, score='accuracy'):
        """
        Making predictions and calculating score on X.
        In:
            - X: features
            - y: true target
            - score: metric to use for evaluation.
                    options: accuracy, precision
        """
        predictions = self.predict(X)
        true_value = np.asarray(y)

        if score == "accuracy":
            score = 1 - sum(np.absolute(predictions - true_value)) / len(predictions)

        elif score == "precision":
            tp = fp = 0
            for i in range(len(predictions)):
                if predictions[i] == 1 and true_value[i] == 1:
                    tp += 1
                elif predictions[i] == 1:
                    fp += 1
            if tp + fp == 0:
                score = 0
            else:
                score = tp / (tp + fp)

        return score

# Source:
# http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#sphx-glr-auto-examples-model-selection-randomized-search-py
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def calc_precision_recall(predicted_val, y, threshold):
    """
    Calculates precision and recall for given threshold.
    In:
        - predicted_val: numpy array of predicted scores
        - y: target values
        - threshold: threshold to use for calculation
    Out:
        - (precision_score, recall_score)
    """
    x = np.zeros(len(predicted_val))
    y = np.asarray(y)

    x[predicted_val >= threshold] = 1

    tp = fp = fn = 0
    for i in range(len(x)):
        if x[i] == 1 and y[i] == 1:
            tp += 1
        elif x[i] == 1:
            fp += 1
        elif x[i] == 0 and y[i] == 1:
            fn += 1
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    return precision, recall


def plot_precision_recall_threshold(threshold_list, precision_list, recall_list):
    """
    Function to plot precision recall by threshold curve.
    In:
        - threshold_list: list of threshold values
        - precision_list: list of achieved precision
        - recall_list: list of achieved recall
    Out:
        - plot
    """

    fig, ax1 = plt.subplots()

    _ = ax1.plot(threshold_list, precision_list, 'b.')
    _ = ax1.set_xlabel('threshold')

    _ = ax1.set_ylabel('precision', color='b')
    _ = ax1.set_ylim(0,1.1)
    _ = ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    _ = ax2.plot(threshold_list, recall_list, 'r.')

    _ = ax2.set_ylabel('recall', color='r')
    _ = ax2.set_ylim(0,1.1)
    _ = ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()


def precision_top_x(predictions, y, x):
    """
    Calculates precision for x first entries.
    In:
        - predictions: array of predicted scores
        - y: target values
        - x: number of first entries to consider
    Out:
        - precision_score
    """
    y = np.asarray(y)
    predictions = np.asarray(predictions)

    tp = fp = 0
    for i in range(x):
        if predictions[i] == 1 and y[i] == 1:
            tp += 1
        elif predictions[i] == 1:
            fp += 1
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    return precision
