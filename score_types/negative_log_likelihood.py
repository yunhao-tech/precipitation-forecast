from tensorflow.keras.losses import BinaryCrossentropy

from .base import BaseScoreType
import numpy as np

class NegativeLogLikelihood(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='negative log likelihood', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_proba, y_proba):
        bce = BinaryCrossentropy()
        y_true_proba = y_true_proba.reshape(y_true_proba.shape[0], -1)
        y_proba = y_proba.reshape(y_proba.shape[0], -1)
        score = bce(y_true_proba, y_proba).numpy()
        return score