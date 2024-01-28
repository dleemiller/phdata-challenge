import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Scaler999(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        return (X == 999).astype(int)
