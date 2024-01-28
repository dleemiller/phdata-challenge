import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractBand(BaseEstimator, TransformerMixin):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        return ((X >= self.min_val) & (X < self.max_val)).astype(int)
