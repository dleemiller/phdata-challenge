import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SinusoidalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_value, column_map):
        self.base_value = base_value
        self.column_map = column_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.map(lambda x: self.column_map[x])

        X = np.array(X)

        # sinusoidal transformation
        sin_transformed = np.sin(2 * np.pi * X / self.base_value)
        cos_transformed = np.cos(2 * np.pi * X / self.base_value)
        transformed = np.hstack((sin_transformed, cos_transformed))

        return transformed
