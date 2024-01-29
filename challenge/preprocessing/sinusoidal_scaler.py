import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SinusoidalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_value, column_map):
        self.base_value = base_value
        self.column_map = column_map
        self.__name = None

    def fit(self, X, y=None):
        return self

    @staticmethod
    def get_feature_names_out(input_features):
        feature_names = []
        for col in input_features:
            feature_names.extend([col, f"{col}_cos", f"{col}_sin"])
        return feature_names

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            self.__name = X.columns
            X = X.map(lambda x: self.column_map[x])

        X = np.array(X)

        # sinusoidal transformation
        sin_transformed = np.sin(2 * np.pi * X / self.base_value)
        cos_transformed = np.cos(2 * np.pi * X / self.base_value)
        transformed = np.hstack((X, sin_transformed, cos_transformed))

        return transformed
