import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Scaler999(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    @staticmethod
    def get_feature_names_out(input_features):
        return [f"{x}_999" for x in input_features]

    def transform(self, X):
        X = np.array(X)
        return (X == 999).astype(int)
