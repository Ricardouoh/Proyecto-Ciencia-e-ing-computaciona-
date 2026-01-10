from __future__ import annotations
"""
Custom transformers used in preprocessing.
"""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AgeWeightAdder(BaseEstimator, TransformerMixin):
    """
    Adds a weighted copy of the age feature after scaling.
    If weight == 1.0 or age_index is None, it returns X unchanged.
    """

    def __init__(self, age_index: Optional[int], weight: float = 1.0, add_copy: bool = True):
        self.age_index = age_index
        self.weight = weight
        self.add_copy = add_copy

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        if self.age_index is None or self.weight == 1.0:
            return X
        X = np.asarray(X)
        if self.age_index >= X.shape[1]:
            return X
        age = X[:, self.age_index]
        if self.add_copy:
            return np.column_stack([X, age * self.weight])
        X2 = X.copy()
        X2[:, self.age_index] = X2[:, self.age_index] * self.weight
        return X2

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        if self.age_index is None or self.weight == 1.0:
            return np.asarray(input_features, dtype=object)
        if self.add_copy:
            out = list(input_features) + [f"{input_features[self.age_index]}_weighted"]
            return np.asarray(out, dtype=object)
        return np.asarray(input_features, dtype=object)
