"""Module for Base Estimator"""
from typing import Optional, Tuple

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SciKitContainer(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def _fit(self, X, y):
        raise NotImplementedError

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._fit(X, y)
        return self

    def _predict(self, X):
        raise NotImplementedError

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self._predict(X)


class StatsModelsContainer(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def _fit(self, X, y):
        raise NotImplementedError

    def fit(self, X=None, y=None):
        X, y = check_X_y(X, y)
        self._fit(X, y)
        return self

    def _predict(self, X):
        raise NotImplementedError

    def predict(self, X=None):
        check_is_fitted(self)
        X = check_array(X)
        return self._predict(X)
