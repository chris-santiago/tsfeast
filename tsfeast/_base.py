"""Module for Base Estimators."""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class BaseContainer(BaseEstimator, RegressorMixin):
    """Container class for Scikit-Learn models."""
    def __init__(self):
        """Instantiate container."""

    def _fit(self, X, y):
        """Method not implemented."""
        raise NotImplementedError

    def fit(self, X, y):
        """Fit the estimator."""
        X, y = check_X_y(X, y)
        self._fit(X, y)
        return self

    def _predict(self, X):
        """Method not implemented."""
        raise NotImplementedError

    def predict(self, X):
        """Make predictions with fitted estimator."""
        check_is_fitted(self)
        X = check_array(X)
        return self._predict(X)
