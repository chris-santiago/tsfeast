"""Module for Base Estimator."""
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from tsfeast.utils import Data


class BaseContainer(BaseEstimator, RegressorMixin):
    """Container class for Scikit-Learn models."""
    def __init__(self):
        """Instantiate container."""

    def _fit(self, X: Data, y: Data) -> Data:
        """Method not implemented."""
        raise NotImplementedError

    def fit(self, X: Data, y: Data) -> "BaseContainer":
        """
        Fit the estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).
        Returns
        -------
        BaseContainer
            Self.
        """
        X, y = check_X_y(X, y)
        self._fit(X, y)
        return self

    def _predict(self, X: Data) -> Data:
        """Method not implemented."""
        raise NotImplementedError

    def predict(self, X: Data) -> Data:
        """
        Make predictions with fitted estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        np.ndarray
            Array of predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self._predict(X)
