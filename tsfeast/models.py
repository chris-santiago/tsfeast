"""
Module for Scikit-Learn Regressor with ARMA Residuals and
Scikit-Learn API wrapper for Statsmodels TSA models.
"""
from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel
from statsmodels.base.model import Model
from statsmodels.tsa.arima.model import ARIMA

from tsfeast._base import BaseContainer
from tsfeast.utils import Data


class ARMARegressor(BaseContainer):
    """Estimator for Scikit-Learn estimator with ARMA residuals."""
    def __init__(
            self, estimator: LinearModel = LinearRegression(),
            order: Tuple[int, int, int] = (1, 0, 0)
    ):
        """
        Instantiate ARMARegressor object.

        Parameters
        ----------
        estimator: LinearRegression
            Scikit-Learn linear estimator.
        order: Tuple[int, int, int]
            ARIMA order for residuals.

        """
        super().__init__()
        self.estimator = estimator
        self.order = order

    def _fit(self, X: Data, y: Data) -> "ARMARegressor":
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
        ARMARegressor
            Self.
        """
        self.estimator.fit(X, y)
        self.intercept_ = self.estimator.intercept_
        self.coef_ = self.estimator.coef_
        estimator_fitted = self.estimator.predict(X)
        estimator_resid = np.ravel(y) - estimator_fitted
        self.arma_ = ARIMA(estimator_resid, order=self.order).fit()
        self.fitted_values_ = estimator_fitted + self.arma_.fittedvalues
        self.resid_ = np.ravel(y) - self.fitted_values_
        return self

    def _predict(self, X: Data) -> Data:
        """
        Predict the response.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        np.ndarray
            Array of predicted values.
        """
        estimator_pred = self.estimator.predict(X)
        arma_pred = self.arma_.forecast(steps=estimator_pred.shape[0])  # todo this won't work for in sample predictions
        return estimator_pred + arma_pred


class TSARegressor(BaseContainer):
    """Estimator for StatsModels TSA model."""
    def __init__(self, model: Model, use_exog: bool = False, **kwargs):
        """
        Instantiate TSARegressor object.

        model: Model
            An uninstantiated Statsmodels TSA model.
        use_exog: bool
            Whether to use exogenous features; default False.
        kwargs:
            Additional kwargs for Statsmodels model.
        """
        super().__init__()
        self.model = model
        self.use_exog = use_exog
        self.kwargs = kwargs

    def _fit(self, X: Data, y: Data) -> "TSARegressor":
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
        TSARegressor
            Self.
        """
        if self.use_exog:
            self.fitted_model_ = self.model(endog=y, exog=X, **self.kwargs).fit()
        else:
            self.fitted_model_ = self.model(endog=y, **self.kwargs).fit()
        self.fitted_values_ = self.fitted_model_.fittedvalues
        self.summary_ = self.fitted_model_.summary()
        return self

    def _predict(self, X: Data) -> Data:
        """
        Predict the response.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        np.ndarray
            Array of predicted values.
        """
        return self.fitted_model_.forecast(steps=X.shape[0])
