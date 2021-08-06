"""Module for Scikit-Learn Regressor with ARMA Residuals."""
from typing import Tuple, Optional

import numpy as np
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.linear_model._base import LinearModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg, AutoRegResults

from tsfeast._base import SciKitContainer, StatsModelsContainer
from tsfeast.utils import Data


class ARMARegressor(SciKitContainer):
    """Estimator for Scikit-Learn estimator with ARMA residuals."""
    def __init__(
            self, estimator: LinearModel = LinearRegression(),
            order: Tuple[int, int, int] = (1, 0, 0)
    ):
        """Instantiate ARMARegressor object."""
        super().__init__()
        self.estimator = estimator
        self.order = order

    def _fit(self, X: Data, y: Data) -> "ARMARegressor":
        """Fit the estimator."""
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
        """Predict the response."""
        estimator_pred = self.estimator.predict(X)
        arma_pred = self.arma_.forecast(steps=estimator_pred.shape[0])
        return estimator_pred + arma_pred


class TSARegressor(StatsModelsContainer):
    """Estimator for StatsModels TSA model."""
    def __init__(self, model, use_exog=False, **kwargs):
        """Instantiate TSARegressor object."""
        super().__init__()
        self.model = model
        self.use_exog = use_exog
        self.kwargs = kwargs

    def _fit(self, X: Data, y: Data) -> "TSARegressor":
        """Fit the estimator."""
        if not self.use_exog:
            X = None
        self.fitted_model_ = self.model(endog=y, exog=X, **self.kwargs).fit()
        self.fitted_values_ = self.fitted_model_.fittedvalues
        self.summary_ = self.fitted_model_.summary()
        return self

    def _predict(self, X: Data) -> Data:
        """Predict the response."""
        return self.fitted_model_.forecast(steps=X.shape[0])


class PoissonAutoReg(SciKitContainer):
    """Estimator for Generalized Poisson model with autoregressive residuals."""
    def __init__(self, lags: int = 1):
        """Constructor"""
        super().__init__()
        self.lags = lags
        self._poisson_fit: Optional[PoissonRegressor] = None
        self._autoreg_fit: Optional[AutoRegResults] = None

    def _fit_poisson(self, X, y) -> None:
        """Fit a Poisson model."""
        mod = PoissonRegressor()
        self._poisson_fit = mod.fit(X, y)
        self.intercept_ = mod.intercept_
        self.coef_ = mod.coef_
        self.resid_ = np.ravel(y) - mod.predict(X)

    def _fit_autoreg(self) -> None:
        """Fit an AR model."""
        mod = AutoReg(self.resid_, self.lags)
        self._autoreg_fit = mod.fit()

    def _fit(self, X: Data, y: Data) -> "PoissonAutoReg":
        """Fit the estimator."""
        self._fit_poisson(X, y)
        self._fit_autoreg()
        return self

    def _predict(self, X: Data) -> np.ndarray:
        """Predict the response."""
        if self._poisson_fit is None:
            raise AttributeError('Model must be fit before predicting.')
        if self._autoreg_fit is None:
            raise AttributeError('Model must be fit before predicting.')
        poisson_pred = self._poisson_fit.predict(X)
        autoreg_pred = self._autoreg_fit.forecast(len(X))
        return poisson_pred + autoreg_pred
