"""Module for Scikit-Learn Regressor with ARMA Residuals."""
from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel
from statsmodels.tsa.arima.model import ARIMA

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
        if self.use_exog:
            self.fitted_model_ = self.model(endog=y, exog=X, **self.kwargs).fit()
        else:
            self.fitted_model_ = self.model(endog=y, **self.kwargs).fit()
        self.fitted_values_ = self.fitted_model_.fittedvalues
        self.summary_ = self.fitted_model_.summary()
        return self

    def _predict(self, X: Data) -> Data:
        """Predict the response."""
        return self.fitted_model_.forecast(steps=X.shape[0])
