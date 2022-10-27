"""Custom scoring metrics."""
from numpy import log
from sklearn.metrics import mean_squared_error

from tsfeast._base import BaseContainer
from tsfeast.utils import Data


def bic_score(mse: float, n: int, p: int):
    """
    Calcuate BIC score.

    Parameters
    ----------
    mse: float
        Mean-squared error.
    n: int
        Number of observations.
    p: int
        Number of parameters

    Returns
    -------
    float
        BIC value.
    """
    return n * log(mse) + log(n) * p


def bic_scorer(estimator: BaseContainer, X: Data, y: Data):
    """Score SciKit-Learn estimator using BIC."""
    y_pred = estimator.predict(X)
    mse = mean_squared_error(y, y_pred)
    return bic_score(mse, X.shape[0], X.shape[1])
