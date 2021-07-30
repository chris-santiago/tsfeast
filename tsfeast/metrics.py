from numpy import log
from sklearn.metrics import mean_squared_error


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


def bic_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    mse = mean_squared_error(y, y_pred)
    return bic_score(mse, X.shape[0], X.shape[1])
