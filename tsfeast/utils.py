"""Miscellaneous utility functions."""
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

Data = Union[pd.DataFrame, pd.Series, np.ndarray]


def to_list(x: Union[int, List]) -> List[int]:
    """Ensure parameter is list of integer(s)."""
    if isinstance(x, list):
        return x
    return [x]


def array_to_dataframe(x: np.ndarray) -> pd.DataFrame:
    """Convert Numpy array to Pandas DataFrame with default column names."""
    return pd.DataFrame(x, columns=[f'x{i}' for i in range(x.shape[1])])


def array_to_series(x: np.ndarray) -> pd.Series:
    """Convert Numpy array to Pandas Series with default name."""
    return pd.Series(x, name='y')


def plot_diag(
        residuals: Optional[Data] = None, estimator: Optional[LinearModel] = None,
        X: Optional[Data] = None, y: Optional[Data] = None
):
    """
    Plot regression diagnostics.

    Generate residuals plot, QQ plot, ACF plot and PACF plot, given either an array-like object
    of residuals or and estimator and X and y data arrays.

    Parameters
    ----------
    residuals: Data
        Model residual errors.
    estimator: LinearModel
        Scikit-Learn generalized linear model.
    X: array of shape [n_samples, n_features]
            The input samples.
    y :  array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
        Target values (None for unsupervised transformations).

    Raises
    ------
    ValueError
        - If neither residuals or estimator provider.
        - If estimator provided without X and y data.
    """
    if residuals is None and estimator is None:
        raise ValueError('Either residuals or estimator and X, y must be given.')
    if estimator and (X is None or y is None):
        raise ValueError('Both X and y must be given if passing an estimator.')
    if residuals is None and y is not None and estimator is not None:
        residuals = y - estimator.predict(X)
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    ax1.plot(residuals)
    ax1.set_title('Residuals')
    qqplot(residuals, line='s', ax=ax2)
    ax2.set_title('QQ-Plot')
    plot_acf(residuals, ax=ax3)
    plot_pacf(residuals, ax=ax4)
