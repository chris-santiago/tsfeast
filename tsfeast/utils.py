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
    """Plot regression diagnostics."""
    if not residuals and not estimator:
        raise ValueError('Either residuals or estimator and X, y must be given.')
    if estimator and not X or not y:
        raise ValueError('Both X and y must be given if passing an estimator.')
    if not residuals:
        residuals = y - estimator.predict(X)
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    ax1.plot(residuals)
    ax1.set_title('Residuals')
    qqplot(residuals, line='s', ax=ax2)
    ax2.set_title('QQ-Plot')
    plot_acf(residuals, ax=ax3)
    plot_pacf(residuals, ax=ax4)
