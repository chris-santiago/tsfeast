"""Miscellaneous utility functions."""
from typing import List, Union

import numpy as np
import pandas as pd

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
