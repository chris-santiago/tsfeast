"""
Time Series Windows Module.

*Note* These classes split data into n, equal-length, sliding training/test windows.  This differs
from Scikit-Learn's TimeSeriesSplit implementation where windows are accumulated:

Scikit-Learn
-------------
Win 0   |----|
Win 1   |--------|
Win2    |------------|

TimeSeriesWindows
-----------------
Win 0   |----|--------
Win 1   -|----|-------
Win2    --|----|------
"""
from typing import List, Optional

import pandas as pd

# pylint: disable=missing-docstring


class TimeSeriesWindows:
    def __init__(self, train_length: int, test_length: int, gap_length: int = 0) -> None:
        self.train_length = train_length
        self.test_length = test_length
        self.gap_length = gap_length

    def split(self, y: pd.DataFrame, x: pd.DataFrame) -> List[pd.DataFrame]:
        windows = []
        for i in range(len(x)):
            train_start = i
            train_end = i + self.train_length
            test_start = train_end + self.gap_length
            test_end = test_start + self.test_length
            if test_end <= len(x):
                x_train = x.iloc[train_start:train_end]
                y_train = y.iloc[train_start:train_end]
                x_test = x.iloc[test_start:test_end]
                y_test = y.iloc[test_start:test_end]
                split = x_train, x_test, y_train, y_test
                windows.append(split)
        return windows


class EndogSeriesWindows(TimeSeriesWindows):
    def __init__(
            self, min_train_length: int, test_length: int, max_train_length: Optional[int] = None,
            gap_length: int = 0) -> None:
        super().__init__(min_train_length, test_length, gap_length)
        self.min_train_length = min_train_length
        self.max_train_length = max_train_length

    def split(self, y: pd.DataFrame, x=None) -> List[pd.DataFrame]:
        windows = []
        for i in range(self.min_train_length, len(y)):
            if i + self.test_length <= len(y):
                if self.max_train_length is not None:
                    train_start = i - self.max_train_length
                else:
                    train_start = 0
                train_end = i
                if self.gap_length is not None:
                    test_start = train_end + self.gap_length
                else:
                    test_start = train_end
                test_end = test_start + self.test_length
                y_train = y.iloc[train_start:train_end]
                y_test = y.iloc[test_start:test_end]
                split = y_train, y_test
                windows.append(split)
        return windows
