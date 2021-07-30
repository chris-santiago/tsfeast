import numpy as np
import pandas as pd
import pytest

from tsfeast.splitter import EndogSeriesWindows, TimeSeriesWindows


class TestTimeSeriesWindows:
    def test_split_sets(self, endog_uni, exog):
        tsw = TimeSeriesWindows(train_length=3, test_length=1, gap_length=0)
        windows = tsw.split(endog_uni, exog)
        for w in windows:
            assert len(w) == 4

    @pytest.mark.parametrize(
        'train_length, test_length', [
            (2, 1),
            (3, 1),
            (3, 2)
        ]
    )
    def test_num_splits(self, endog_uni, exog, train_length, test_length):
        tsw = TimeSeriesWindows(train_length=train_length, test_length=test_length, gap_length=0)
        windows = tsw.split(endog_uni, exog)
        assert len(windows) == (len(endog_uni) - train_length - (test_length - 1))

    @pytest.mark.parametrize(
        'train_length, test_length, gap_length', [
            (2, 1, 1),
            (3, 1, 2),
            (3, 2, 1)
        ]
    )
    def test_gap_n_splits(self, endog_uni, exog, train_length, test_length, gap_length):
        tsw = TimeSeriesWindows(train_length=train_length, test_length=test_length, gap_length=gap_length)
        windows = tsw.split(endog_uni, exog)
        assert len(windows) == (len(endog_uni) - (train_length + gap_length) - (test_length - 1))

    @pytest.mark.parametrize(
        'train_length, test_length, gap_length', [
            (2, 1, 1),
            (3, 1, 2),
            (3, 2, 0)
        ]
    )
    def test_split_shape(self, endog_uni, exog, train_length, test_length, gap_length):
        tsw = TimeSeriesWindows(train_length=train_length, test_length=test_length, gap_length=gap_length)
        windows = tsw.split(endog_uni, exog)
        for split in windows:
            x_train, x_test, y_train, y_test = split
            assert x_train.shape[0] == train_length
            assert x_test.shape[0] == test_length
            assert y_train.shape[0] == train_length
            assert y_test.shape[0] == test_length

    @pytest.mark.parametrize(
        'train_length, test_length, gap_length', [
            (2, 1, 1),
            (3, 1, 2),
            (3, 2, 0)
        ]
    )
    def test_gap_exists(self, endog_uni, exog, train_length, test_length, gap_length):
        tsw = TimeSeriesWindows(train_length=train_length, test_length=test_length, gap_length=gap_length)
        windows = tsw.split(endog_uni, exog)
        for split in windows:
            x_train, x_test, y_train, y_test = split
            assert (x_train.index[-1] + pd.tseries.offsets.MonthEnd(1+gap_length)) == x_test.index[0]
            assert (y_train.index[-1] + pd.tseries.offsets.MonthEnd(1 + gap_length)) == y_test.index[0]
