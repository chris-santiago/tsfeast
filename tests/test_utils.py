import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from tsfeast.utils import array_to_dataframe, array_to_series, plot_diag

ARR = np.array([
            [99.99999518, 100.99999518, 101.99999518, 102.99999518],
            [103.99999518, 104.99999518, 105.99999518, 106.99999518],
            [107.99999518, 108.99999518, 109.99999518, 110.99999518]
        ])

X, y = make_regression(n_features=25)


def test_array_to_dataframe():
    actual = array_to_dataframe(ARR)
    expected = pd.DataFrame(ARR, columns=['x0', 'x1', 'x2', 'x3'])
    pd.testing.assert_frame_equal(actual, expected)


def test_array_to_series():
    actual = array_to_series(np.array([0, 1, 2, 3, 4, 5]))
    expected = pd.Series(actual, name='y')
    pd.testing.assert_series_equal(actual, expected)


class TestPlotDiag:
    def test_both_none_raises(self):
        with pytest.raises(ValueError):
            plot_diag(residuals=None, estimator=None)

    def test_no_x_raises(self):
        with pytest.raises(ValueError):
            plot_diag(estimator=LinearRegression, X=None, y=ARR)

    def test_no_y_raises(self):
        with pytest.raises(ValueError):
            plot_diag(estimator=LinearRegression, X=ARR, y=None)

    def test_estimator(self):
        lr = LinearRegression()
        lr.fit(X, y)
        plot_diag(estimator=lr, X=X, y=y)

    def test_resid(self):
        lr = LinearRegression()
        lr.fit(X, y)
        resid = y - lr.predict(X)
        plot_diag(resid)
