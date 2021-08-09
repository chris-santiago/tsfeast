import numpy as np
import pandas as pd
import pytest

from tsfeast.utils import array_to_dataframe, array_to_series

ARR = np.array([
            [99.99999518, 100.99999518, 101.99999518, 102.99999518],
            [103.99999518, 104.99999518, 105.99999518, 106.99999518],
            [107.99999518, 108.99999518, 109.99999518, 110.99999518]
        ])


def test_array_to_dataframe():
    actual = array_to_dataframe(ARR)
    expected = pd.DataFrame(ARR, columns=['x0', 'x1', 'x2', 'x3'])
    pd.testing.assert_frame_equal(actual, expected)


def test_array_to_series():
    actual = array_to_series(np.array([0, 1, 2, 3, 4, 5]))
    expected = pd.Series(actual, name='y')
    pd.testing.assert_series_equal(actual, expected)
