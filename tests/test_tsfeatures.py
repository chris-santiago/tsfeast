import numpy as np
import pandas as pd
import pytest

from tsfeast.tsfeatures import TimeSeriesFeatures


class TestTimeSeriesFeatures:
    def test_fit_constant(self, exog):
        ts = TimeSeriesFeatures(datetime='index', trend='c')
        actual = ts.fit_transform(exog.reset_index())
        assert actual['const'].sum() == float(len(exog))

    def test_fit_trend(self, exog):
        ts = TimeSeriesFeatures(datetime='index', trend='ct')
        actual = ts.fit_transform(exog.reset_index())['trend']
        expected = list(range(1, len(exog)+1))
        np.testing.assert_allclose(actual, expected)

    def test_fit_datetime_only(self, exog, curr_dir):
        ts = TimeSeriesFeatures(datetime='index', interactions=False)
        actual = ts.fit_transform(exog.reset_index())
        fp = curr_dir.joinpath('valid_outputs', 'dt_transform_only.json')
        expected = pd.read_json(fp).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)

    def test_fit_all(self, exog, curr_dir):
        ts = TimeSeriesFeatures(
            datetime='index', trend='ct', lags=4, rolling=[3, 6], ewma=[3, 6], pct_chg=[1, 12],
            diffs=1, polynomial=2, interactions=True
        )
        actual = ts.fit_transform(exog.reset_index())
        fp = curr_dir.joinpath('valid_outputs', 'all_transforms.json')
        expected = pd.read_json(fp).reset_index(drop=True).fillna(0)
        pd.testing.assert_frame_equal(actual, expected, check_freq=False, check_dtype=False)
