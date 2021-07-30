import pytest

from tsfeast.funcs import *


class TestGetBusdaysInMonth:
    def test_dt_monthbegin(self):
        actual = get_busdays_in_month(pd.to_datetime('2020-01-01'))
        expected = 21
        assert actual == expected

    def test_dt_monthend(self):
        actual = get_busdays_in_month(pd.to_datetime('2020-01-31'))
        expected = 21
        assert actual == expected

    @pytest.mark.parametrize(
        "test_date, result", [
            (pd.to_datetime('2020-01-15'), 21),
            (pd.to_datetime('2020-02-15'), 19),
            (pd.to_datetime('2020-04-15'), 22),
        ]
    )
    def test_dt_other(self, test_date, result):
        actual = get_busdays_in_month(pd.to_datetime(test_date))
        expected = result
        assert actual == expected


class TestGetDatetimeFeatures:
    def test_no_date_col_raises(self, exog, curr_dir):
        with pytest.raises(ValueError):
            actual = get_datetime_features(exog.reset_index())

    def test_bad_type_raise(self, exog):
        with pytest.raises(ValueError):
            actual = get_datetime_features(exog.reset_index().to_dict())

    def test_outputs(self, exog, curr_dir):
        actual = get_datetime_features(exog.reset_index(), date_col='index')
        fp = curr_dir.joinpath('valid_outputs', 'uni_dt_features.json')
        expected = pd.read_json(fp).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)


class TestGetLagFeatures:
    @pytest.mark.parametrize(
        "lag", [1, 3, 6]
    )
    def test_multivariate(self, exog, lag, curr_dir):
        actual = get_lag_features(exog, lag)
        fp = curr_dir.joinpath('valid_outputs', f'multi_lags_{lag}.json')
        expected = pd.read_json(fp)
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)


class TestGetRollingFeatures:
    @pytest.mark.parametrize(
        "win", [[2], [3, 6]]
    )
    def test_multivariate(self, exog, win, curr_dir):
        actual = get_rolling_features(exog, win)
        fp = curr_dir.joinpath('valid_outputs', f'multi_rolling_{win}.json')
        expected = pd.read_json(fp)
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)

    @pytest.mark.parametrize(
        "i, expected", [
            (0, 135.0),
            (1, 141.0),
            (6, 23.5),
            (10, 9.35414),
            (15, 10.0),
            (20, 35.0)
        ]
    )
    def test_metrics(self, exog, i, expected):
        actual = get_rolling_features(exog, [6]).iloc[5, i]
        assert round(actual, 5) == expected


class TestGetEwmFeatures:
    @pytest.mark.parametrize(
        "win", [[2], [3, 6]]
    )
    def test_multivariate(self, exog, win, curr_dir):
        actual = get_ewma_features(exog, win)
        fp = curr_dir.joinpath('valid_outputs', f'multi_ewm_{win}.json')
        expected = pd.read_json(fp)
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)


class TestGetChangeFeatures:
    @pytest.mark.parametrize(
        "win", [1, 6]
    )
    def test_multivariate(self, exog, win, curr_dir):
        actual = get_change_features(exog, win)
        fp = curr_dir.joinpath('valid_outputs', f'multi_pct_chg_{win}.json')
        expected = pd.read_json(fp)
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)


class TestGetDifferenceFeatures:
    def test_multivariate(self, exog, curr_dir):
        actual = get_difference_features(exog, 2)
        fp = curr_dir.joinpath('valid_outputs', f'differences.json')
        expected = pd.read_json(fp)
        np.testing.assert_allclose(actual, expected)
