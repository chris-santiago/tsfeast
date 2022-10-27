from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tests.conftest import HERE, X, train_test
from tsfeast.funcs import (
    get_change_features,
    get_datetime_features,
    get_difference_features,
    get_ewma_features,
    get_lag_features,
    get_rolling_features,
)
from tsfeast.transformers import (
    BaseTransformer,
    ChangeFeatures,
    DateTimeFeatures,
    DifferenceFeatures,
    EwmaFeatures,
    InteractionFeatures,
    LagFeatures,
    OriginalFeatures,
    PolyFeatures,
    RollingFeatures,
    Scaler,
)

x_train, y_train, x_test, y_test = train_test()


class TestBaseTransformer:
    def test_array(self, monkeypatch, exog):
        t = OriginalFeatures()
        mock = MagicMock(pd.DataFrame)
        monkeypatch.setattr('tsfeast.transformers.array_to_dataframe', mock)
        t.transform(exog.values)
        mock.assert_called()

    def test_transform_raises(self, exog):
        with pytest.raises(NotImplementedError):
            t = BaseTransformer()
            t._transform(exog)


def test_inverse_scaler(exog):
    sc = Scaler()
    trans = sc.fit_transform(exog)
    actual = sc.inverse_transform(trans)
    expected = X.astype(float)
    pd.testing.assert_frame_equal(actual, expected)


def test_bad_data_raises():
    dt = DateTimeFeatures('index')
    with pytest.raises(ValueError):
        dt.fit([[1, 2, 3], [4, 5, 6]])


class TestOriginalFeatures:
    @pytest.mark.parametrize(
        'input, transformer, expected',
        [
            (X, OriginalFeatures(), X),
            (X.reset_index(), DateTimeFeatures('index'), get_datetime_features(X.reset_index(), 'index')),
            (X, LagFeatures(2), get_lag_features(X, n_lags=2).fillna(0)),
            (X, LagFeatures(2, fillna=False), get_lag_features(X, n_lags=2)),
            (X, RollingFeatures([4]), get_rolling_features(X, [4]).fillna(0)),
            (X, EwmaFeatures([4]), get_ewma_features(X, [4]).fillna(0)),
            (X, ChangeFeatures([4]), get_change_features(X, [4]).fillna(0)),
            (X, DifferenceFeatures(2), get_difference_features(X, 2).fillna(0)),
            (X, PolyFeatures(3), pd.read_json(HERE.joinpath('valid_outputs', 'poly_[2, 3].json'))),
            (X, InteractionFeatures(), pd.read_json(HERE.joinpath('valid_outputs', 'interactions.json')).astype(float)),
            (X, Scaler(), pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index))
        ]
    )
    def test_solo(self, endog_uni, input, transformer, expected):
        feat = transformer
        actual = feat.fit_transform(input, endog_uni)
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)

    @pytest.mark.parametrize(
        'transformer, expected',
        [
            (OriginalFeatures(), x_test),
            (DateTimeFeatures('index'), get_datetime_features(x_test.reset_index(), 'index', freq='M')),
            (LagFeatures(2), get_lag_features(X, n_lags=2).fillna(0).iloc[-len(x_test):, :]),
            (RollingFeatures([4]), get_rolling_features(X, [4]).fillna(0).iloc[-len(x_test):, :]),
            (EwmaFeatures([4]), get_ewma_features(X, [4]).fillna(0).iloc[-len(x_test):, :]),
            (ChangeFeatures([4]), get_change_features(X, [4]).fillna(0).iloc[-len(x_test):, :]),
            (DifferenceFeatures(2), get_difference_features(X, 2).fillna(0).iloc[-len(x_test):, :]),
            (PolyFeatures(3), pd.read_json(HERE.joinpath('valid_outputs', 'poly_[2, 3].json')).iloc[-len(x_test):, :]),
            (InteractionFeatures(), pd.read_json(HERE.joinpath('valid_outputs', 'interactions.json')).iloc[-len(x_test):, :].astype(float))
        ]
    )
    def test_pipeline(self, transformer, expected):
        pl = Pipeline([
            ('features', transformer),
            ('regression', LinearRegression())
        ])
        if isinstance(transformer, DateTimeFeatures):
            pl.fit(x_train.reset_index(), y_train)
            pl.predict(x_test.reset_index())
        else:
            pl.fit(x_train, y_train)
            pl.predict(x_test)
        actual = pl.named_steps.features.output_features_
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)
