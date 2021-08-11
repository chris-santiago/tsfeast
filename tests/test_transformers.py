import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from tests.conftest import X
from tsfeast.transformers import LagFeatures, OriginalFeatures


class TestOriginalFeatures:
    def test_solo(self, exog, endog_uni):
        feat = OriginalFeatures()
        actual = feat.fit_transform(exog, endog_uni)
        expected = exog
        pd.testing.assert_frame_equal(actual, expected)

    def test_pipeline(self, train_test):
        x_train, y_train, x_test, y_test = train_test
        pl = Pipeline([
            ('features', OriginalFeatures()),
            ('regression', LinearRegression())
        ])
        pl.fit(x_train, y_train)
        pl.predict(x_test)
        actual = pl.named_steps.features.output_features_
        expected = x_test
        pd.testing.assert_frame_equal(actual, expected)
