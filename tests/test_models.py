import numpy as np
import pandas as pd
import pytest
from statsmodels.iolib.summary import Summary
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tsfeast.models import ARMARegressor, TSARegressor

VALID = {
    'arma': {
        'fit': np.array([
            99.99999518, 100.99999518, 101.99999518, 102.99999518,
            103.99999518, 104.99999518, 105.99999518, 106.99999518,
            107.99999518, 108.99999518, 109.99999518, 110.99999518
        ]),
        'resid': np.array([
            4.82455816e-06, 4.82455816e-06, 4.82455816e-06, 4.82455816e-06,
            4.82455816e-06, 4.82455816e-06, 4.82455816e-06, 4.82455816e-06,
            4.82455816e-06, 4.82455816e-06, 4.82455816e-06, 4.82455816e-06
        ]),
        'int': 97.6,
        'coef': np.array([0.04, 0.04, 0.04, 0.04, 0.04]),
        'arma_fit': np.array([
            -4.82455817e-06, -4.82455817e-06, -4.82455817e-06, -4.82455817e-06,
            -4.82455817e-06, -4.82455817e-06, -4.82455817e-06, -4.82455817e-06,
            -4.82455817e-06, -4.82455817e-06, -4.82455817e-06, -4.82455817e-06
        ])
    },
    'tsa': {
        'fit': np.array([
            100.00000001, 100., 100.99999999, 101.99999999,
            102.99999999, 103.99999999, 104.99999999, 105.99999999,
            106.99999999, 107.99999999, 108.99999999, 109.99999999
        ]),
        'forecast': np.array([
            110.99999999, 110.99999999, 110.99999999, 110.99999999,
            110.99999999, 110.99999999, 110.99999999, 110.99999999,
            110.99999999, 110.99999999, 110.99999999, 110.99999999
        ]),
        'with_kwargs': np.array([
            99.99969881, 100.99939108, 101.99987387, 103.00021847,
            104.00043141, 105.0005278, 106.00052128, 107.00042408,
            108.00024716, 109.00000034, 109.99969241, 110.99933123
        ]),
        'with_exog': np.array([
            100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 110.,
            111.
        ])
    },
}


class TestArmaRegressor:
    @pytest.mark.parametrize(
        'attr, expected',
        [
            ('fitted_values_', VALID['arma']['fit']),
            ('intercept_', VALID['arma']['int']),
            ('coef_', VALID['arma']['coef']),
            ('resid_', VALID['arma']['resid']),
        ]
    )
    def test_fit(self, exog, endog_uni, attr, expected):
        mod = ARMARegressor()
        mod.fit(exog, endog_uni)
        actual = getattr(mod, attr)
        np.testing.assert_allclose(actual, expected)

    def test_arma_fit(self, exog, endog_uni):
        mod = ARMARegressor()
        mod.fit(exog, endog_uni)
        actual = mod.arma_.fittedvalues
        expected = VALID['arma']['arma_fit']
        np.testing.assert_allclose(actual, expected)

    def test_predict(self, exog, endog_uni):
        mod = ARMARegressor()
        mod.fit(exog, endog_uni)
        actual = mod.predict(exog)
        np.testing.assert_allclose(actual, VALID['arma']['fit'])


class TestTsaRegressor:
    def test_fit_no_kwargs(self, exog, endog_uni):
        mod = TSARegressor(model=ExponentialSmoothing)
        mod.fit(exog, endog_uni)
        actual = mod.fitted_values_
        expected = VALID['tsa']['fit']
        np.testing.assert_allclose(actual, expected)

    def test_fit_kwargs(self, exog, endog_uni):
        mod = TSARegressor(
            model=ExponentialSmoothing,
            damped=True,
            trend='mul'
        )
        mod.fit(exog, endog_uni)
        actual = mod.fitted_values_
        expected = VALID['tsa']['with_kwargs']
        np.testing.assert_allclose(actual, expected)

    def test_with_exog(self, exog, endog_uni):
        mod = TSARegressor(
            model=SARIMAX,
            use_exog=True,
            order=(1, 0, 0),
            trend='c'
        )
        mod.fit(exog, endog_uni)
        actual = mod.fitted_values_
        expected = VALID['tsa']['with_exog']
        np.testing.assert_allclose(actual, expected)

    def test_predict(self, exog, endog_uni):
        mod = TSARegressor(model=ExponentialSmoothing)
        mod.fit(exog, endog_uni)
        actual = mod.predict(exog)
        expected = VALID['tsa']['forecast']
        np.testing.assert_allclose(actual, expected)

    def test_summary(self, exog, endog_uni):
        mod = TSARegressor(model=ExponentialSmoothing)
        mod.fit(exog, endog_uni)
        assert isinstance(mod.summary_, Summary)
