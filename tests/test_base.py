import pytest

from tsfeast._base import BaseContainer


class TestBaseEstimator:
    def test_fit_not_implemented(self, exog, endog_uni):
        mod = BaseContainer()
        with pytest.raises(NotImplementedError):
            mod._fit(exog, endog_uni)

    def test_predict_not_implemented(self, exog, endog_uni):
        mod = BaseContainer()
        with pytest.raises(NotImplementedError):
            mod._predict(exog)
