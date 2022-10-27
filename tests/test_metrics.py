import pytest
from sklearn.linear_model import LinearRegression

from tsfeast.metrics import bic_score, bic_scorer


def test_bic_score():
    actual = bic_score(100, 12, 5)
    expected = 67.6865754807971
    assert round(actual, 4) == round(expected, 4)


def test_bic_scorer(monkeypatch, exog, endog_uni):
    mod = LinearRegression()
    monkeypatch.setattr(mod, 'predict', lambda *args: endog_uni+10)
    actual = bic_scorer(mod, exog, endog_uni)
    expected = 67.6865754807971
    assert round(actual, 4) == round(expected, 4)
