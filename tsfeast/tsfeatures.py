"""Time series features module."""
from typing import List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from statsmodels.tsa.tsatools import add_trend

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
)


class TimeSeriesFeatures(BaseTransformer):
    """Generate multiple time series feature in one transformer."""

    def __init__(
            self, datetime: str, trend: str = 'n', lags: Optional[int] = None,
            rolling: Optional[List[int]] = None, ewma: Optional[List[int]] = None,
            pct_chg: Optional[List[int]] = None, diffs: Optional[int] = None,
            polynomial: Optional[int] = None, interactions: bool = True,
            fillna: bool = True
    ):
        """Instanatiate transformer object."""
        super().__init__()
        self.datetime = datetime
        self.trend = trend
        self.lags = lags
        self.rolling = rolling
        self.ewma = ewma
        self.pct_chg = pct_chg
        self.diffs = diffs
        self.polynomial = polynomial
        self.interactions = interactions
        self.fillna = fillna

    def _transform(self, X, y=None):
        """Fit transformer to data."""
        if not hasattr(self, 'freq_'):
            self.freq_ = pd.infer_freq(pd.DatetimeIndex(pd.to_datetime(X[self.datetime])))

        transforms = {
            'lags': LagFeatures(self.lags),
            'rolling': RollingFeatures(self.rolling),
            'ewma': EwmaFeatures(self.ewma),
            'pct_chg': ChangeFeatures(self.pct_chg),
            'diffs': DifferenceFeatures(self.diffs),
            'polynomial': PolyFeatures(self.polynomial),
            'interactions': InteractionFeatures()
        }
        # don't want `_` attributes from BaseTransformer.fit() method
        self.steps_ = [k for k, v in vars(self).items() if '_' not in k and v]
        numeric = X.select_dtypes('number').columns
        try:
            union = FeatureUnion([(k, v) for k, v in transforms.items() if k in self.steps_])
            transformer = ColumnTransformer([
                ('original', OriginalFeatures(), numeric),
                ('datetime', DateTimeFeatures(freq=self.freq_), self.datetime),
                ('features', union, numeric)
            ])
        except ValueError:
            transformer = ColumnTransformer([
                ('original', OriginalFeatures(), numeric),
                ('datetime', DateTimeFeatures(freq=self.freq_), self.datetime)
            ])

        features = pd.DataFrame(
            transformer.fit_transform(X, y), columns=transformer.get_feature_names()
        )
        if self.trend:
            features = add_trend(features, trend=self.trend, prepend=True, has_constant='add')
        if self.fillna:
            return features.fillna(0)
        return features
