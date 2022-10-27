"""Time series feature generators as Scikit-Learn compatible transformers."""

from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.validation import check_is_fitted

from tsfeast.funcs import (
    get_change_features,
    get_datetime_features,
    get_difference_features,
    get_ewma_features,
    get_lag_features,
    get_rolling_features,
)
from tsfeast.utils import Data, array_to_dataframe


class BaseTransformer(BaseEstimator, TransformerMixin):
    """Base transformer object."""

    def __init__(self, fillna: bool = True):
        """Instantiate transformer object."""
        self.fillna = fillna

    def transform(self, X: Data, y=None) -> Data:
        """
        Transform fitted data.

        Parameters
        ----------
        X: array of shape [n_samples, n_features]
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Array-like object of transformed data.

        Notes
        -----
        Scikit-Learn Pipelines only call the `.transform()` method during the `.predict()` method,
        which is appropriate to prevent data leakage in predictions.  However, most of the
        transformers in this module take a set of features and generate new features; there's no
        inherent method to transform some timeseries features given a fitted estimator.

        For time series lags, changes, etc., we have access to past data for feature
        generation without risk of data leakage; certain features (e.g. lags) require this
        to avoid NaNs or zeros.

        We append new X to our original features and transform on entire dataset, keeping
        only the last n rows.  Appropriate for time series transformations, only.
        """
        if isinstance(X, np.ndarray):
            X = array_to_dataframe(X)

        if hasattr(self, 'input_features_'):
            rows = X.shape[0]
            X = pd.concat([self.input_features_, X])  # pylint: disable=E0203
            self.output_features_ = self._transform(X, y).iloc[-rows:, :]
            if self.fillna:
                return self.output_features_.fillna(0)
            return self.output_features_
        self.input_features_: pd.DataFrame = X
        self.n_features_in_ = X.shape[0]
        self.output_features_ = self._transform(X, y)
        self.feature_names_ = self.output_features_.columns
        if self.fillna:
            return self.output_features_.fillna(0)
        return self.output_features_

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        check_is_fitted(self)
        return list(self.feature_names_)

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transform input data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        raise NotImplementedError

    def fit(self, X: Data, y=None) -> "BaseTransformer":
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: array of shape [n_samples, n_features]
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        BaseTransformer
            Self.
        """
        _, _ = X, y
        return self


class OriginalFeatures(BaseTransformer):
    """Return original features."""

    def _transform(self, X: pd.DataFrame, y=None) -> Data:
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        return X


class Scaler(BaseTransformer):
    """Wrap StandardScaler to maintain column names."""

    def __init__(self):
        """Instantiate transformer object."""
        super().__init__()
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None) -> "Scaler":
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        self.scaler.fit(X)
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self

    def transform(self, X: pd.DataFrame, y=None) -> Data:
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        self.feature_names_ = X.columns
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

    def inverse_transform(self, X: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Transform scaled data into original feature space.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        copy: bool
            Default True; if False, try to avoid a copy and do inplace scaling instead.

        Returns
        -------
        Data
            Data in original feature space.
        """
        return pd.DataFrame(
            self.scaler.inverse_transform(X, copy=copy),
            columns=self.feature_names_,
            index=X.index
        )


class DateTimeFeatures(BaseTransformer):
    """Generate datetime features."""

    def __init__(
            self, date_col: Optional[str] = None, dt_format: Optional[str] = None,
            freq: Optional[str] = None
    ):
        """
        Instantiate transformer object.

        date_col: Optional[str]
            Column name containing date/timestamp.
        dt_format: Optional[str]
            Date/timestamp format, e.g. `%Y-%m-%d` for `2020-01-31`.
        """
        super().__init__()
        self.date_col = date_col
        self.dt_format = dt_format
        self.freq = freq

    def fit(self, X: Data, y=None) -> "DateTimeFeatures":
        _ = y
        if isinstance(X, pd.DataFrame):
            dates = X[self.date_col]
        elif isinstance(X, pd.Series):
            dates = X
        else:
            raise ValueError('`data` must be a DataFrame or Series.')
        if not self.freq:
            self.freq = pd.infer_freq(
                pd.DatetimeIndex(pd.to_datetime(dates, format=self.dt_format))
            )
        return self

    def _transform(self, X: pd.DataFrame, y=None) -> Data:
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        return get_datetime_features(X, self.date_col, dt_format=self.dt_format, freq=self.freq)


class LagFeatures(BaseTransformer):
    """Generate lag features."""

    def __init__(self, n_lags: int, fillna: bool = True):
        """
        Instantiate transformer object.

        Parameters
        ----------
        n_lags: int
            Number of lags to generate.
        """
        super().__init__(fillna=fillna)
        self.n_lags = n_lags

    def _transform(self, X: pd.DataFrame, y=None) -> Data:
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        return get_lag_features(X, n_lags=self.n_lags)


class RollingFeatures(BaseTransformer):
    """Generate rolling features."""

    def __init__(self, window_lengths: List[int], fillna: bool = True):
        """
        Instantiate transformer object.

        Parameters
        ----------
        window_lengths: L:ist[int]
            Length of window(s) to create.
        """
        super().__init__(fillna=fillna)
        self.window_lengths = window_lengths

    def _transform(self, X: pd.DataFrame, y=None) -> Data:
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        return get_rolling_features(X, window_lengths=self.window_lengths)


class EwmaFeatures(BaseTransformer):
    """Generate exponentially-weighted moving-average features."""

    def __init__(self, window_lengths: List[int], fillna: bool = True):
        """
        Instantiate transformer object.

        Parameters
        ----------
        window_lengths: L:ist[int]
            Length of window(s) to create.
        """
        super().__init__(fillna=fillna)
        self.window_lengths = window_lengths

    def _transform(self, X: pd.DataFrame, y=None) -> Data:
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        return get_ewma_features(X, window_lengths=self.window_lengths)


class ChangeFeatures(BaseTransformer):
    """Generate period change features."""

    def __init__(self, period_lengths: List[int], fillna: bool = True):
        """
        Instantiate transformer object.

        Parameters
        ----------
        period_lengths: List[int]
            Length of period[s] to generate change features.
        """
        super().__init__(fillna=fillna)
        self.period_lengths = period_lengths

    def _transform(self, X: pd.DataFrame, y=None) -> Data:
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        return get_change_features(X, period_lengths=self.period_lengths)


class DifferenceFeatures(BaseTransformer):
    """Generate difference features."""

    def __init__(self, n_diffs: int, fillna: bool = True):
        """
        Instantiate transformer object.

        Parameters
        ----------
        n_diffs: int
            Number of differences to calculate.
        """
        super().__init__(fillna=fillna)
        self.n_diffs = n_diffs

    def _transform(self, X: pd.DataFrame, y=None) -> Data:
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        return get_difference_features(X, n_diffs=self.n_diffs)


class PolyFeatures(BaseTransformer):
    """Generate polynomial features."""

    def __init__(self, degree=2):
        """
        Instantiate transformer object.

        Parameters
        ----------
        degree: int
            Degree of polynomial to use.
        """
        super().__init__()
        self.degree = degree

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        poly = []
        df = X.copy()
        for i in range(2, self.degree + 1):
            poly.append(
                pd.DataFrame(
                    df.values ** i,
                    columns=[f'{c}^{i}' for c in df.columns],
                    index=df.index
                )
            )
        return pd.concat(poly, axis=1)


class InteractionFeatures(BaseTransformer):
    """Wrap PolynomialFeatures to extract interactions and keep column names."""

    def _transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit transformer object to data.

        Parameters
        ----------
        X: pd.DataFrame
            The input samples.
        y: None
            Not used; included for compatibility, only.

        Returns
        -------
        Data
            Transformed features.
        """
        transformer = PolynomialFeatures(interaction_only=True, include_bias=False)
        interactions = transformer.fit_transform(X.fillna(0))
        cols = [':'.join(x) for x in combinations(X.columns, r=2)]
        return pd.DataFrame(
            interactions[:, X.shape[1]:],  # drop original values
            columns=cols,
            index=X.index
        )
