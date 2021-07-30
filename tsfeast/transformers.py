import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.validation import check_is_fitted

from tsfeast.funcs import *
from tsfeast.utils import array_to_dataframe


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X: pd.DataFrame, y=None):
        check_is_fitted(self)
        return self.output_features_

    def get_feature_names(self) -> List[str]:
        check_is_fitted(self)
        return list(self.feature_names_)

    def _fit(self, X: pd.DataFrame, y=None):
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y=None) -> "BaseTransformer":
        if isinstance(X, np.ndarray):
            X = array_to_dataframe(X)
        self.input_features_ = X
        self.n_features_in_ = X.shape[0]
        self.output_features_ = self._fit(X, y)
        self.feature_names_ = self.output_features_.columns
        return self


class OriginalFeatures(BaseTransformer):
    def __init__(self):
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None):
        return X


class Scaler(BaseTransformer):
    """Wraps StandardScaler to maintain column names."""
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def _fit(self, X: pd.DataFrame, y=None):
        return pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

    def inverse_transform(self, X: pd.DataFrame, copy=None):
        return pd.DataFrame(
            self.scaler.inverse_transform(X),
            columns=self.feature_names_,
            index=X.index
        )


class DateTimeFeatures(BaseTransformer):
    def __init__(self, date_col: str = None, dt_format: str = None):
        super().__init__()
        self.date_col = date_col
        self.dt_format = dt_format

    def _fit(self, X: pd.DataFrame, y=None):
        return get_datetime_features(X, self.date_col, dt_format=self.dt_format)


class LagFeatures(BaseTransformer):
    def __init__(self, n_lags: int):
        super().__init__()
        self.n_lags = n_lags

    def _fit(self, X: pd.DataFrame, y=None):
        return get_lag_features(X, n_lags=self.n_lags)


class RollingFeatures(BaseTransformer):
    def __init__(self, window_lengths: List[int]):
        super().__init__()
        self.window_lengths = window_lengths

    def _fit(self, X: pd.DataFrame, y=None):
        return get_rolling_features(X, window_lengths=self.window_lengths)


class EwmaFeatures(BaseTransformer):
    def __init__(self, window_lengths: List[int]):
        super().__init__()
        self.window_lengths = window_lengths

    def _fit(self, X: pd.DataFrame, y=None):
        return get_ewma_features(X, window_lengths=self.window_lengths)


class ChangeFeatures(BaseTransformer):
    def __init__(self, period_lengths: List[int]):
        super().__init__()
        self.period_lengths = period_lengths

    def _fit(self, X: pd.DataFrame, y=None):
        return get_change_features(X, period_lengths=self.period_lengths)


class DifferenceFeatures(BaseTransformer):
    def __init__(self, n_diffs: int):
        super().__init__()
        self.n_diffs = n_diffs

    def _fit(self, X: pd.DataFrame, y=None):
        return get_difference_features(X, n_diffs=self.n_diffs)


class PolyFeatures(BaseTransformer):
    """Extract polynomial features."""
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree

    def _fit(self, X: pd.DataFrame, y=None):
        poly = []
        df = X.copy()
        for i in range(2, self.degree+1):
            poly.append(
                pd.DataFrame(
                    df.values ** i,
                    columns=[f'{c}^{i}' for c in df.columns],
                    index=df.index
                )
            )
        return pd.concat(poly, axis=1)


class InteractionFeatures(BaseTransformer):
    """Wraps PolynomialFeatures to extract interactions and keep column names."""
    def __init__(self):
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None):
        transformer = PolynomialFeatures(interaction_only=True, include_bias=False)
        interactions = transformer.fit_transform(X.fillna(0))
        cols = [':'.join(x) for x in combinations(X.columns, r=2)]
        return pd.DataFrame(
            interactions[:, X.shape[1]:],  # drop original values
            columns=cols,
            index=X.index
        )
