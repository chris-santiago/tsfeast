"""Time series feature generator functions."""
from typing import Dict, List, Optional, Union

import holidays
import numpy as np
import pandas as pd

from tsfeast.utils import to_list


def get_busdays_in_month(dt: pd.Timestamp) -> int:
    """
    Get the number of business days in a month period, using US holidays.

    Parameters
    ----------
    dt: pd.Timestamp
        Desired month.

    Returns
    -------
    int
        Number of business days in the month.
    """
    chooser: Dict[bool, pd.Timestamp] = {
        True: dt,
        False: dt - pd.tseries.offsets.MonthBegin()
    }
    month_begin = chooser[dt.is_month_start]
    month_end = dt + pd.tseries.offsets.MonthBegin(1)  # np.busday_count end date is exclusive
    us_holidays = list(holidays.US(years=dt.year).keys())
    return np.busday_count(month_begin.date(), month_end.date(), holidays=us_holidays)  # type: ignore #pylint: disable=line-too-long # noqa


def get_datetime_features(
        data: Union[pd.DataFrame, pd.Series], date_col: Optional[str] = None,
        dt_format: Optional[str] = None, freq: Optional[str] = None
) -> pd.DataFrame:
    """
    Get features based on datetime index, including year, month, week, weekday, quarter, days in
    month, business days in month and leap year.

    Parameters
    ----------
    data: pd.DataFrame, pd.Series
        Original data.
    date_col: Optional[str]
        Column name containing date/timestamp.
    dt_format: Optional[str]
        Date/timestamp format, e.g. `%Y-%m-%d` for `2020-01-31`.
    freq: Optional[str]
        Date frequency.

    Returns
    -------
    pd.DataFrame
        Date features.
    """
    if isinstance(data, pd.DataFrame):
        if date_col is None:
            raise ValueError('`date_col` cannot be none when passing a DataFrame.')
        dates = data[date_col]
    elif isinstance(data, pd.Series):
        dates = data
    else:
        raise ValueError('`data` must be a DataFrame or Series.')

    if not freq:
        freq = pd.infer_freq(pd.DatetimeIndex(pd.to_datetime(dates, format=dt_format)))

    X_dt = pd.DatetimeIndex(pd.to_datetime(dates, format=dt_format))  # enforce DatetimeIndex
    dt_features = pd.DataFrame()
    dt_features['year'] = X_dt.year  # pylint: disable=no-member
    dt_features['quarter'] = X_dt.quarter  # pylint: disable=no-member
    dt_features['month'] = X_dt.month  # pylint: disable=no-member
    if freq == 'D':
        dt_features['week'] = X_dt.week  # pylint: disable=no-member
        dt_features['weekday'] = X_dt.weekday  # pylint: disable=no-member
    if freq and 'M' in freq:
        dt_features['days_in_month'] = X_dt.days_in_month  # pylint: disable=no-member
        dt_features['bdays_in_month'] = pd.Series(X_dt).apply(get_busdays_in_month)
    dt_features['leap_year'] = X_dt.is_leap_year.astype(int)  # pylint: disable=no-member
    dt_features.index = data.index
    return dt_features


def get_lag_features(data: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """
    Get n-lagged features for data.

    Parameters
    ----------
    data: pd.DataFrame
        Original data.
    n_lags: int
        Number of lags to generate.

    Returns
    -------
    pd.DataFrame
        Lagged values of specified dataset.

    """
    lags = []
    for n in range(1, n_lags+1):
        df = data.copy().shift(n)
        df.columns = [f'{x}_lag_{n}' for x in data.columns]
        lags.append(df)
    return pd.concat(lags, axis=1)


def get_rolling_features(data: pd.DataFrame, window_lengths: List[int]) -> pd.DataFrame:
    """
    Get rolling metrics (mean, std, min, max) for each specified window length.

    Parameters
    ----------
    data: pd.DataFrame
        Original data.
    window_lengths: List[int]
        List of window lengths to generate.

    Returns
    -------
    pd.DataFrame
        Rolling mean, std, min and max for each specified window length.

    """
    window_lengths = to_list(window_lengths)
    df = data.copy()
    windows = []
    metrics = ['sum', 'mean', 'std', 'min', 'max']
    for win in window_lengths:
        for m in metrics:
            windows.append(
                pd.DataFrame(
                    df.rolling(win).agg(m).values,
                    columns=[f'{c}_{win}_pd_{m}' for c in df.columns],
                    index=df.index
                )
            )
    return pd.concat(windows, axis=1)


def get_ewma_features(data: pd.DataFrame, window_lengths: List[int]) -> pd.DataFrame:
    """
    Get an exponentially-weighted moving average for each specified window length.

    Parameters
    ----------
    data: pd.DataFrame
        Original data.
    window_lengths: List[int]
        List of window lengths to generate.

    Returns
    -------
    pd.DataFrame
        Exponentially-weighted moving average for each specified window length.

    """
    window_lengths = to_list(window_lengths)
    df = data.copy()
    windows = []
    for win in window_lengths:
        windows.append(
            pd.DataFrame(
                df.ewm(span=win, min_periods=win-1).mean().values,
                columns=[f'{c}_{win}_ewm_mean' for c in df.columns],
                index=df.index
            )
        )
    return pd.concat(windows, axis=1)


def get_change_features(data: pd.DataFrame, period_lengths: List[int]) -> pd.DataFrame:
    """
    Get percent change for all features for each specified period length.

    Parameters
    ----------
    data: pd.DataFrame
        Original data.
    period_lengths: List[int]
        A list of period lengths to generate.

    Returns
    -------
    pd.DataFrame
        Percent changes for all features.
    """
    period_lengths = to_list(period_lengths)
    df = data.copy()
    changes = []
    for period in period_lengths:
        changes.append(
            pd.DataFrame(
                df.pct_change(periods=period).values,
                columns=[f'{c}_{period}_pd_pct_chg' for c in df.columns],
                index=df.index
            )
        )
    return pd.concat(changes, axis=1)


def get_difference_features(data: pd.DataFrame, n_diffs: int) -> pd.DataFrame:
    """
    Get n differences for all features.

    Parameters
    ----------
    data: pd.DataFrame
        Original data.
    n_diffs: int
        Number of differences to return.

    Returns
    -------
    pd.DataFrame
        N-differenced data.
    """
    diffs = []
    for n in range(1, n_diffs + 1):
        df = data.copy().shift(n)
        df.columns = [f'{x}_diff_{n}' for x in data.columns]
        diffs.append(df)
    return pd.concat(diffs, axis=1)
