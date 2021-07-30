import pathlib

import numpy as np
import pandas as pd
import pytest

HERE = pathlib.Path(__file__).parent
N = 12
IDX = pd.date_range(start='2020-01', periods=N, freq='M')
y = pd.Series(np.arange(100, 100+N), index=IDX, name='series-0')
Y = pd.DataFrame(np.arange(100, 100+(N*3)).reshape(N, 3), index=IDX, columns=[f'series-{x}' for x in range(3)])
X = pd.DataFrame(np.arange(10, 10+(N*5)).reshape(N, 5), index=IDX, columns=[f'feature-{x}' for x in range(5)])
dates = [x.strftime('%Y-%m-%d') for x in IDX]


@pytest.fixture
def curr_dir():
    return HERE


@pytest.fixture
def endog_uni():
    return y


@pytest.fixture
def endog_multi():
    return Y


@pytest.fixture
def exog():
    return X


@pytest.fixture
def date_col():
    return dates


# @pytest.fixture
# def tsfe_uni():
#     return TSFeatureExtractor(y)
#
#
# @pytest.fixture
# def tsfe_multi():
#     return TSFeatureExtractor(X)
