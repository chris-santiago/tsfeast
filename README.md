# tsfeast
![build](https://travis-ci.com/chris-santiago/tsfeast.svg?branch=master)
[![codecov](https://codecov.io/gh/chris-santiago/tsfeast/branch/master/graph/badge.svg?token=MSO9ZBH6UD)](https://codecov.io/gh/chris-santiago/tsfeast)

A collection of Scikit-Learn compatible time series transformers and tools.

## Installation

Create a virtual environment and install:

### From PyPi

```bash
pip install tsfeast
```

### From this repo

```bash
pip install git+https://github.com/chris-santiago/tsfeast.git
```

## Use

### Preliminaries

This example shows both the use of individual transformers and the `TimeSeriesFeatures` convenience class that wraps multiple transformers. Both methods are compatible with Scikit-Learn `Pipeline` objects.


```python
import warnings
warnings.filterwarnings("ignore")  # ignore pandas concat warnings from statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.arima_process import arma_generate_sample
from steps.forward import ForwardSelector

from tsfeast.transformers import DateTimeFeatures, InteractionFeatures, LagFeatures
from tsfeast.tsfeatures import TimeSeriesFeatures
from tsfeast.funcs import get_datetime_features
from tsfeast.utils import plot_diag
from tsfeast.models import ARMARegressor
```


```python
def make_dummy_data(n=200):
    n_lags = 2
    coefs = {'ar': [1, -0.85], 'ma': [1, 0], 'trend': 3.2, 'bdays_in_month': 231, 'marketing': 0.0026}
    rng = np.random.default_rng(seed=42)
    
    sales = pd.DataFrame({
        'date': pd.date_range(end='2020-08-31', periods=n, freq='M'),
        'sales_base': rng.poisson(200, n),
        'sales_ar': arma_generate_sample(ar=coefs['ar'], ma=coefs['ma'], nsample=n, scale=100),
        'sales_trend': [x * coefs['trend'] + rng.poisson(300) for x in range(1, n+1)],
    })
    
    sales = sales.join(get_datetime_features(sales['date'])[['bdays_in_month', 'quarter']])
    sales['sales_per_day'] = sales['bdays_in_month'] * coefs['bdays_in_month'] + rng.poisson(100, n)
    
    sales['mkt_base'] = rng.normal(1e6, 1e4, n)
    sales['mkt_trend'] = np.array([x * 5e3 for x in range(1, n+1)]) + rng.poisson(100)
    sales['mkt_season'] = np.where(sales['quarter'] == 3, sales['mkt_base'] * .35, 0)
    sales['mkt_total'] = sales.loc[:, 'mkt_base': 'mkt_season'].sum(1) + rng.poisson(100, n)
    sales['sales_mkting'] = sales['mkt_total'].shift(n_lags) * coefs['marketing']
    
    final = pd.DataFrame({
        'y': sales[['sales_base', 'sales_ar', 'sales_trend', 'sales_per_day', 'sales_mkting']].sum(1).astype(int),
        'date': sales['date'],
        'marketing': sales['mkt_total'],
        'x2': rng.random(n),
        'x3': rng.normal(loc=320, scale=4, size=n)
    })
    return sales.iloc[2:, :], final.iloc[2:, :]
```


```python
def get_results(estimator, x_train, x_test, y_train, y_test):
    return pd.DataFrame(
        {
            'training': [
                mean_absolute_error(y_train, estimator.predict(x_train)), 
                mean_absolute_percentage_error(y_train, estimator.predict(x_train))
            ],
            'testing':  [
                mean_absolute_error(y_test, estimator.predict(x_test)), 
                mean_absolute_percentage_error(y_test, estimator.predict(x_test))
            ],
        },
        index = ['MAE', 'MAPE']
    )
```

### Example Data

The dummy dataset in this example includes trend, seasonal, autoregressive and other factor components. Below, we visualize the individual components (`comps`) and features of the dummy dataset `data`.


```python
comps, data = make_dummy_data()
```

#### Sales Components


```python
comps.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>sales_base</th>
      <th>sales_ar</th>
      <th>sales_trend</th>
      <th>bdays_in_month</th>
      <th>quarter</th>
      <th>sales_per_day</th>
      <th>mkt_base</th>
      <th>mkt_trend</th>
      <th>mkt_season</th>
      <th>mkt_total</th>
      <th>sales_mkting</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2004-03-31</td>
      <td>211</td>
      <td>153.620257</td>
      <td>285.6</td>
      <td>23</td>
      <td>1</td>
      <td>5402</td>
      <td>1.012456e+06</td>
      <td>15128.0</td>
      <td>0.000000</td>
      <td>1.027692e+06</td>
      <td>2584.285914</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004-04-30</td>
      <td>181</td>
      <td>18.958345</td>
      <td>300.8</td>
      <td>22</td>
      <td>2</td>
      <td>5180</td>
      <td>1.009596e+06</td>
      <td>20128.0</td>
      <td>0.000000</td>
      <td>1.029835e+06</td>
      <td>2661.116408</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004-05-31</td>
      <td>195</td>
      <td>54.420246</td>
      <td>312.0</td>
      <td>20</td>
      <td>2</td>
      <td>4726</td>
      <td>9.848525e+05</td>
      <td>25128.0</td>
      <td>0.000000</td>
      <td>1.010071e+06</td>
      <td>2672.000109</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2004-06-30</td>
      <td>206</td>
      <td>31.100042</td>
      <td>326.2</td>
      <td>22</td>
      <td>2</td>
      <td>5195</td>
      <td>1.008291e+06</td>
      <td>30128.0</td>
      <td>0.000000</td>
      <td>1.038529e+06</td>
      <td>2677.570754</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2004-07-31</td>
      <td>198</td>
      <td>34.283905</td>
      <td>317.4</td>
      <td>21</td>
      <td>3</td>
      <td>4952</td>
      <td>1.004049e+06</td>
      <td>35128.0</td>
      <td>351416.992807</td>
      <td>1.390691e+06</td>
      <td>2626.185776</td>
    </tr>
  </tbody>
</table>
</div>




```python
for col in comps.columns:
    print(f'Column: {col}')
    plt.figure(figsize=(10, 5))
    plt.plot(comps[col])
    plt.show()
```

    Column: date



    
![png](_static/output_8_1.png)
    


    Column: sales_base



    
![png](_static/output_8_3.png)
    


    Column: sales_ar



    
![png](_static/output_8_5.png)
    


    Column: sales_trend



    
![png](_static/output_8_7.png)
    


    Column: bdays_in_month



    
![png](_static/output_8_9.png)
    


    Column: quarter



    
![png](_static/output_8_11.png)
    


    Column: sales_per_day



    
![png](_static/output_8_13.png)
    


    Column: mkt_base



    
![png](_static/output_8_15.png)
    


    Column: mkt_trend



    
![png](_static/output_8_17.png)
    


    Column: mkt_season



    
![png](_static/output_8_19.png)
    


    Column: mkt_total



    
![png](_static/output_8_21.png)
    


    Column: sales_mkting



    
![png](_static/output_8_23.png)
    


#### Dummy Dataset


```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>date</th>
      <th>marketing</th>
      <th>x2</th>
      <th>x3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>8636</td>
      <td>2004-03-31</td>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8341</td>
      <td>2004-04-30</td>
      <td>1.029835e+06</td>
      <td>0.466509</td>
      <td>318.780107</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7959</td>
      <td>2004-05-31</td>
      <td>1.010071e+06</td>
      <td>0.361299</td>
      <td>324.917503</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8435</td>
      <td>2004-06-30</td>
      <td>1.038529e+06</td>
      <td>0.852623</td>
      <td>316.776026</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8127</td>
      <td>2004-07-31</td>
      <td>1.390691e+06</td>
      <td>0.571951</td>
      <td>314.425310</td>
    </tr>
  </tbody>
</table>
</div>




```python
for col in data.columns:
    print(f'Column: {col}')
    plt.figure(figsize=(10, 5))
    plt.plot(data[col])
    plt.show()
```

    Column: y



    
![png](_static/output_11_1.png)
    


    Column: date



    
![png](_static/output_11_3.png)
    


    Column: marketing



    
![png](_static/output_11_5.png)
    


    Column: x2



    
![png](_static/output_11_7.png)
    


    Column: x3



    
![png](_static/output_11_9.png)
    



```python
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
x_train, x_test = X.iloc[:-40, :], X.iloc[-40:, :]
y_train, y_test = y.iloc[:-40], y.iloc[-40:]
```

### Individual Transformers

`tsfeast` provides individual time series transformers that can be used by themselves or within Scikit-Learn `Pipeline` objects:

|Transformer|Parameters|Description|
|-----------|----------|-----------|
|`OriginalFeatures`|None|Passes original features through pipeline.|
|`Scaler`|None|Wraps Scikit-Learn `StandardScaler` to maintain DataFrame columns.|
|`DateTimeFeatures`|date_col: `str`, dt_format: `str`, freq: `str`|Generates datetime features from a given date column.|
|`LaggedFeatures`|n_lags: `int`, fillna: `bool`|Generate lag features.|
|`RollingFeatures`|window_lengths: `List[int]`, fillna: `bool`|Generate rolling features (mean, std, min, max) for each specified window length.|
|`EwmaFeatures`|window_lengths: `List[int]`, fillna: `bool`|Generate exponentially-weighted moving average for each specified window length.
|
|`ChangeFeatures`|period_lengths: `List[int]`, fillna: `bool`|Generate percent change for all features for each specified period length.|
|`DifferenceFeatures`|n_diffs: `int`, fillna: `bool`|Generate `n` differences for all features.|
|`PolyFeatures`|degree: `int`|Generate polynomial features.|
|`InteractionFeatures`|None|Wraps Scikit-Learn `PolynomialFeatures` to generate interaction features and maintain DataFrame columns.|


#### Notes on Pipeline Use

Behavior of Scikit-Learn `Pipeline` objects is appropriate and intended for **independent** data observations, but not necessarily appropriate for the **temporal dependencies** inherent in time series.

Scikit-Learn pipelines only call the `.transform()` method during the `.predict()` method, which is appropriate to prevent data leakage in predictions.  However, most of the transformers in this package take a set of features and generate new features; there's no inherent method to transform some time series features given a fitted estimator.

For time series lags, changes, etc., we have access to past data for feature generation without risk of data leakage; certain features (e.g. lags) require this to avoid NaNs or zeros. This behavior is appropriate for time series transformations, only.

#### Generate DateTime Features


```python
dt = DateTimeFeatures(date_col='date')
dt.fit_transform(X, y)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>quarter</th>
      <th>month</th>
      <th>days_in_month</th>
      <th>bdays_in_month</th>
      <th>leap_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2004</td>
      <td>1</td>
      <td>3</td>
      <td>31</td>
      <td>23</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004</td>
      <td>2</td>
      <td>4</td>
      <td>30</td>
      <td>22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2004</td>
      <td>2</td>
      <td>5</td>
      <td>31</td>
      <td>20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2004</td>
      <td>2</td>
      <td>6</td>
      <td>30</td>
      <td>22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2004</td>
      <td>3</td>
      <td>7</td>
      <td>31</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>2020</td>
      <td>2</td>
      <td>4</td>
      <td>30</td>
      <td>22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>196</th>
      <td>2020</td>
      <td>2</td>
      <td>5</td>
      <td>31</td>
      <td>20</td>
      <td>1</td>
    </tr>
    <tr>
      <th>197</th>
      <td>2020</td>
      <td>2</td>
      <td>6</td>
      <td>30</td>
      <td>22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>198</th>
      <td>2020</td>
      <td>3</td>
      <td>7</td>
      <td>31</td>
      <td>22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>199</th>
      <td>2020</td>
      <td>3</td>
      <td>8</td>
      <td>31</td>
      <td>21</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>198 rows × 6 columns</p>
</div>



#### Generate Interaction Features


```python
feat = LagFeatures(n_lags=4)
feat.fit_transform(X.iloc[:, 1:], y)  # skipping date column
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>marketing_lag_1</th>
      <th>x2_lag_1</th>
      <th>x3_lag_1</th>
      <th>marketing_lag_2</th>
      <th>x2_lag_2</th>
      <th>x3_lag_2</th>
      <th>marketing_lag_3</th>
      <th>x2_lag_3</th>
      <th>x3_lag_3</th>
      <th>marketing_lag_4</th>
      <th>x2_lag_4</th>
      <th>x3_lag_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.029835e+06</td>
      <td>0.466509</td>
      <td>318.780107</td>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.010071e+06</td>
      <td>0.361299</td>
      <td>324.917503</td>
      <td>1.029835e+06</td>
      <td>0.466509</td>
      <td>318.780107</td>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.038529e+06</td>
      <td>0.852623</td>
      <td>316.776026</td>
      <td>1.010071e+06</td>
      <td>0.361299</td>
      <td>324.917503</td>
      <td>1.029835e+06</td>
      <td>0.466509</td>
      <td>318.780107</td>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>1.971301e+06</td>
      <td>0.420222</td>
      <td>313.911203</td>
      <td>1.968782e+06</td>
      <td>0.648398</td>
      <td>327.288221</td>
      <td>1.973312e+06</td>
      <td>0.860346</td>
      <td>319.932653</td>
      <td>1.967943e+06</td>
      <td>0.216269</td>
      <td>317.692606</td>
    </tr>
    <tr>
      <th>196</th>
      <td>1.981624e+06</td>
      <td>0.188104</td>
      <td>324.110324</td>
      <td>1.971301e+06</td>
      <td>0.420222</td>
      <td>313.911203</td>
      <td>1.968782e+06</td>
      <td>0.648398</td>
      <td>327.288221</td>
      <td>1.973312e+06</td>
      <td>0.860346</td>
      <td>319.932653</td>
    </tr>
    <tr>
      <th>197</th>
      <td>1.977056e+06</td>
      <td>0.339024</td>
      <td>315.926738</td>
      <td>1.981624e+06</td>
      <td>0.188104</td>
      <td>324.110324</td>
      <td>1.971301e+06</td>
      <td>0.420222</td>
      <td>313.911203</td>
      <td>1.968782e+06</td>
      <td>0.648398</td>
      <td>327.288221</td>
    </tr>
    <tr>
      <th>198</th>
      <td>1.978757e+06</td>
      <td>0.703778</td>
      <td>320.409889</td>
      <td>1.977056e+06</td>
      <td>0.339024</td>
      <td>315.926738</td>
      <td>1.981624e+06</td>
      <td>0.188104</td>
      <td>324.110324</td>
      <td>1.971301e+06</td>
      <td>0.420222</td>
      <td>313.911203</td>
    </tr>
    <tr>
      <th>199</th>
      <td>2.332540e+06</td>
      <td>0.204360</td>
      <td>319.029524</td>
      <td>1.978757e+06</td>
      <td>0.703778</td>
      <td>320.409889</td>
      <td>1.977056e+06</td>
      <td>0.339024</td>
      <td>315.926738</td>
      <td>1.981624e+06</td>
      <td>0.188104</td>
      <td>324.110324</td>
    </tr>
  </tbody>
</table>
<p>198 rows × 12 columns</p>
</div>



### TimeSeriesFeatures Class

`tsfeast` also includes a `TimeSeriesFeatures` class that generates multiple time series features in one transformer. The only required parameter is the column of datetimes; the optional parameters control what additional transformers are included.

|Parameter|Type|Description|
|---------|----|-----------|
|datetime|str|Column that holds datetime information|
|trend|str|Trend to include, options are {'n': no trend, 'c': constant only, 't': linear trend, 'ct': constant and linear trend, 'ctt': constant, linear and quadratric trend}; defaults to no trend| 
|lags|int|Number of lags to include (optional).|
|rolling|List[int]|Number of rolling windows to include (optional).|
|ewma|List[int]|Number of ewma windows to include (optional).|
|pct_chg|List[int]|Periods to use for percent change features (optional).|
|diffs|int|Number of differences to include (optional).|
|polynomial|int|Polynomial(s) to include (optional).|
|interactions|bool|Whether to include interactions of original featutes; deault True.|
|fillna|bool|Whether to fill NaN values with zero; default True.|




```python
feat = TimeSeriesFeatures(
    datetime='date',
    trend='t',
    lags=4,
    interactions=False,
    polynomial=3
)
features = feat.fit_transform(X, y)
```


```python
features.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trend</th>
      <th>original__marketing</th>
      <th>original__x2</th>
      <th>original__x3</th>
      <th>datetime__year</th>
      <th>datetime__quarter</th>
      <th>datetime__month</th>
      <th>datetime__days_in_month</th>
      <th>datetime__bdays_in_month</th>
      <th>datetime__leap_year</th>
      <th>...</th>
      <th>features__lags__x3_lag_3</th>
      <th>features__lags__marketing_lag_4</th>
      <th>features__lags__x2_lag_4</th>
      <th>features__lags__x3_lag_4</th>
      <th>features__polynomial__marketing^2</th>
      <th>features__polynomial__x2^2</th>
      <th>features__polynomial__x3^2</th>
      <th>features__polynomial__marketing^3</th>
      <th>features__polynomial__x2^3</th>
      <th>features__polynomial__x3^3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
      <td>2004.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.056152e+12</td>
      <td>0.513733</td>
      <td>100102.615631</td>
      <td>1.085399e+18</td>
      <td>0.368219</td>
      <td>3.167146e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>1.029835e+06</td>
      <td>0.466509</td>
      <td>318.780107</td>
      <td>2004.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>30.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.060560e+12</td>
      <td>0.217631</td>
      <td>101620.756699</td>
      <td>1.092202e+18</td>
      <td>0.101527</td>
      <td>3.239468e+07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1.010071e+06</td>
      <td>0.361299</td>
      <td>324.917503</td>
      <td>2004.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>31.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.020244e+12</td>
      <td>0.130537</td>
      <td>105571.383672</td>
      <td>1.030520e+18</td>
      <td>0.047163</td>
      <td>3.430199e+07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1.038529e+06</td>
      <td>0.852623</td>
      <td>316.776026</td>
      <td>2004.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>30.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>316.389974</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.078543e+12</td>
      <td>0.726966</td>
      <td>100347.050373</td>
      <td>1.120098e+18</td>
      <td>0.619827</td>
      <td>3.178754e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>1.390691e+06</td>
      <td>0.571951</td>
      <td>314.425310</td>
      <td>2004.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>31.0</td>
      <td>21.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>318.780107</td>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
      <td>1.934020e+12</td>
      <td>0.327128</td>
      <td>98863.275608</td>
      <td>2.689624e+18</td>
      <td>0.187101</td>
      <td>3.108512e+07</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
[x for x in features.columns]
```




    ['trend',
     'original__marketing',
     'original__x2',
     'original__x3',
     'datetime__year',
     'datetime__quarter',
     'datetime__month',
     'datetime__days_in_month',
     'datetime__bdays_in_month',
     'datetime__leap_year',
     'features__lags__marketing_lag_1',
     'features__lags__x2_lag_1',
     'features__lags__x3_lag_1',
     'features__lags__marketing_lag_2',
     'features__lags__x2_lag_2',
     'features__lags__x3_lag_2',
     'features__lags__marketing_lag_3',
     'features__lags__x2_lag_3',
     'features__lags__x3_lag_3',
     'features__lags__marketing_lag_4',
     'features__lags__x2_lag_4',
     'features__lags__x3_lag_4',
     'features__polynomial__marketing^2',
     'features__polynomial__x2^2',
     'features__polynomial__x3^2',
     'features__polynomial__marketing^3',
     'features__polynomial__x2^3',
     'features__polynomial__x3^3']



### Pipeline Example

The `TimeSeriesFeatures` class can be used as a feature generation step within a Scikit-Learn `Pipeline`. Given the temporal nature of the data and models, this may not be appropriate for all use cases-- though the class remains *fully compatible* with `Pipeline` objects.

We'll instantiate a `TimeSeriesFeatures` object with a linear trend, four lags and no interactions.  Our pipeline will include feature generation, feature scaling and feature selection steps, before modeling with ordinary least squares. 

*Note: the `ForwardSelector` class is available in the `step-select` package (https://pypi.org/project/step-select/).*

The pipeline creates a total of 22 features, before selecting only four to use in the final model. Note that 3 of the 4 final features corresponed with features from our "true model" that created the dummy dataset ('trend', 'datetime__bdays_in_month' and 'marketing_lag_2').

Regression diagnostic plots show evidence of slightly non-normal residuals and (1) autoregressive term (again, as specified in the "true model").  We'll address the autoregressive term in the next example.


```python
feat = TimeSeriesFeatures(
    datetime='date',
    trend='t',
    lags=4,
    interactions=False
)

pl = Pipeline([
    ('feature_extraction', feat),
    ('scaler', StandardScaler()),
    ('feature_selection', ForwardSelector(metric='bic')),
    ('regression', LinearRegression())
])

pl.fit(x_train, y_train)
```




    Pipeline(steps=[('feature_extraction',
                     TimeSeriesFeatures(datetime='date', interactions=False, lags=4,
                                        trend='t')),
                    ('scaler', StandardScaler()),
                    ('feature_selection', ForwardSelector(metric='bic')),
                    ('regression', LinearRegression())])




```python
pl.named_steps.feature_extraction.output_features_
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trend</th>
      <th>original__marketing</th>
      <th>original__x2</th>
      <th>original__x3</th>
      <th>datetime__year</th>
      <th>datetime__quarter</th>
      <th>datetime__month</th>
      <th>datetime__days_in_month</th>
      <th>datetime__bdays_in_month</th>
      <th>datetime__leap_year</th>
      <th>...</th>
      <th>features__lags__x3_lag_1</th>
      <th>features__lags__marketing_lag_2</th>
      <th>features__lags__x2_lag_2</th>
      <th>features__lags__x3_lag_2</th>
      <th>features__lags__marketing_lag_3</th>
      <th>features__lags__x2_lag_3</th>
      <th>features__lags__x3_lag_3</th>
      <th>features__lags__marketing_lag_4</th>
      <th>features__lags__x2_lag_4</th>
      <th>features__lags__x3_lag_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
      <td>2004.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>1.029835e+06</td>
      <td>0.466509</td>
      <td>318.780107</td>
      <td>2004.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>30.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>316.389974</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1.010071e+06</td>
      <td>0.361299</td>
      <td>324.917503</td>
      <td>2004.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>31.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>318.780107</td>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1.038529e+06</td>
      <td>0.852623</td>
      <td>316.776026</td>
      <td>2004.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>30.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>324.917503</td>
      <td>1.029835e+06</td>
      <td>0.466509</td>
      <td>318.780107</td>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>1.390691e+06</td>
      <td>0.571951</td>
      <td>314.425310</td>
      <td>2004.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>31.0</td>
      <td>21.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>316.776026</td>
      <td>1.010071e+06</td>
      <td>0.361299</td>
      <td>324.917503</td>
      <td>1.029835e+06</td>
      <td>0.466509</td>
      <td>318.780107</td>
      <td>1.027692e+06</td>
      <td>0.716752</td>
      <td>316.389974</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>153</th>
      <td>154.0</td>
      <td>1.752743e+06</td>
      <td>0.060631</td>
      <td>322.823879</td>
      <td>2016.0</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>31.0</td>
      <td>21.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>312.156618</td>
      <td>1.750890e+06</td>
      <td>0.537173</td>
      <td>319.820019</td>
      <td>2.110972e+06</td>
      <td>0.368344</td>
      <td>324.492379</td>
      <td>2.127929e+06</td>
      <td>0.320161</td>
      <td>322.674221</td>
    </tr>
    <tr>
      <th>154</th>
      <td>155.0</td>
      <td>1.782890e+06</td>
      <td>0.368878</td>
      <td>313.360448</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>322.823879</td>
      <td>1.762560e+06</td>
      <td>0.296868</td>
      <td>312.156618</td>
      <td>1.750890e+06</td>
      <td>0.537173</td>
      <td>319.820019</td>
      <td>2.110972e+06</td>
      <td>0.368344</td>
      <td>324.492379</td>
    </tr>
    <tr>
      <th>155</th>
      <td>156.0</td>
      <td>1.788336e+06</td>
      <td>0.254549</td>
      <td>321.235197</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>28.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>313.360448</td>
      <td>1.752743e+06</td>
      <td>0.060631</td>
      <td>322.823879</td>
      <td>1.762560e+06</td>
      <td>0.296868</td>
      <td>312.156618</td>
      <td>1.750890e+06</td>
      <td>0.537173</td>
      <td>319.820019</td>
    </tr>
    <tr>
      <th>156</th>
      <td>157.0</td>
      <td>1.790967e+06</td>
      <td>0.385921</td>
      <td>316.450145</td>
      <td>2017.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>321.235197</td>
      <td>1.782890e+06</td>
      <td>0.368878</td>
      <td>313.360448</td>
      <td>1.752743e+06</td>
      <td>0.060631</td>
      <td>322.823879</td>
      <td>1.762560e+06</td>
      <td>0.296868</td>
      <td>312.156618</td>
    </tr>
    <tr>
      <th>157</th>
      <td>158.0</td>
      <td>1.811012e+06</td>
      <td>0.196960</td>
      <td>315.360643</td>
      <td>2017.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>30.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>316.450145</td>
      <td>1.788336e+06</td>
      <td>0.254549</td>
      <td>321.235197</td>
      <td>1.782890e+06</td>
      <td>0.368878</td>
      <td>313.360448</td>
      <td>1.752743e+06</td>
      <td>0.060631</td>
      <td>322.823879</td>
    </tr>
  </tbody>
</table>
<p>158 rows × 22 columns</p>
</div>




```python
new_features = pl.named_steps.feature_extraction.feature_names_
mask = pl.named_steps.feature_selection.get_support()
new_features[mask]
```




    Index(['trend', 'datetime__bdays_in_month', 'features__lags__marketing_lag_2',
           'features__lags__x3_lag_2'],
          dtype='object')




```python
get_results(pl, x_train, x_test, y_train, y_test)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>training</th>
      <th>testing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MAE</th>
      <td>373.819325</td>
      <td>201.999695</td>
    </tr>
    <tr>
      <th>MAPE</th>
      <td>0.040046</td>
      <td>0.017827</td>
    </tr>
  </tbody>
</table>
</div>




```python
resid = (y_train - pl.predict(x_train))
plot_diag(resid.iloc[2:])  # throw out first two residuals b/c of lags
```


    
![png](_static/output_26_0.png)
    


### ARMA Regressor

`tsfeast` includes a `models` module that provides an `ARMARegressor` class for extending Scikit-Learn regressors by adding support for AR/MA or ARIMA residuals. It accepts an arbitrary Scikit-Learn regressor and a tuple indicating the `(p,d,q)` order for the residuals model.

|Attribute|Description|
|---------|-----------|
|`estimator`|The Scikit-Learn regressor.|
|`order`|The (p,d,q,) order of the ARMA model.|
|`intercept_`|The fitted estimator's intercept.|
|`coef_`|The fitted estimator's coefficients.|
|`arma_`|The fitted ARMA model.|
|`fitted_values_`|The combined estimator and ARMA fitted values.|
|`resid_`|The combined estimator and ARMA residual values.|


**Note**
The `predict` method should not be used to get fitted values from the training set; rather, users should access this same data using the `fitted_values_` attribute.  The `predict` method calls the ARMA regresor's forecast method, which generates predictions from the last time step in the training data, thus would not match, temporally, in a `predict` call with training data.

The pipeline follows the same steps as the previous example, with the only change beging the regression model-- in this case, the `ARMARegressor`.  Metrics on test set slightly improve and we no longer see evidence of autoregressive term in the residuals.


```python
feat = TimeSeriesFeatures(
    datetime='date',
    trend='t',
    lags=4,
    interactions=False
)

mod = ARMARegressor(
    estimator=PoissonRegressor(),
    order=(1,0,0)
)

pl = Pipeline([
    ('feature_extraction', feat),
    ('scaler', StandardScaler()),
    ('feature_selection', ForwardSelector(metric='bic')),
    ('regression', mod)
])

pl.fit(x_train, y_train)
```




    Pipeline(steps=[('feature_extraction',
                     TimeSeriesFeatures(datetime='date', interactions=False, lags=4,
                                        trend='t')),
                    ('scaler', StandardScaler()),
                    ('feature_selection', ForwardSelector(metric='bic')),
                    ('regression', ARMARegressor(estimator=PoissonRegressor()))])




```python
new_features = pl.named_steps.feature_extraction.feature_names_
mask = pl.named_steps.feature_selection.get_support()
new_features[mask]
```




    Index(['trend', 'datetime__bdays_in_month', 'features__lags__marketing_lag_2',
           'features__lags__x3_lag_2'],
          dtype='object')




```python
get_results(pl, x_train, x_test, y_train, y_test)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>training</th>
      <th>testing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MAE</th>
      <td>409.572082</td>
      <td>143.269046</td>
    </tr>
    <tr>
      <th>MAPE</th>
      <td>0.043573</td>
      <td>0.012745</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_diag(pl.named_steps.regression.resid_)
```


    
![png](_static/output_31_0.png)
    


