**This package is designed for empirical asset pricing (EAP).**

# Quick Start

## Portfolio Analysis

Portfolio Analysis is a common method in asset pricing. For more details, please read Bail et al. Empirical Asset Pricing. 

To do univariate analysis and bivariate analysis, the sample need specified structure. 

### Univariate Analysis

Univariate Analysis is dividing stocks by the sort variable in ascending order. If the return between the highest group and the lowest group has statistical significance, then we say the sort variable provides excess return.  

The basic analysis programs are listed below, where the stocks are divided into 10 groups (the param 'number' is the number of breakpoints used to divide the stocks). 

The Vanilla sample for univariate analysis requires three columns, 

1. The first column is the return series.

2. The second column is the sort variable (like beta, size, BM).

3. The third column is the time index.

```python
# Univariate Analysis
from EAP.portfolio_analysis import Univariate as uni

# instantiate the model
model = uni(sample)
# fit the model
model.fit(number=9)
# print the summary
model.summary()
```



### Bivariate Analysis

Bivariate analysis is the similar approach like univariate analysis. The difference is bivariate analysis has two sort variables.

The Vanilla sample for bivariate analysis requires four columns, 

1. The first column is the return series.

2. The second column is the row sort variable (like beta, size, BM).
3. The third column is the column sort variable (like beta, size, BM).

4. The fourth column is the time index.

```python
# Bivariate Analysis
from EAP.portfolio_analysis import Bivariate as bi

# instantiate the model
model = bi(sample)
# fit the model
model.fit(number=4)
# print the summary
model.summary()
```



### Construct Sample

We show you how to construct basic dataset by tushare, an open source data API for China stock market. 

```python
import tushare as ts
import time

pro = ts.pro_api()
# get stock code
data_info = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

# get monthly data
num = 0
df = list()
for i in data_info['ts_code']:
    num = num + 1
    df.append(pro.monthly(ts_code=i, start_date='20160101', end_date='20201231', fields='ts_code,trade_date,open,high,low,close,vol,amount'))
    # the ask cannot exceed 120 per minute
    if num % 110 == 0:
        time.sleep(60)
```





## Fama-Macbeth Regression

Fama-Macbeth regression is the common method in asset pricing. The method is to test the significance of factors' risk premium. 

The Vanilla sample for  Fama-Macbeth regression requires has the structure, 

1. The first column is the return series.

2. The second column to the last two column is the factors or characteristics of stocks.

3. The last column is the time index.

```python
from EAP.fama_macbeth import Fama_macbeth_regress as fmr

model = fmr(data)
model.fit(add_constant=True)
model.summary()
```



## Factor Mimicking Portfolio

Some factors are not tradeable like small minus big (SMB), high minus low (HML), and up minus down (UMD). To construct these factors' return (the return is also risk premium), we need techniques by Fama and French in their 1993 paper. Precisely, they use bivariate analysis to divide the stocks into a 2*3 matrix. The average difference between the highest and the lowest row groups is the row factor return (in their paper the SMB). By similar procedure, the col factor return is created. 

The data structure is the same with Fama-Macbeth regression.

```python
from EAP.fama_macbeth import Factor_mimicking_portfolio as fmp

model = fmp(data)
fac_row, fac_col = model.portfolio_return()
```



## Time Series Regression

Time series regression is used for estimating the beta of the stock and testing the explanation power of factor models. We put some anomaly series data and some competitive factor models in the regression to compare the explanation power of these models to the anomalies. 

The data is DataFrame of anomaly return series. The factor is the DataFrame of factors return series. 

If you want to compare the explanation power of the competitive models, you should run the regression based on different factor return series corresponding to the factor models.

```python
from time_series_regress import TS_regress

model = TS_regress(data, factor)
model.fit()
model.summary()
```



## Cross Section Regression

Cross section regression is used for estimating the risk premium of the factors and testing the explanation power of competitive factor models. We put some asset return or portfolio return series data and some competitive factor models in the regression. 

The data is DataFrame of asset return or portfolio return series. The factor is the DataFrame of factors return series.

If you want to compare the explanation power of the competitive models, you should run the regression based on different factor return series corresponding to the factor models.

```python
from cross_section_regress import CS_regress

model = CS_regress(data, factor)
model.fit()
model.summary()
```
