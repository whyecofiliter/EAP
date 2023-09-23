## fama_macbeth

### Factor mimicking portfolio

#### factor: Size(SMB) and Value (HML)

Following Fama and French (1993), the SMB and HML portfolio and corresponding risk premium are calculated through the class **Factor_mimicking_portfolio.** The detail are introduced in EAP.fama_macbeth.Factor_mimicking_portfolio.

The data are collected from CSMAR dataset, by which SMB and HML in China stock market is constituted. **WARNING: Do Not use dataset in this demo for any commercial purpose.** 

```python
import sys,os
sys.path.append(os.path.abspath(".."))

# %% import data
# Monthly return of stocks in China security market
import pandas as pd

month_return = pd.read_hdf('.\data\month_return.h5', key='month_return')
company_data = pd.read_hdf('.\data\last_filter_pe.h5', key='data')
trade_data = pd.read_hdf('.\data\mean_filter_trade.h5', key='data')
```

Data need some preprocessing.

```python
# %% data preprocessing
# select the A share stock
month_return = month_return[month_return['Markettype'].isin([1, 4, 16])]

# % distinguish the stocks whose size is among the up 30% stocks in each month
def percentile(stocks) :
    return stocks >= stocks.quantile(q=.3)

month_return['cap'] = month_return.groupby(['Trdmnt'])['Msmvttl'].apply(percentile)
```

Construct proxy variable

```python
# %% Construct proxy variable
import numpy as np

# RMW
# in this demo, the ROE(TTM) are used
# ROE(TTM) = PBV1B/PE(TTM) 
company_data['ROE(TTM)'] = company_data['PBV1B']/company_data['PE1TTM']

# CMA
# % calculate the total asset
# asset = debt + equity
# debt = company_value - market_value
# equity = market_value / PB
company_data['debt'] = company_data['EV1'] - company_data['MarketValue']
company_data['equity'] = company_data['MarketValue']/company_data['PBV1A']
company_data['asset'] = company_data['debt'] + company_data['equity']
# asset growth rate
company_data['asset_growth_rate'] = company_data['asset'].groupby(['Symbol']).diff(12)/company_data['asset']

# Momentum
month_return['rolling_12'] = np.array(month_return.groupby(['Stkcd'])['Mretwd'].rolling(12).sum())
month_return['momentum'] = month_return['rolling_12'] - month_return['Mretwd']

# Turnover  
trade_data['rolling_Turnover'] = np.array(trade_data['Turnover'].groupby('Symbol').rolling(12).mean())
trade_data['specific_Turnover'] = trade_data['Turnover'] / trade_data['rolling_Turnover']
```

Some further data preprocessing.

```python
# %% merge data
from pandas.tseries.offsets import *

month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
#month_return['Date_merge'] += MonthEnd()

company_data['Stkcd_merge'] = company_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
company_data['Date_merge'] = pd.to_datetime(company_data['TradingDate'])
company_data['Date_merge'] += MonthBegin()

trade_data['Stkcd_merge'] = trade_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
trade_data['TradingDate'] = trade_data.index.map(lambda x : x[1])
trade_data['Date_merge'] = pd.to_datetime(trade_data['TradingDate'])
#company_data['Yearmonth'] = company_data['Date_merge'].map(lambda x : 1000*x.year + x.month)
trade_data['Date_merge'] += MonthBegin()

# %% dataset starts from '2000-01'
company_data = company_data[company_data['Date_merge'] >= '2000-01']
month_return = month_return[month_return['Date_merge'] >= '2000-01']
return_company = pd.merge(company_data, month_return, on=['Stkcd_merge', 'Date_merge'])
return_company = pd.merge(return_company, trade_data, on=['Stkcd_merge', 'Date_merge'])
```

Construct factor mimicking portfolio and calculate factor risk premium.

```python
# %% SMB and HML
from fama_macbeth import Factor_mimicking_portfolio as fmp
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
# construct data for univariate analysis
test_data_1 = test_data_1[['Mretwd', 'Msmvttl', 'PE1A', 'Date_merge', 'Msmvttl']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '2000-01-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

# factor mimicking portfolio
fmp_1 = fmp(np.array(test_data_1))
SMB, HML = fmp_1.portfolio_return()
SMB = -SMB

plt.figure('SMB')
plt.plot(SMB, label='SMB')
plt.ylabel('risk premium')
plt.xlabel('time')
plt.legend()

plt.figure('HML')
plt.plot(HML, label='HML')
plt.ylabel('risk premium')
plt.xlabel('time')
plt.legend()
===========================================================
![SMB](G:\Python\EAP\demo\SMB.png)

```



#### factor: Profitability

The profitability factor (RMW) is constituted by the same procedure with HML, using proxy variable ROE(TTM).

```python
# Continue the previous code
# %% RMW
from fama_macbeth import Factor_mimicking_portfolio as fmp
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_2 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
# construct data for univariate analysis
test_data_2 = test_data_2[['Mretwd', 'Msmvttl', 'ROE(TTM)', 'Date_merge', 'Msmvttl']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2004-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]

# factor mimicking portfolio
fmp_2 = fmp(np.array(test_data_2))
Row_fac, RMW = fmp_2.portfolio_return()

plt.figure('RMW')
plt.plot(RMW, label='RMW')
plt.ylabel('risk premium')
plt.xlabel('time')
plt.legend()
===============================================================
![RMW](G:\Python\EAP\demo\RMW.png)
```



#### factor: Investment

The investment factor (CMA) is constituted by the same procedure with HML, using proxy variable, asset_growth_rate.

```python
# %% CMA
from fama_macbeth import Factor_mimicking_portfolio as fmp
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_3 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
# construct data for univariate analysis
test_data_3 = test_data_3[['Mretwd', 'Msmvttl', 'asset_growth_rate', 'Date_merge', 'Msmvttl']].dropna()
test_data_3 = test_data_3[(test_data_3['Date_merge'] >= '2000-01-01') & (test_data_3['Date_merge'] <= '2019-12-01')]

# factor mimicking portfolio
fmp_3 = fmp(np.array(test_data_3))
Row_fac, CMA = fmp_3.portfolio_return()

plt.figure('CMA')
plt.plot(CMA, label='CMA')
plt.ylabel('risk premium')
plt.xlabel('time')
plt.legend()
==============================================================
![CMA](G:\Python\EAP\demo\CMA.png)
```



#### factor: Momentum

The momentum factor (MOM) is constituted by the same procedure with HML, using proxy variable, sum of past 12 month return.

```python
# %% Momentum
from fama_macbeth import Factor_mimicking_portfolio as fmp
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_4 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
# construct data for univariate analysis
test_data_4 = test_data_4[['Mretwd', 'Msmvttl', 'momentum', 'Date_merge', 'Msmvttl']].dropna()
test_data_4 = test_data_4[(test_data_4['Date_merge'] >= '2000-01-01') & (test_data_4['Date_merge'] <= '2019-12-01')]

# factor mimicking portfolio
fmp_4 = fmp(np.array(test_data_4))
Row_fac, MOM = fmp_4.portfolio_return()

plt.figure('MOM')
plt.plot(MOM, label='MOM')
plt.ylabel('risk premium')
plt.xlabel('time')
plt.legend()
===================================================
![MOM](G:\Python\EAP\demo\MOM.png)
```



#### factor: Turnover

The turnover factor (Turn) is constituted by the same procedure with HML, using proxy variable, abnormal turnover rate.

```python
# %% Turnover
from fama_macbeth import Factor_mimicking_portfolio as fmp
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_5 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
# construct data for univariate analysis
test_data_5 = test_data_5[['Mretwd', 'Msmvttl', 'specific_Turnover', 'Date_merge', 'Msmvttl']].dropna()
test_data_5 = test_data_5[(test_data_5['Date_merge'] >= '2000-01-01') & (test_data_5['Date_merge'] <= '2019-12-01')]

# factor mimicking portfolio
fmp_5 = fmp(np.array(test_data_5))
Row_fac, Turn = fmp_5.portfolio_return()

plt.figure('Turnover')
plt.plot(Turn, label='Turnover')
plt.ylabel('risk premium')
plt.xlabel('time')
plt.legend()
=======================================================================
![Turn](G:\Python\EAP\demo\Turn.png)
```



### Fama_macbeth_regress

Fama Macbeth regression tests existence of factor risk premium. Fama-Macbeth Regression follows two steps:

1.  Specify the model and take cross-sectional regression.
2.  Take the time-series average of regression coefficient

For more details, please read Empirical Asset Pricing: The Cross Section of Stock Returns. Bali, Engle, Murray, 2016.

Testing whether characteristics have systematic dynamics to asset return needs adding characteristics of stocks into FM regression model. In this demo, characteristics or factors include size, value, profitability, investment, momentum, and turnover, whose proxy variables are introduced in demo, Factor mimicking portfolio. **WARNING: Do Not use dataset in this demo for any commercial purpose.** 

```python
# %% set system path
import sys,os
sys.path.append(os.path.abspath(".."))
```

Data need some preprocessing.

```python
# %% import data
# Monthly return of stocks in China security market
import pandas as pd

month_return = pd.read_hdf('.\data\month_return.h5', key='month_return')
company_data = pd.read_hdf('.\data\last_filter_pe.h5', key='data')
trade_data = pd.read_hdf('.\data\mean_filter_trade.h5', key='data')
beta = pd.read_hdf('.\\data\\beta.h5', key='data')

# %% data preprocessing
# forward the monthly return for each stock
# emrwd is the return including dividend
month_return['emrwd'] = month_return.groupby(['Stkcd'])['Mretwd'].shift(-1)
# emrnd is the return including no dividend
month_return['emrnd'] = month_return.groupby(['Stkcd'])['Mretnd'].shift(-1)
# select the A share stock
month_return = month_return[month_return['Markettype'].isin([1, 4, 16])]

# % distinguish the stocks whose size is among the up 30% stocks in each month
def percentile(stocks) :
    return stocks >= stocks.quantile(q=.3)

month_return['cap'] = month_return.groupby(['Trdmnt'])['Msmvttl'].apply(percentile)
```

Construct proxy variables for factors.

```python
# %% Construct proxy variable
import numpy as np

# SMB
# log(Size)
month_return['Size'] = np.log(month_return['Msmvttl'])

# HML
company_data['BM'] = 1 / company_data['PBV1A']

# RMW
# in this demo, the ROE(TTM) are used
# ROE(TTM) = PBV1B/PE(TTM) 
company_data['ROE(TTM)'] = company_data['PBV1B']/company_data['PE1TTM']

# CMA
# % calculate the total asset
# asset = debt + equity
# debt = company_value - market_value
# equity = market_value / PB
company_data['debt'] = company_data['EV1'] - company_data['MarketValue']
company_data['equity'] = company_data['MarketValue']/company_data['PBV1A']
company_data['asset'] = company_data['debt'] + company_data['equity']
# asset growth rate
company_data['asset_growth_rate'] = company_data['asset'].groupby(['Symbol']).diff(12)/company_data['asset']

# Momentum
month_return['rolling_12'] = np.array(month_return.groupby(['Stkcd'])['Mretwd'].rolling(12).sum())
month_return['momentum'] = month_return['rolling_12'] - month_return['Mretwd']

# Turnover  
trade_data['rolling_Turnover'] = np.array(trade_data['Turnover'].groupby('Symbol').rolling(12).mean())
trade_data['specific_Turnover'] = trade_data['Turnover'] / trade_data['rolling_Turnover']
```

Some further data preprocessing.

```python
# %% merge data
from pandas.tseries.offsets import *

month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
#month_return['Date_merge'] += MonthEnd()

company_data['Stkcd_merge'] = company_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
company_data['Date_merge'] = pd.to_datetime(company_data['TradingDate'])
company_data['Date_merge'] += MonthBegin()

trade_data['Stkcd_merge'] = trade_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
trade_data['TradingDate'] = trade_data.index.map(lambda x : x[1])
trade_data['Date_merge'] = pd.to_datetime(trade_data['TradingDate'])
#company_data['Yearmonth'] = company_data['Date_merge'].map(lambda x : 1000*x.year + x.month)
trade_data['Date_merge'] += MonthBegin()

# %% dataset starts from '2000-01'
company_data = company_data[company_data['Date_merge'] >= '2000-01']
month_return = month_return[month_return['Date_merge'] >= '2000-01']
return_company = pd.merge(company_data, month_return, on=['Stkcd_merge', 'Date_merge'])
return_company = pd.merge(return_company, trade_data, on=['Stkcd_merge', 'Date_merge'])

# beta
return_company = return_company.set_index(['Stkcd', 'Trdmnt'])
return_company = pd.merge(return_company, beta, left_index=True, right_index=True)
```

Conduct Fama-Macbeth regression.

```python
# %% Fama-Macbeth regression
# dataset : #1
# exclude tail stocks 
# range from 2000-01-01 ~ 2019-12-01 
from fama_macbeth import Fama_macbeth_regress

test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1 = test_data_1[['emrwd', 'beta', 'Size', 'BM', 'ROE(TTM)', 'asset_growth_rate', 'momentum', 'specific_Turnover', 'Date_merge']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '2000-01-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

model = Fama_macbeth_regress(test_data_1)
result = model.fit(add_constant=True)
model.summary_by_time()
model.summary()
=============================================================================================================================
para_average: [ 2.11712697e-16  7.76046176e-03 -2.11404539e-02  2.23096650e-02
  2.59978312e-02  4.93686033e-03  1.13644979e-02 -2.65276499e-02]
tvalue: [ 1.49510717  0.92445793 -2.36234542  2.69894942  4.34532686  1.31038888
  1.3544952  -4.77712106]
R: 0.09379246136154293
ADJ_R: 0.08697059984414297
sample number N: 1071.5882352941176
+---------------------+-----------------------------------------------------------+----------+--------------+---------------+
|         Year        |                           Param                           | R Square | Adj R Square | Sample Number |
+---------------------+-----------------------------------------------------------+----------+--------------+---------------+
| 2000-02-01 00:00:00 |   [0.000 -0.065 -0.212 0.119 0.022 -0.054 0.024 -0.099]   |   0.1    |     0.09     |      549      |
| 2000-03-01 00:00:00 |   [-0.000 -0.093 0.067 0.081 0.062 0.055 -0.083 -0.025]   |   0.04   |     0.03     |      535      |
| 2000-04-01 00:00:00 |   [0.000 -0.028 -0.084 0.199 -0.026 -0.007 0.044 -0.058]  |   0.04   |     0.03     |      542      |
| 2000-08-01 00:00:00 |   [0.000 -0.179 -0.118 -0.089 0.153 -0.071 0.164 -0.084]  |   0.1    |     0.09     |      508      |
| 2000-09-01 00:00:00 |   [0.000 -0.153 -0.129 0.057 -0.088 -0.105 -0.122 0.112]  |   0.15   |     0.14     |      506      |
| 2000-10-01 00:00:00 |   [0.000 0.051 -0.061 0.086 -0.013 -0.051 -0.058 0.003]   |   0.02   |     0.01     |      498      |
| 2001-02-01 00:00:00 |   [-0.000 0.117 -0.122 0.021 0.014 -0.021 -0.058 0.008]   |   0.03   |     0.01     |      598      |
| 2001-03-01 00:00:00 |   [-0.000 0.003 -0.063 -0.256 -0.095 0.054 0.053 -0.048]  |   0.08   |     0.07     |      598      |
| 2001-04-01 00:00:00 |  [-0.000 0.059 -0.174 -0.184 -0.151 -0.005 0.137 -0.016]  |   0.12   |     0.11     |      593      |
| 2001-08-01 00:00:00 |    [0.000 0.044 0.101 0.045 0.063 -0.018 -0.118 -0.141]   |   0.06   |     0.05     |      564      |
| 2001-09-01 00:00:00 |   [-0.000 -0.045 0.098 0.156 -0.076 -0.082 0.013 0.087]   |   0.06   |     0.05     |      565      |
| 2001-10-01 00:00:00 |   [0.000 0.194 -0.068 -0.006 -0.074 0.099 -0.244 -0.219]  |   0.16   |     0.15     |      565      |
| 2002-02-01 00:00:00 |   [0.000 0.115 -0.158 -0.028 -0.122 0.074 -0.108 -0.153]  |   0.13   |     0.12     |      622      |
| 2002-03-01 00:00:00 |   [0.000 0.149 -0.066 0.095 -0.060 -0.186 0.004 -0.069]   |   0.11   |     0.1      |      656      |
| 2002-04-01 00:00:00 |   [-0.000 -0.093 0.072 -0.054 -0.170 0.005 0.112 -0.059]  |   0.05   |     0.04     |      658      |
| 2002-08-01 00:00:00 |   [0.000 -0.191 0.017 -0.095 -0.062 0.070 0.128 -0.073]   |   0.08   |     0.07     |      611      |
| 2002-09-01 00:00:00 |    [0.000 -0.115 0.043 0.044 0.041 -0.002 0.097 -0.031]   |   0.03   |     0.02     |      616      |
| 2002-10-01 00:00:00 |    [-0.000 -0.163 0.192 0.001 0.038 0.073 0.149 -0.043]   |   0.13   |     0.12     |      616      |
| 2003-02-01 00:00:00 |    [-0.000 -0.086 0.174 0.169 0.231 0.033 0.210 0.039]    |   0.19   |     0.19     |      711      |
| 2003-03-01 00:00:00 |    [0.000 -0.027 0.124 0.215 0.214 0.013 0.183 -0.033]    |   0.16   |     0.15     |      718      |
| 2003-04-01 00:00:00 |     [-0.000 0.222 0.092 0.043 0.001 0.049 0.014 0.023]    |   0.06   |     0.06     |      726      |
| 2003-05-01 00:00:00 |   [-0.000 -0.146 0.126 0.058 0.075 -0.009 -0.050 -0.161]  |   0.05   |     0.04     |      676      |
| 2003-06-01 00:00:00 |     [0.000 0.121 0.183 0.084 0.173 0.039 0.146 -0.008]    |   0.14   |     0.13     |      677      |
| 2003-07-01 00:00:00 |   [-0.000 -0.081 -0.034 -0.031 0.012 0.016 -0.203 0.004]  |   0.04   |     0.03     |      677      |
| 2003-08-01 00:00:00 |    [0.000 -0.088 -0.030 0.069 0.047 -0.008 0.021 0.001]   |   0.01   |     0.0      |      710      |
| 2003-09-01 00:00:00 |    [-0.000 -0.059 0.152 0.297 0.270 0.059 0.242 -0.038]   |   0.28   |     0.27     |      716      |
| 2003-10-01 00:00:00 |    [0.000 0.179 0.103 0.163 0.009 -0.003 -0.102 0.028]    |   0.08   |     0.07     |      719      |
| 2003-11-01 00:00:00 |    [-0.000 -0.082 0.189 0.066 0.080 0.081 0.241 -0.008]   |   0.2    |     0.19     |      715      |
| 2003-12-01 00:00:00 |    [-0.000 0.372 -0.095 0.004 0.108 0.079 -0.065 0.073]   |   0.17   |     0.17     |      720      |
| 2004-01-01 00:00:00 |   [-0.000 0.058 -0.065 0.081 0.011 -0.059 -0.116 0.060]   |   0.06   |     0.05     |      720      |
| 2004-02-01 00:00:00 |    [-0.000 0.009 -0.034 0.099 0.121 0.042 -0.105 0.137]   |   0.05   |     0.04     |      770      |
| 2004-03-01 00:00:00 |   [-0.000 -0.049 -0.180 0.111 0.087 0.049 0.077 -0.110]   |   0.05   |     0.04     |      768      |
| 2004-04-01 00:00:00 |   [-0.000 0.108 0.007 -0.015 -0.100 -0.113 0.041 0.061]   |   0.05   |     0.05     |      775      |
| 2004-05-01 00:00:00 |    [-0.000 -0.164 0.050 0.114 0.183 0.007 0.076 -0.124]   |   0.12   |     0.11     |      747      |
| 2004-06-01 00:00:00 |    [0.000 0.057 -0.009 0.021 0.065 -0.010 0.268 0.064]    |   0.08   |     0.07     |      745      |
| 2004-07-01 00:00:00 |    [0.000 -0.040 0.062 0.160 0.063 0.011 -0.052 -0.163]   |   0.07   |     0.06     |      754      |
| 2004-08-01 00:00:00 |    [-0.000 0.145 -0.036 0.062 0.072 0.106 0.259 0.151]    |   0.14   |     0.13     |      767      |
| 2004-09-01 00:00:00 |   [0.000 -0.008 -0.035 -0.083 0.068 0.088 0.155 -0.036]   |   0.07   |     0.06     |      773      |
| 2004-10-01 00:00:00 |  [-0.000 0.120 -0.143 -0.071 -0.095 0.022 -0.250 -0.053]  |   0.13   |     0.13     |      774      |
| 2004-11-01 00:00:00 |   [-0.000 -0.139 -0.002 0.083 0.183 -0.013 0.183 -0.168]  |   0.1    |     0.09     |      778      |
| 2004-12-01 00:00:00 |    [-0.000 -0.045 0.038 0.095 0.141 -0.000 0.071 0.036]   |   0.03   |     0.02     |      784      |
| 2005-01-01 00:00:00 |    [0.000 0.221 -0.071 0.051 0.159 0.083 -0.107 0.066]    |   0.08   |     0.07     |      789      |
| 2005-02-01 00:00:00 |    [0.000 -0.093 0.046 0.056 0.181 -0.012 0.272 -0.122]   |   0.2    |     0.19     |      805      |
| 2005-03-01 00:00:00 |    [-0.000 -0.105 0.088 0.085 0.139 0.042 0.179 -0.032]   |   0.13   |     0.12     |      811      |
| 2005-04-01 00:00:00 |   [0.000 0.057 -0.112 0.043 -0.130 -0.089 -0.328 -0.105]  |   0.33   |     0.33     |      821      |
| 2005-05-01 00:00:00 |    [-0.000 -0.012 0.102 0.053 0.089 0.064 0.100 -0.037]   |   0.06   |     0.05     |      755      |
| 2005-06-01 00:00:00 |     [0.000 -0.029 0.228 0.057 0.132 0.006 0.204 0.069]    |   0.2    |     0.19     |      751      |
| 2005-07-01 00:00:00 |   [0.000 0.178 -0.179 -0.114 -0.067 -0.048 -0.266 0.044]  |   0.24   |     0.23     |      747      |
| 2005-08-01 00:00:00 |   [0.000 -0.085 -0.193 -0.183 0.016 0.092 -0.025 0.108]   |   0.08   |     0.08     |      790      |
| 2005-09-01 00:00:00 |   [0.000 0.111 -0.019 -0.064 -0.047 0.075 0.203 -0.133]   |   0.07   |     0.07     |      805      |
| 2005-10-01 00:00:00 |   [0.000 0.108 -0.016 0.146 -0.057 0.058 -0.089 -0.002]   |   0.08   |     0.07     |      708      |
| 2005-11-01 00:00:00 |    [0.000 0.114 0.201 0.105 0.212 -0.005 0.177 -0.111]    |   0.14   |     0.14     |      736      |
| 2005-12-01 00:00:00 |    [0.000 0.151 0.059 0.032 0.071 -0.027 -0.014 -0.025]   |   0.02   |     0.01     |      692      |
| 2006-01-01 00:00:00 |   [-0.000 -0.174 0.085 0.190 0.052 -0.050 -0.096 -0.074]  |   0.09   |     0.08     |      636      |
| 2006-02-01 00:00:00 |    [-0.000 0.079 -0.033 0.008 0.155 -0.022 0.276 0.066]   |   0.12   |     0.12     |      661      |
| 2006-03-01 00:00:00 |  [-0.000 0.068 -0.013 -0.099 -0.011 -0.058 0.236 -0.002]  |   0.08   |     0.07     |      654      |
| 2006-04-01 00:00:00 |  [-0.000 0.010 -0.066 -0.052 -0.193 -0.030 -0.099 -0.098] |   0.09   |     0.08     |      615      |
| 2006-05-01 00:00:00 |   [0.000 -0.154 -0.201 0.012 0.016 0.098 -0.018 -0.067]   |   0.06   |     0.05     |      583      |
| 2006-06-01 00:00:00 |   [0.000 -0.066 -0.149 0.016 0.020 -0.036 -0.015 0.002]   |   0.03   |     0.01     |      598      |
| 2006-07-01 00:00:00 |  [-0.000 -0.079 0.020 -0.022 -0.129 -0.022 0.180 -0.054]  |   0.04   |     0.03     |      596      |
| 2006-08-01 00:00:00 |   [-0.000 0.021 0.060 -0.224 -0.165 0.015 -0.106 -0.083]  |   0.05   |     0.04     |      639      |
| 2006-09-01 00:00:00 |    [0.000 -0.015 0.019 0.187 0.178 0.007 0.034 -0.089]    |   0.05   |     0.04     |      632      |
| 2006-10-01 00:00:00 |     [0.000 -0.168 0.384 0.081 0.120 0.030 0.208 0.089]    |   0.28   |     0.27     |      620      |
| 2006-11-01 00:00:00 |   [-0.000 -0.054 0.269 0.010 -0.008 -0.067 0.159 -0.001]  |   0.12   |     0.11     |      628      |
| 2006-12-01 00:00:00 |   [-0.000 0.112 -0.101 0.050 0.079 -0.024 -0.056 -0.033]  |   0.03   |     0.02     |      607      |
| 2007-01-01 00:00:00 |   [0.000 -0.019 -0.188 0.146 -0.037 -0.075 -0.005 0.029]  |   0.1    |     0.09     |      609      |
| 2007-02-01 00:00:00 |  [-0.000 -0.053 -0.214 0.140 -0.022 -0.003 -0.058 0.076]  |   0.13   |     0.12     |      685      |
| 2007-03-01 00:00:00 |    [-0.000 0.119 -0.022 0.192 0.000 -0.051 0.076 0.141]   |   0.12   |     0.11     |      698      |
| 2007-04-01 00:00:00 |    [0.000 -0.281 0.081 -0.036 -0.021 0.050 0.086 0.018]   |   0.1    |     0.09     |      722      |
| 2007-05-01 00:00:00 |   [-0.000 -0.070 0.081 -0.043 0.272 0.088 0.021 -0.213]   |   0.27   |     0.26     |      678      |
| 2007-06-01 00:00:00 |    [0.000 0.257 -0.156 0.048 0.031 0.116 0.061 -0.007]    |   0.12   |     0.11     |      694      |
| 2007-07-01 00:00:00 |    [-0.000 0.029 0.157 0.127 0.115 0.026 -0.040 -0.122]   |   0.07   |     0.06     |      711      |
| 2007-08-01 00:00:00 |    [-0.000 0.147 0.066 0.121 0.187 0.019 -0.040 -0.060]   |   0.05   |     0.04     |      715      |
| 2007-09-01 00:00:00 |    [0.000 -0.152 0.164 0.010 0.107 0.025 0.093 -0.015]    |   0.13   |     0.12     |      736      |
| 2007-10-01 00:00:00 |   [0.000 0.166 -0.131 -0.023 -0.136 -0.031 -0.437 0.006]  |   0.33   |     0.32     |      732      |
| 2007-11-01 00:00:00 |    [0.000 -0.006 -0.274 -0.083 0.069 0.002 0.030 0.096]   |   0.07   |     0.06     |      738      |
| 2007-12-01 00:00:00 |   [-0.000 -0.068 -0.223 0.038 0.045 0.110 -0.075 -0.093]  |   0.07   |     0.06     |      741      |
| 2008-01-01 00:00:00 |   [-0.000 0.055 -0.246 0.099 -0.033 -0.045 0.071 0.009]   |   0.12   |     0.11     |      743      |
| 2008-02-01 00:00:00 |   [0.000 -0.115 -0.022 0.119 -0.068 -0.042 0.010 -0.016]  |   0.03   |     0.02     |      799      |
| 2008-03-01 00:00:00 |     [-0.000 0.055 0.183 0.016 0.095 0.093 0.088 0.110]    |   0.1    |     0.1      |      816      |
| 2008-04-01 00:00:00 |   [0.000 0.005 -0.143 -0.098 -0.161 -0.056 0.066 -0.137]  |   0.08   |     0.07     |      813      |
| 2008-05-01 00:00:00 |  [-0.000 -0.184 0.110 -0.113 -0.012 -0.023 0.087 -0.060]  |   0.08   |     0.07     |      801      |
| 2008-06-01 00:00:00 |   [0.000 0.087 -0.324 -0.046 0.004 0.065 -0.160 -0.025]   |   0.13   |     0.12     |      812      |
| 2008-07-01 00:00:00 |   [-0.000 -0.142 0.160 0.097 0.049 -0.012 -0.172 -0.219]  |   0.16   |     0.15     |      818      |
| 2008-08-01 00:00:00 |   [0.000 0.074 0.205 0.057 -0.052 -0.004 -0.110 -0.011]   |   0.05   |     0.04     |      819      |
| 2008-09-01 00:00:00 |   [0.000 -0.183 -0.070 0.008 -0.078 -0.031 0.102 -0.198]  |   0.08   |     0.07     |      829      |
| 2008-10-01 00:00:00 |  [-0.000 0.153 -0.183 -0.043 -0.069 0.083 -0.141 -0.085]  |   0.1    |     0.09     |      836      |
| 2008-11-01 00:00:00 |   [0.000 -0.104 -0.286 -0.216 0.169 0.064 0.056 -0.156]   |   0.15   |     0.15     |      836      |
| 2008-12-01 00:00:00 |    [0.000 0.128 -0.112 0.054 0.043 0.046 -0.160 0.190]    |   0.13   |     0.12     |      834      |
| 2009-01-01 00:00:00 |   [0.000 -0.057 -0.125 0.052 0.004 -0.073 -0.235 -0.158]  |   0.1    |     0.09     |      844      |
| 2009-02-01 00:00:00 |   [0.000 0.209 -0.103 -0.005 -0.004 0.066 -0.117 0.043]   |   0.09   |     0.08     |      857      |
| 2009-03-01 00:00:00 |   [0.000 -0.045 -0.093 0.015 0.070 -0.023 -0.074 -0.012]  |   0.01   |     0.01     |      868      |
| 2009-04-01 00:00:00 |    [0.000 0.189 -0.008 0.003 -0.097 0.007 -0.144 0.063]   |   0.07   |     0.06     |      871      |
| 2009-05-01 00:00:00 |   [0.000 -0.025 0.253 0.124 0.099 -0.007 -0.010 -0.040]   |   0.1    |     0.1      |      817      |
| 2009-06-01 00:00:00 |   [-0.000 0.268 0.117 0.151 0.017 -0.016 -0.100 -0.077]   |   0.11   |     0.1      |      818      |
| 2009-07-01 00:00:00 |   [0.000 -0.309 -0.326 -0.104 0.079 0.045 -0.148 0.026]   |   0.22   |     0.22     |      811      |
| 2009-08-01 00:00:00 |    [0.000 0.011 0.027 -0.021 0.212 0.043 0.028 -0.014]    |   0.06   |     0.05     |      810      |
| 2009-09-01 00:00:00 |   [0.000 0.127 -0.125 -0.018 -0.006 -0.062 0.068 -0.044]  |   0.06   |     0.05     |      810      |
| 2009-10-01 00:00:00 |   [-0.000 -0.036 -0.283 0.044 0.006 0.037 -0.025 -0.102]  |   0.08   |     0.08     |      812      |
| 2009-11-01 00:00:00 |   [0.000 -0.065 -0.120 0.026 0.032 -0.001 -0.072 0.016]   |   0.03   |     0.02     |      824      |
| 2009-12-01 00:00:00 |  [0.000 -0.118 -0.318 -0.048 0.032 -0.108 -0.257 -0.002]  |   0.21   |     0.2      |      824      |
| 2010-01-01 00:00:00 |   [0.000 -0.018 -0.177 0.067 -0.087 -0.030 0.081 0.038]   |   0.07   |     0.06     |      834      |
| 2010-02-01 00:00:00 |    [0.000 0.203 -0.111 -0.043 0.005 0.039 0.052 -0.027]   |   0.06   |     0.06     |      984      |
| 2010-03-01 00:00:00 |  [-0.000 -0.214 -0.004 -0.120 0.028 -0.046 0.036 -0.205]  |   0.11   |     0.11     |      982      |
| 2010-04-01 00:00:00 |    [-0.000 -0.245 0.054 -0.106 0.035 0.019 0.111 0.022]   |   0.1    |     0.1      |      976      |
| 2010-05-01 00:00:00 |    [-0.000 0.033 0.018 0.006 0.076 0.100 -0.074 -0.066]   |   0.03   |     0.03     |      955      |
| 2010-06-01 00:00:00 |    [-0.000 0.252 -0.200 0.027 0.110 0.018 -0.157 0.115]   |   0.13   |     0.12     |      952      |
| 2010-07-01 00:00:00 |  [-0.000 -0.051 -0.183 -0.142 -0.027 -0.061 0.173 0.042]  |   0.16   |     0.15     |      957      |
| 2010-08-01 00:00:00 |  [-0.000 -0.115 0.012 -0.138 -0.018 -0.005 -0.024 -0.025] |   0.03   |     0.03     |      958      |
| 2010-09-01 00:00:00 |    [-0.000 0.245 0.211 -0.074 0.028 0.040 -0.126 0.083]   |   0.11   |     0.11     |      964      |
| 2010-10-01 00:00:00 |   [-0.000 -0.208 -0.202 -0.097 0.030 0.001 0.176 -0.108]  |   0.17   |     0.17     |      971      |
| 2010-11-01 00:00:00 |    [-0.000 0.146 0.060 -0.001 0.013 0.151 0.001 0.137]    |   0.08   |     0.08     |      978      |
| 2010-12-01 00:00:00 |    [0.000 0.060 0.039 0.209 0.038 -0.037 -0.164 -0.019]   |   0.12   |     0.12     |      984      |
| 2011-01-01 00:00:00 |   [0.000 0.235 -0.171 -0.075 -0.110 -0.031 0.162 0.035]   |   0.18   |     0.17     |      983      |
| 2011-02-01 00:00:00 |    [0.000 0.159 -0.064 0.182 0.075 0.003 -0.099 0.049]    |   0.08   |     0.07     |      1092     |
| 2011-03-01 00:00:00 |   [-0.000 -0.069 0.034 0.222 0.033 -0.036 0.005 -0.025]   |   0.06   |     0.05     |      1109     |
| 2011-04-01 00:00:00 |   [-0.000 -0.080 0.027 -0.095 0.014 0.005 -0.095 0.060]   |   0.02   |     0.01     |      1137     |
| 2011-05-01 00:00:00 |     [0.000 0.090 -0.056 0.026 0.039 0.077 0.257 0.004]    |   0.1    |     0.09     |      1128     |
| 2011-06-01 00:00:00 |  [-0.000 -0.029 -0.126 -0.236 -0.002 0.015 -0.048 -0.111] |   0.09   |     0.08     |      1131     |
| 2011-07-01 00:00:00 |   [0.000 -0.046 -0.104 -0.172 0.104 0.026 -0.108 -0.106]  |   0.07   |     0.06     |      1141     |
| 2011-08-01 00:00:00 |    [0.000 0.059 0.096 0.135 0.058 -0.137 -0.208 -0.107]   |   0.13   |     0.12     |      1186     |
| 2011-09-01 00:00:00 |    [0.000 -0.114 0.007 0.013 -0.070 0.054 0.053 0.073]    |   0.02   |     0.02     |      1200     |
| 2011-10-01 00:00:00 | [-0.000 -0.044 -0.092 -0.190 -0.036 -0.006 -0.000 -0.098] |   0.05   |     0.04     |      1191     |
| 2011-11-01 00:00:00 |    [0.000 -0.010 0.247 0.263 0.201 0.060 -0.017 0.022]    |   0.19   |     0.18     |      1230     |
| 2011-12-01 00:00:00 |    [0.000 0.307 0.198 0.279 0.131 -0.206 -0.059 -0.055]   |   0.18   |     0.18     |      1243     |
| 2012-01-01 00:00:00 |    [0.000 0.155 -0.238 -0.040 0.003 0.068 -0.028 0.063]   |   0.12   |     0.12     |      1242     |
| 2012-02-01 00:00:00 |   [0.000 -0.122 -0.116 -0.011 0.115 0.028 0.146 -0.155]   |   0.08   |     0.08     |      1314     |
| 2012-03-01 00:00:00 |   [-0.000 0.179 -0.019 0.191 0.087 -0.123 -0.118 -0.150]  |   0.1    |     0.09     |      1349     |
| 2012-04-01 00:00:00 |    [-0.000 0.026 -0.040 -0.092 0.034 0.058 0.087 0.021]   |   0.04   |     0.03     |      1369     |
| 2012-05-01 00:00:00 |   [-0.000 -0.153 -0.052 -0.053 0.011 0.078 0.159 -0.075]  |   0.07   |     0.06     |      1321     |
| 2012-06-01 00:00:00 |    [0.000 -0.022 0.121 0.109 0.059 0.042 0.202 -0.082]    |   0.08   |     0.07     |      1327     |
| 2012-07-01 00:00:00 |  [0.000 -0.013 -0.265 -0.130 -0.103 0.040 -0.033 -0.035]  |   0.12   |     0.11     |      1334     |
| 2012-08-01 00:00:00 |   [0.000 0.064 0.196 -0.064 0.030 -0.018 -0.075 -0.037]   |   0.04   |     0.04     |      1334     |
| 2012-09-01 00:00:00 |    [0.000 -0.047 -0.141 0.194 0.052 -0.055 0.151 0.051]   |   0.05   |     0.05     |      1348     |
| 2012-10-01 00:00:00 |   [-0.000 -0.074 0.195 0.319 0.091 -0.030 -0.009 -0.076]  |   0.19   |     0.18     |      1338     |
| 2012-11-01 00:00:00 |     [0.000 0.128 0.046 0.003 0.023 0.075 0.067 0.025]     |   0.03   |     0.02     |      1331     |
| 2012-12-01 00:00:00 |  [-0.000 -0.079 -0.071 -0.078 -0.068 0.017 -0.032 -0.046] |   0.02   |     0.02     |      1322     |
| 2013-01-01 00:00:00 |  [-0.000 -0.033 -0.144 -0.140 -0.054 0.073 0.069 -0.065]  |   0.07   |     0.06     |      1320     |
| 2013-02-01 00:00:00 |   [-0.000 -0.185 -0.144 -0.068 0.033 0.003 0.153 -0.092]  |   0.09   |     0.08     |      1404     |
| 2013-03-01 00:00:00 |    [0.000 -0.003 -0.007 0.010 0.022 0.072 0.126 -0.083]   |   0.03   |     0.03     |      1414     |
| 2013-04-01 00:00:00 |    [0.000 0.094 -0.181 -0.169 -0.065 0.043 0.065 0.095]   |   0.12   |     0.12     |      1422     |
| 2013-05-01 00:00:00 |   [0.000 -0.069 0.008 -0.059 0.078 -0.001 0.176 -0.064]   |   0.06   |     0.06     |      1395     |
| 2013-06-01 00:00:00 |   [-0.000 0.020 -0.129 -0.121 -0.020 -0.029 0.131 0.101]  |   0.09   |     0.09     |      1378     |
| 2013-07-01 00:00:00 |   [0.000 -0.052 -0.136 0.191 0.033 0.026 -0.096 -0.037]   |   0.08   |     0.08     |      1380     |
| 2013-08-01 00:00:00 |    [0.000 -0.030 0.006 -0.065 -0.077 0.038 0.092 0.138]   |   0.05   |     0.05     |      1380     |
| 2013-09-01 00:00:00 |   [0.000 -0.085 -0.085 0.169 0.074 -0.044 -0.079 -0.120]  |   0.08   |     0.08     |      1374     |
| 2013-10-01 00:00:00 |   [0.000 0.095 -0.097 -0.063 -0.127 0.041 0.160 -0.035]   |   0.1    |     0.09     |      1358     |
| 2013-11-01 00:00:00 |  [0.000 -0.088 -0.117 -0.001 0.143 -0.064 -0.036 -0.072]  |   0.04   |     0.04     |      1380     |
| 2013-12-01 00:00:00 |   [0.000 0.034 -0.105 -0.068 -0.045 0.041 0.365 -0.072]   |   0.2    |     0.19     |      1367     |
| 2014-01-01 00:00:00 |  [0.000 0.042 -0.178 -0.008 -0.039 -0.062 -0.088 -0.112]  |   0.07   |     0.06     |      1371     |
| 2014-02-01 00:00:00 |   [0.000 -0.060 -0.104 0.238 0.100 0.004 -0.165 -0.076]   |   0.15   |     0.15     |      1415     |
| 2014-03-01 00:00:00 |    [0.000 -0.006 0.059 0.018 0.020 0.034 -0.025 -0.027]   |   0.01   |     0.01     |      1413     |
| 2014-04-01 00:00:00 |   [-0.000 0.053 -0.021 -0.077 -0.159 0.015 0.111 -0.089]  |   0.08   |     0.07     |      1371     |
| 2014-05-01 00:00:00 |  [-0.000 0.071 -0.060 -0.126 -0.038 -0.005 0.020 -0.100]  |   0.06   |     0.05     |      1320     |
| 2014-06-01 00:00:00 |   [0.000 -0.059 -0.061 0.149 0.050 -0.016 -0.207 0.074]   |   0.11   |     0.1      |      1299     |
| 2014-07-01 00:00:00 |   [0.000 -0.016 -0.132 -0.124 -0.135 0.014 0.096 -0.058]  |   0.08   |     0.08     |      1298     |
| 2014-08-01 00:00:00 |    [0.000 0.058 -0.270 0.109 -0.031 0.013 -0.018 0.034]   |   0.09   |     0.08     |      1264     |
| 2014-09-01 00:00:00 |   [-0.000 -0.009 -0.059 0.149 0.024 0.050 -0.024 0.006]   |   0.02   |     0.02     |      1241     |
| 2014-10-01 00:00:00 |    [0.000 -0.007 0.103 0.233 -0.074 0.001 -0.017 0.010]   |   0.09   |     0.08     |      1209     |
| 2014-11-01 00:00:00 |    [-0.000 0.019 0.333 0.355 0.049 0.025 -0.028 0.119]    |   0.37   |     0.37     |      1205     |
| 2014-12-01 00:00:00 |    [0.000 0.174 -0.059 -0.210 0.078 0.011 0.027 -0.166]   |   0.21   |     0.21     |      1198     |
| 2015-01-01 00:00:00 |    [0.000 0.164 0.030 -0.060 -0.016 0.008 0.077 -0.040]   |   0.04   |     0.03     |      1181     |
| 2015-02-01 00:00:00 |    [0.000 0.117 -0.211 -0.003 -0.016 0.054 0.041 0.049]   |   0.07   |     0.06     |      1210     |
| 2015-03-01 00:00:00 |   [-0.000 -0.016 -0.041 0.123 -0.019 -0.069 0.100 0.062]  |   0.03   |     0.03     |      1229     |
| 2015-04-01 00:00:00 |   [0.000 0.111 -0.245 -0.244 -0.013 0.016 -0.007 -0.047]  |   0.2    |     0.2      |      1175     |
| 2015-05-01 00:00:00 |   [-0.000 -0.315 -0.123 0.220 0.081 -0.038 -0.000 0.054]  |   0.2    |     0.2      |      1101     |
| 2015-06-01 00:00:00 |   [-0.000 0.281 0.222 -0.008 0.172 -0.046 -0.166 -0.079]  |   0.11   |     0.1      |      1081     |
| 2015-07-01 00:00:00 |  [-0.000 -0.066 -0.048 0.113 -0.012 -0.080 -0.112 0.087]  |   0.08   |     0.07     |      998      |
| 2015-08-01 00:00:00 |    [-0.000 0.298 0.005 -0.020 0.083 0.030 -0.076 0.004]   |   0.08   |     0.08     |      1007     |
| 2015-09-01 00:00:00 |   [0.000 0.227 -0.181 -0.184 -0.123 0.039 0.024 -0.031]   |   0.23   |     0.22     |      1007     |
| 2015-10-01 00:00:00 |    [0.000 0.203 -0.141 0.051 0.096 0.121 -0.029 0.035]    |   0.09   |     0.08     |      969      |
| 2015-11-01 00:00:00 |   [0.000 0.024 -0.157 0.017 0.088 -0.013 -0.065 -0.092]   |   0.03   |     0.02     |      961      |
| 2015-12-01 00:00:00 |    [-0.000 -0.245 0.126 0.153 0.055 0.016 0.026 -0.094]   |   0.21   |     0.21     |      989      |
| 2016-01-01 00:00:00 |   [0.000 -0.129 -0.090 0.088 -0.011 -0.039 0.036 -0.029]  |   0.03   |     0.02     |      995      |
| 2016-02-01 00:00:00 |   [-0.000 0.341 -0.078 -0.065 0.091 0.069 -0.142 0.137]   |   0.16   |     0.16     |      1037     |
| 2016-03-01 00:00:00 |   [-0.000 -0.119 -0.150 0.004 0.061 0.043 0.079 -0.130]   |   0.03   |     0.03     |      1068     |
| 2016-04-01 00:00:00 |   [0.000 0.130 0.091 -0.095 0.058 -0.011 -0.104 -0.077]   |   0.03   |     0.03     |      1076     |
| 2016-05-01 00:00:00 |    [-0.000 0.094 -0.166 -0.103 0.035 0.079 0.040 0.011]   |   0.1    |     0.09     |      1080     |
| 2016-06-01 00:00:00 |   [0.000 -0.276 -0.042 0.112 0.090 -0.104 -0.097 -0.101]  |   0.21   |     0.2      |      1088     |
| 2016-07-01 00:00:00 |   [-0.000 0.042 -0.104 0.087 -0.028 0.025 -0.044 -0.116]  |   0.03   |     0.03     |      1100     |
| 2016-08-01 00:00:00 |   [0.000 -0.115 -0.157 0.065 0.070 0.039 -0.030 -0.151]   |   0.05   |     0.05     |      1183     |
| 2016-09-01 00:00:00 |    [0.000 0.022 -0.046 0.069 -0.068 -0.028 0.113 0.099]   |   0.04   |     0.03     |      1213     |
| 2016-10-01 00:00:00 |   [-0.000 0.044 0.047 0.243 0.000 -0.024 -0.060 -0.061]   |   0.08   |     0.08     |      1250     |
| 2016-11-01 00:00:00 |   [-0.000 -0.151 -0.113 0.140 0.014 -0.066 0.067 -0.009]  |   0.06   |     0.05     |      1286     |
| 2016-12-01 00:00:00 |    [0.000 -0.044 0.110 0.260 0.067 -0.070 0.038 -0.034]   |   0.12   |     0.11     |      1324     |
| 2017-01-01 00:00:00 |     [0.000 0.070 -0.102 0.076 0.018 0.036 0.063 0.023]    |   0.02   |     0.01     |      1350     |
| 2017-02-01 00:00:00 |    [-0.000 -0.029 0.009 0.094 0.172 0.056 0.114 -0.046]   |   0.05   |     0.05     |      1470     |
| 2017-03-01 00:00:00 |    [0.000 -0.048 0.093 0.156 0.153 0.045 -0.037 0.026]    |   0.08   |     0.07     |      1497     |
| 2017-04-01 00:00:00 |   [0.000 -0.051 0.247 0.089 0.044 -0.018 -0.045 -0.073]   |   0.1    |     0.1      |      1527     |
| 2017-05-01 00:00:00 |    [-0.000 0.123 0.031 0.049 0.159 -0.056 0.177 -0.166]   |   0.09   |     0.09     |      1517     |
| 2017-06-01 00:00:00 |    [0.000 0.100 -0.013 0.307 0.101 -0.075 0.081 -0.110]   |   0.13   |     0.13     |      1556     |
| 2017-07-01 00:00:00 |   [0.000 -0.014 -0.001 -0.117 -0.092 0.026 -0.004 0.040]  |   0.02   |     0.01     |      1577     |
| 2017-08-01 00:00:00 |    [0.000 0.054 0.066 -0.148 0.015 0.075 -0.070 -0.067]   |   0.05   |     0.05     |      1579     |
| 2017-09-01 00:00:00 |    [0.000 -0.093 0.187 -0.041 0.130 0.058 0.093 -0.272]   |   0.17   |     0.16     |      1594     |
| 2017-10-01 00:00:00 |    [-0.000 0.017 0.143 0.149 0.077 -0.055 0.085 0.121]    |   0.09   |     0.09     |      1591     |
| 2017-11-01 00:00:00 |    [-0.000 -0.011 0.055 -0.030 0.068 0.035 0.138 0.038]   |   0.05   |     0.05     |      1643     |
| 2017-12-01 00:00:00 |    [0.000 -0.053 0.191 0.261 0.137 -0.033 0.055 -0.095]   |   0.16   |     0.16     |      1632     |
| 2018-01-01 00:00:00 |    [-0.000 0.091 -0.034 -0.122 0.030 0.073 0.145 0.045]   |   0.07   |     0.07     |      1648     |
| 2018-02-01 00:00:00 |   [-0.000 0.048 -0.048 -0.321 -0.122 0.068 -0.209 0.004]  |   0.18   |     0.17     |      1688     |
| 2018-03-01 00:00:00 |   [0.000 -0.013 0.019 -0.062 -0.024 -0.032 0.054 0.008]   |   0.01   |     0.0      |      1743     |
| 2018-04-01 00:00:00 |   [-0.000 -0.078 0.046 -0.058 0.066 -0.021 0.059 -0.088]  |   0.04   |     0.04     |      1759     |
| 2018-05-01 00:00:00 |     [0.000 0.005 0.060 0.126 0.156 -0.009 0.193 0.000]    |   0.09   |     0.09     |      1773     |
| 2018-06-01 00:00:00 |   [-0.000 0.017 -0.001 0.171 0.015 -0.098 -0.030 -0.090]  |   0.07   |     0.06     |      1737     |
| 2018-07-01 00:00:00 |   [-0.000 -0.090 0.011 0.101 -0.123 -0.060 0.113 -0.150]  |   0.06   |     0.06     |      1762     |
| 2018-08-01 00:00:00 |    [0.000 0.067 0.190 0.123 0.008 -0.095 -0.094 0.036]    |   0.07   |     0.06     |      1835     |
| 2018-09-01 00:00:00 |  [-0.000 -0.087 0.033 0.153 -0.035 -0.021 -0.007 -0.024]  |   0.05   |     0.04     |      1849     |
| 2018-10-01 00:00:00 |  [-0.000 0.210 -0.088 -0.040 -0.050 -0.012 -0.183 -0.046] |   0.12   |     0.12     |      1858     |
| 2018-11-01 00:00:00 |   [0.000 -0.055 -0.007 0.020 -0.009 -0.006 0.065 -0.081]  |   0.01   |     0.01     |      1900     |
| 2018-12-01 00:00:00 |    [0.000 -0.043 0.145 0.072 0.075 -0.087 0.021 -0.098]   |   0.07   |     0.07     |      1915     |
| 2019-01-01 00:00:00 |   [0.000 0.103 -0.063 -0.160 -0.125 0.008 -0.217 0.145]   |   0.12   |     0.12     |      1952     |
| 2019-02-01 00:00:00 |    [0.000 0.112 -0.122 0.006 0.022 -0.024 0.117 -0.010]   |   0.03   |     0.03     |      1947     |
| 2019-03-01 00:00:00 |    [0.000 -0.082 0.008 0.106 0.108 -0.018 0.073 0.016]    |   0.04   |     0.03     |      1979     |
| 2019-04-01 00:00:00 |  [-0.000 -0.050 -0.054 -0.103 -0.102 -0.041 0.031 -0.045] |   0.02   |     0.02     |      2018     |
| 2019-05-01 00:00:00 |    [0.000 -0.021 0.163 -0.004 0.019 0.022 0.027 -0.085]   |   0.04   |     0.04     |      2041     |
| 2019-06-01 00:00:00 |   [-0.000 -0.062 0.079 -0.086 0.016 0.043 0.007 -0.054]   |   0.02   |     0.02     |      2059     |
| 2019-07-01 00:00:00 |   [-0.000 -0.023 0.041 -0.245 -0.048 0.016 0.052 0.017]   |   0.07   |     0.06     |      2101     |
| 2019-08-01 00:00:00 |   [-0.000 0.072 -0.029 -0.064 0.058 0.042 -0.081 -0.009]  |   0.02   |     0.02     |      2096     |
| 2019-09-01 00:00:00 |    [0.000 0.037 0.055 -0.025 0.018 0.008 0.075 -0.117]    |   0.02   |     0.02     |      2108     |
| 2019-10-01 00:00:00 |    [0.000 -0.068 0.074 0.022 0.047 -0.018 -0.037 0.004]   |   0.02   |     0.02     |      2127     |
| 2019-11-01 00:00:00 |    [0.000 0.109 0.011 -0.060 -0.027 0.065 -0.069 0.077]   |   0.02   |     0.02     |      2118     |
| 2019-12-01 00:00:00 |   [-0.000 0.047 -0.040 -0.231 -0.035 0.051 0.048 0.071]   |   0.09   |     0.08     |      2116     |
+---------------------+-----------------------------------------------------------+----------+--------------+---------------+


+-----------+--------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+
| Intercept |  beta  |   Size  |   BM   | ROE(TTM) | asset_growth_rate | momentum | specific_Turnover | Average R | Average adj R | Average n |
+-----------+--------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+
|    0.0    | 0.0078 | -0.0211 | 0.0223 |  0.026   |       0.0049      |  0.0114  |      -0.0265      |   0.094   |     0.087     |  1071.59  |
|   1.495   | 0.924  |  -2.362 | 2.699  |  4.345   |        1.31       |  1.354   |       -4.777      |     -     |       -       |     -     |
+-----------+--------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+

```

The first dataset **excludes 30% tail **stocks from **2000-01-01 to 2019-12-01**. Factors including **size factor (Size), value factor (BM), profitability factor (ROE(TTM))** and **turnover factor (specific_Turnover)** have **significant** risk premium. Factors including **market, investment, and momentum factor** have **insignificant** risk premium.

```python
# %% Fama-Macbeth regression
# dataset : #2
# include tail stocks
# range from 2000-01 ~ 2019-12-01
from fama_macbeth import Fama_macbeth_regress

test_data_2 = return_company[(return_company['Ndaytrd']>=10)]
test_data_2 = test_data_2[['emrwd', 'beta', 'Msmvttl', 'PE1A', 'ROE(TTM)', 'asset_growth_rate', 'momentum', 'specific_Turnover', 'Date_merge']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2000-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]

model = Fama_macbeth_regress(test_data_2)
result = model.fit(add_constant=True)
print(result)
model.summary()
model.summary_by_time()
================================================================================================================================

+-----------+--------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+
| Intercept |  beta  |   Size  |   BM   | ROE(TTM) | asset_growth_rate | momentum | specific_Turnover | Average R | Average adj R | Average n |
+-----------+--------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+
|    0.0    | 0.0107 | -0.0409 | 0.0206 |  0.0224  |       0.006       |  0.0069  |       -0.027      |   0.088   |     0.083     |  1429.13  |
|   1.616   | 1.397  |  -4.114 | 2.845  |  4.346   |       1.644       |  0.875   |       -5.287      |     -     |       -       |     -     |
+-----------+--------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+

```

The second dataset **contains 30% tail **stocks from **2000-01-01 to 2019-12-01**. Factors including **size factor (Size), value factor (BM), profitability (ROE(TTM)), turnover factor (specific_Turnover)** have **significant** risk premium. Factors including **market, investment, and momentum factor** have **insignificant** risk premium.

```python
# %% Fama-Macbeth regression
# dataset : #3
# exclude tail stocks 
# range from 2000-01-01 ~ 2016-12-01 
from fama_macbeth import Fama_macbeth_regress

test_data_3 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_3 = test_data_3[['emrwd', 'beta', 'Size', 'BM', 'ROE(TTM)', 'asset_growth_rate', 'momentum', 'specific_Turnover', 'Date_merge']].dropna()
test_data_3 = test_data_3[(test_data_3['Date_merge'] >= '2000-01-01') & (test_data_3['Date_merge'] <= '2016-12-01')]

model = Fama_macbeth_regress(test_data_3)
result = model.fit(add_constant=True)
print(result)
model.summary()
model.summary_by_time()
================================================================================================================================
+-----------+--------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+
| Intercept |  beta  |   Size  |   BM   | ROE(TTM) | asset_growth_rate | momentum | specific_Turnover | Average R | Average adj R | Average n |
+-----------+--------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+
|    0.0    | 0.0076 | -0.0326 | 0.0258 |  0.0261  |       0.0062      |  0.0092  |       -0.026      |   0.099   |     0.092     |   930.59  |
|   0.632   | 0.778  |  -3.225 |  2.93  |   3.93   |       1.466       |  0.971   |       -4.276      |     -     |       -       |     -     |
+-----------+--------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+
```

The third dataset **excludes 30% tail **stocks from **2000-01-01 to 2016-12-01**. Factors including **size factor (Size), value factor (BM), profitability (ROE(TTM)), turnover factor (specific_Turnover)** have **significant** risk premium. Factors including **market, investment, and momentum factor** have **insignificant** risk premium.

```python
# %% Fama-Macbeth regression
# dataset : #4
# include tail stocks 
# range from 2000-01-01 ~ 2016-12-01 
from fama_macbeth import Fama_macbeth_regress

test_data_4 = return_company[(return_company['Ndaytrd']>=10)]
test_data_4 = test_data_4[['emrwd', 'beta', 'Size', 'BM', 'ROE(TTM)', 'asset_growth_rate', 'momentum', 'specific_Turnover', 'Date_merge']].dropna()
test_data_4 = test_data_4[(test_data_4['Date_merge'] >= '2000-01-01') & (test_data_4['Date_merge'] <= '2016-12-01')]

model = Fama_macbeth_regress(test_data_4)
result = model.fit(add_constant=True)
print(result)
model.summary()
model.summary_by_time()
================================================================================================================================
+-----------+-------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+
| Intercept |  beta |   Size  |   BM   | ROE(TTM) | asset_growth_rate | momentum | specific_Turnover | Average R | Average adj R | Average n |
+-----------+-------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+
|    0.0    |  0.01 | -0.0541 | 0.0235 |  0.0224  |       0.007       |  0.0053  |       -0.027      |   0.094   |     0.089     |  1237.96  |
|   1.223   | 1.125 |  -4.807 | 3.053  |   3.84   |       1.722       |  0.595   |       -4.857      |     -     |       -       |     -     |
+-----------+-------+---------+--------+----------+-------------------+----------+-------------------+-----------+---------------+-----------+
```

The fourth dataset **contains 30% tail **stocks from **2000-01-01 to 2016-12-01**. Factors including **size factor (Size), value factor (BM), profitability (ROE(TTM)), turnover factor (specific_Turnover)** have **significant** risk premium. Factors including **market, investment, and momentum factor** have **insignificant** risk premium.





### Factor risk premium

This demo constructs factor risk premium, including market system risk premium, SMB, HML, RMW, CMA. Following the convention of Fama-French(1993), the factor mimicking portfolio is constituted first, from which then the factor risk premium is calculated. Details are in the introduction of **class Factor_mimicking_portfolio.**    

In this demo, data are collected from CSMAR dataset. **WARNING: Do Not use dataset in this demo for any commercial purpose.** 

```python
# %% set system path
import sys,os

sys.path.append(os.path.abspath(".."))
```

Data preprocessing.

```python
# %% import data
import pandas as pd

month_return = pd.read_hdf('.\data\month_return.h5', key='month_return')
company_data = pd.read_hdf('.\data\last_filter_pe.h5', key='data')
trade_data = pd.read_hdf('.\data\mean_filter_trade.h5', key='data')
beta = pd.read_hdf('.\\data\\beta.h5', key='data')

# %% data preprocessing
# forward the monthly return for each stock
# emrwd is the return including dividend
month_return['emrwd'] = month_return.groupby(['Stkcd'])['Mretwd'].shift(-1)
# emrnd is the return including no dividend
month_return['emrnd'] = month_return.groupby(['Stkcd'])['Mretnd'].shift(-1)
# select the A share stock
month_return = month_return[month_return['Markettype'].isin([1, 4, 16])]

# % distinguish the stocks whose size is among the up 30% stocks in each month
def percentile(stocks) :
    return stocks >= stocks.quantile(q=.3)

month_return['cap'] = month_return.groupby(['Trdmnt'])['Msmvttl'].apply(percentile)
```

Construct proxy variable.

```python
# %% Construct proxy variable
import numpy as np

# SMB
# log(Size)
month_return['Size'] = np.log(month_return['Msmvttl'])

# HML
company_data['BM'] = 1 / company_data['PBV1A']

# RMW
# in this demo, the ROE(TTM) are used
# ROE(TTM) = PBV1B/PE(TTM) 
company_data['ROE(TTM)'] = company_data['PBV1B']/company_data['PE1TTM']

# CMA
# % calculate the total asset
# asset = debt + equity
# debt = company_value - market_value
# equity = market_value / PB
company_data['debt'] = company_data['EV1'] - company_data['MarketValue']
company_data['equity'] = company_data['MarketValue']/company_data['PBV1A']
company_data['asset'] = company_data['debt'] + company_data['equity']
# asset growth rate
company_data['asset_growth_rate'] = company_data['asset'].groupby(['Symbol']).diff(12)/company_data['asset']

# Momentum
month_return['rolling_12'] = np.array(month_return.groupby(['Stkcd'])['Mretwd'].rolling(12).sum())
month_return['momentum'] = month_return['rolling_12'] - month_return['Mretwd']

# Turnover  
trade_data['rolling_Turnover'] = np.array(trade_data['Turnover'].groupby('Symbol').rolling(12).mean())
trade_data['specific_Turnover'] = trade_data['Turnover'] / trade_data['rolling_Turnover']
```

Merge data.

```python
# %% merge data
from pandas.tseries.offsets import *

month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
#month_return['Date_merge'] += MonthEnd()

company_data['Stkcd_merge'] = company_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
company_data['Date_merge'] = pd.to_datetime(company_data['TradingDate'])
company_data['Date_merge'] += MonthBegin()

trade_data['Stkcd_merge'] = trade_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
trade_data['TradingDate'] = trade_data.index.map(lambda x : x[1])
trade_data['Date_merge'] = pd.to_datetime(trade_data['TradingDate'])
#company_data['Yearmonth'] = company_data['Date_merge'].map(lambda x : 1000*x.year + x.month)
trade_data['Date_merge'] += MonthBegin()

# dataset starts from '2000-01'
company_data = company_data[company_data['Date_merge'] >= '2000-01']
month_return = month_return[month_return['Date_merge'] >= '2000-01']
return_company = pd.merge(month_return, company_data, on=['Stkcd_merge', 'Date_merge'])
return_company = pd.merge(return_company, trade_data, on=['Stkcd_merge', 'Date_merge'])

# beta
return_company = return_company.set_index(['Stkcd', 'Trdmnt'])
return_company = pd.merge(return_company, beta, left_index=True, right_index=True)
```

Generate factor risk premium.

```python
# %% generate factor risk premium
from fama_macbeth import Factor_mimicking_portfolio
import numpy as np

# Size and Value factor risk premium
# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
size_bm = return_company[(return_company['Ndaytrd']>=10)]
size_bm = size_bm[['emrwd', 'Size', 'BM', 'Date_merge', 'Size']].dropna()
size_bm = size_bm[(size_bm['Date_merge'] >= '2000-01-01') & (size_bm['Date_merge'] <= '2019-12-01')]
# construct portfolio
size_bm_portfolio = Factor_mimicking_portfolio(np.array(size_bm))
CNSMB, CNHML = size_bm_portfolio.portfolio_return()
CNSMB = - CNSMB
CNSMB = CNSMB.rename('SMB')
CNHML = CNHML.rename('HML')

size_rmw = return_company[(return_company['Ndaytrd']>=10)]
size_rmw = size_rmw[['emrwd', 'Size', 'ROE(TTM)', 'Date_merge', 'Size']].dropna()
size_rmw = size_rmw[(size_rmw['Date_merge'] >= '2004-01-01') & (size_rmw['Date_merge'] <= '2019-12-01')]
# construct portoflio
size_rmw_portfolio = Factor_mimicking_portfolio(np.array(size_rmw))
CNrow, CNRMW = size_rmw_portfolio.portfolio_return()
CNRMW = CNRMW.rename('RMW')

size_cma = return_company[(return_company['Ndaytrd']>=10)]
size_cma = size_cma[['emrwd', 'Size', 'asset_growth_rate', 'Date_merge', 'Size']].dropna()
size_cma = size_cma[(size_cma['Date_merge'] >= '2000-01-01') & (size_cma['Date_merge'] <= '2019-12-01')]
# construct portoflio
size_cma_portfolio = Factor_mimicking_portfolio(np.array(size_cma))
CNrow, CNCMA = size_cma_portfolio.portfolio_return()
CNCMA = CNCMA.rename('CMA')

# generate market portoflio and market risk premium
from portfolio_analysis import Univariate
beta_portfolio = return_company[(return_company['Ndaytrd']>=10)]
beta_portfolio = beta_portfolio[['emrwd', 'Size', 'Date_merge']].dropna()
beta_portfolio = Univariate(np.array(beta_portfolio), number=0)

beta = beta_portfolio.average_by_time()
CNBETA = pd.Series(beta[0], index=np.unique(beta_portfolio.sample[:, 2]))
CNBETA = CNBETA.rename('BETA')

# %% merge data
risk_premium = pd.concat([CNBETA, CNSMB, CNHML, CNRMW, CNCMA], axis=1).shift(1)
```



### Anomaly Portfolio

#### PCF, Asset growth rate, and turnover rate

This demo exams whether classic asset pricing models can explain some anomaly portfolio return. Anomaly portfolios, including price cash flow, asset growth rate, and turnover rate, are constructed by univariate analysis. Precisely, stocks are grouped by characteristics or proxy variables into 10 groups, and the difference return between the head group and the tail group is taken as anomaly portfolio return. classic asset pricing models include Fama-French 3 factors model, Carhart 4 factors model, and Fama-French 5 factors model, whose factor risk premium is constructed through factor mimicking portfolio.

In this demo, Fama-French 3 factors model and Fama-French 5 factors are used. Data are collected from CSMAR dataset. **WARNING: Do Not use dataset in this demo for any commercial purpose.** 

```python
# %% set system path
import sys,os

sys.path.append(os.path.abspath(".."))

# %% import data
import pandas as pd

month_return = pd.read_hdf('.\\data\\month_return.h5', key='month_return')
company_data = pd.read_hdf('.\\data\\last_filter_pe.h5', key='data')
trade_data = pd.read_hdf('.\\data\\mean_filter_trade.h5', key='data')
beta = pd.read_hdf('.\\data\\beta.h5', key='data')
risk_premium = pd.read_hdf('.\\data\\risk_premium.h5', key='data')
```

Data preprocessing.

```python
# %% data preprocessing
# forward the monthly return for each stock
# emrwd is the return including dividend
month_return['emrwd'] = month_return.groupby(['Stkcd'])['Mretwd'].shift(-1)
# emrnd is the return including no dividend
month_return['emrnd'] = month_return.groupby(['Stkcd'])['Mretnd'].shift(-1)
# select the A share stock
month_return = month_return[month_return['Markettype'].isin([1, 4, 16])]

# % distinguish the stocks whose size is among the up 30% stocks in each month
def percentile(stocks) :
    return stocks >= stocks.quantile(q=.3)

month_return['cap'] = month_return.groupby(['Trdmnt'])['Msmvttl'].apply(percentile)
```

Construct proxy variable.

```python
# %% Construct proxy variable
import numpy as np

# CMA
# % calculate the total asset
# asset = debt + equity
# debt = company_value - market_value
# equity = market_value / PB
company_data['debt'] = company_data['EV1'] - company_data['MarketValue']
company_data['equity'] = company_data['MarketValue']/company_data['PBV1A']
company_data['asset'] = company_data['debt'] + company_data['equity']
# asset growth rate
company_data['asset_growth_rate'] = company_data['asset'].groupby(['Symbol']).diff(12)/company_data['asset']

# Turnover  
trade_data['rolling_Turnover'] = np.array(trade_data['Turnover'].groupby('Symbol').rolling(12).mean())
trade_data['specific_Turnover'] = trade_data['Turnover'] / trade_data['rolling_Turnover']
```

Merge data.

```python
# %% merge data
from pandas.tseries.offsets import *

month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
#month_return['Date_merge'] += MonthEnd()

company_data['Stkcd_merge'] = company_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
company_data['Date_merge'] = pd.to_datetime(company_data['TradingDate'])
company_data['Date_merge'] += MonthBegin()

trade_data['Stkcd_merge'] = trade_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
trade_data['TradingDate'] = trade_data.index.map(lambda x : x[1])
trade_data['Date_merge'] = pd.to_datetime(trade_data['TradingDate'])
#company_data['Yearmonth'] = company_data['Date_merge'].map(lambda x : 1000*x.year + x.month)
trade_data['Date_merge'] += MonthBegin()

# dataset starts from '2000-01'
company_data = company_data[company_data['Date_merge'] >= '2000-01']
month_return = month_return[month_return['Date_merge'] >= '2000-01']
return_company = pd.merge(month_return, company_data, on=['Stkcd_merge', 'Date_merge'])
return_company = pd.merge(return_company, trade_data, on=['Stkcd_merge', 'Date_merge'])

# beta
return_company = return_company.set_index(['Stkcd', 'Trdmnt'])
return_company = pd.merge(return_company, beta, left_index=True, right_index=True)
```

Construct anomaly portfolios and their return.

```python
# %% construct anomaly portfolio and return
from portfolio_analysis import Univariate

# PCF : Price cash flow ratio
pcf = return_company[(return_company['Ndaytrd']>=10)]
pcf = pcf[['emrwd', 'PCF1A', 'Date_merge']].dropna()
pcf = pcf[(pcf['Date_merge'] >= '2000-01-01') & (pcf['Date_merge'] <= '2019-12-01')]

model_pcf = Univariate(np.array(pcf), number=9)
ret_pcf = model_pcf.print_summary_by_time(export=True)[['Time', 'diff']]
ret_pcf.index = pd.to_datetime(ret_pcf['Time'])
ret_pcf = ret_pcf['diff'].shift(1)
ret_pcf = ret_pcf.rename('PCF')

# Investment: Asset growth rate
inv = return_company[(return_company['Ndaytrd']>=10)]
inv = inv[['emrwd', 'asset_growth_rate', 'Date_merge']].dropna()
inv = inv[(inv['Date_merge'] >= '2000-01-01') & (inv['Date_merge'] <= '2019-12-01')]

model_inv = Univariate(np.array(inv), number=9)
ret_inv = model_inv.print_summary_by_time(export=True)[['Time', 'diff']]
ret_inv.index = pd.to_datetime(ret_inv['Time'])
ret_inv = ret_inv['diff'].shift(1)
ret_inv = ret_inv.rename('INV')

# abnormal turnover rate (one month): abtr1mon
abtr1mon = return_company[(return_company['Ndaytrd']>=10)]
abtr1mon = abtr1mon[['emrwd', 'specific_Turnover', 'Date_merge']].dropna()
abtr1mon = abtr1mon[(abtr1mon['Date_merge'] >= '2000-01-01') & (abtr1mon['Date_merge'] <= '2019-12-01')]

model_abtr1mon = Univariate(np.array(abtr1mon), number=9)
ret_abtr1mon = model_abtr1mon.print_summary_by_time(export=True)[['Time', 'diff']]
ret_abtr1mon.index = pd.to_datetime(ret_abtr1mon['Time'])
ret_abtr1mon = ret_abtr1mon['diff'].shift(1)
ret_abtr1mon = ret_abtr1mon.rename('ABT')

# %% merge data
data = pd.concat([ret_pcf, ret_inv, ret_abtr1mon, risk_premium], axis=1)
data = data['2004':'2019'].dropna()
```

Fama-French 3 factors model without Newey-West adjustment. Using time series regression, all alphas in regression are significant at 0.05, meaning that the anomaly portfolio return cannot be completely explained by Fama-French 3 factors model. The GRS test presents the same result. 

```python
# %% Fama-French 3 factors model
# Without Newey-West adjustment
# import data type: Dataframe

list_data = data.iloc[:, :3]
factor = data.iloc[:, 3:6]
model = TS_regress(list_y=list_data, factor=np.array(factor))
model.fit(newey_west=False)
model.summary()
=====================================================================
+----------+---------+---------+---------+---------+
| Variable |  alpha  |   BETA  |   SMB   |   HML   |
+----------+---------+---------+---------+---------+
|   PCF    | -0.0096 |  0.0892 |  0.9127 | -0.7753 |
| t-value  |  -5.891 |  5.363  |  12.393 | -16.443 |
| p-value  |   0.0   |   0.0   |   0.0   |   0.0   |
|   INV    |  0.0086 | -0.0293 | -0.9563 | -0.7849 |
| t-value  |  4.074  |  -1.352 |  -9.972 | -12.784 |
| p-value  |   0.0   |  0.178  |   0.0   |   0.0   |
|   ABT    | -0.0065 |  0.1271 |  -0.067 |  0.3228 |
| t-value  |  -2.305 |  4.415  |  -0.526 |  3.955  |
| p-value  |  0.022  |   0.0   |  0.599  |   0.0   |
+----------+---------+---------+---------+---------+
----------------------------------- GRS Test --------------------------------

GRS Statistics: 17.2 GRS p_value: 0.0
-----------------------------------------------------------------------------
```

Fama-French 3 factors model with Newey-West adjustment. Using time series regression, all alphas in regression are significant at 0.05, meaning that the anomaly portfolio return cannot be completely explained by Fama-French 3 factors model. The GRS test presents the same result. 

```python
# %% Fama-French 3 factors model
# With Newey-West adjustment
# import data type: Dataframe
model = TS_regress(list_y=data.iloc[:, :3], factor=data.iloc[:, 3:6])
model.fit(newey_west=True)
model.summary()
=====================================================================
+----------+---------+---------+---------+---------+
| Variable |  alpha  |   BETA  |   SMB   |   HML   |
+----------+---------+---------+---------+---------+
|   PCF    | -0.0096 |  0.0892 |  0.9127 | -0.7753 |
| t-value  |  -4.92  |  6.101  |  8.435  |  -11.03 |
| p-value  |   0.0   |   0.0   |   0.0   |   0.0   |
|   INV    |  0.0086 | -0.0293 | -0.9563 | -0.7849 |
| t-value  |  5.113  |  -1.016 |  -8.562 |  -8.948 |
| p-value  |   0.0   |  0.155  |   0.0   |   0.0   |
|   ABT    | -0.0065 |  0.1271 |  -0.067 |  0.3228 |
| t-value  |  -2.15  |  3.257  |  -0.294 |  1.916  |
| p-value  |  0.016  |  0.001  |  0.385  |  0.028  |
+----------+---------+---------+---------+---------+
----------------------------------- GRS Test --------------------------------

GRS Statistics: 17.2 GRS p_value: 0.0
-----------------------------------------------------------------------------
```

Fama-French 5 factors model with Newey-West adjustment. Using time series regression, all alphas in regression are insignificant, meaning that the anomaly portfolio return can be largely explained by Fama-French 5 factors model. The GRS test presents the same result. 

```python
# %% Fama-French 5 factors model
# With Newey-West adjustment
# import data type: Dataframe
model = TS_regress(list_y=data.iloc[:, :3], factor=data.iloc[:, 3:])
model.fit(newey_west=True)
model.summary()
=====================================================================
+----------+---------+---------+---------+---------+---------+--------+
| Variable |  alpha  |   BETA  |   SMB   |   HML   |   RMW   |  CMA   |
+----------+---------+---------+---------+---------+---------+--------+
|   PCF    | -0.0031 |  0.028  |  0.5858 | -0.9971 | -0.7922 | 0.2925 |
| t-value  |  -1.486 |  1.767  |  5.303  | -12.079 |  -7.24  | 1.975  |
| p-value  |   0.07  |  0.039  |   0.0   |   0.0   |   0.0   | 0.025  |
|   INV    |  0.0008 | -0.0569 |  -0.453 | -0.2155 | -0.1186 | 1.6106 |
| t-value  |  0.442  |  -2.118 |  -4.511 |  -2.78  |  -0.808 |  5.81  |
| p-value  |  0.329  |  0.018  |   0.0   |  0.003  |   0.21  |  0.0   |
|   ABT    | -0.0037 |  0.0932 | -0.2032 |  0.2461 | -0.4225 | 0.2642 |
| t-value  |  -1.223 |  2.401  |  -1.031 |  1.377  |  -1.227 | 0.784  |
| p-value  |  0.112  |  0.009  |  0.152  |  0.085  |  0.111  | 0.217  |
+----------+---------+---------+---------+---------+---------+--------+
----------------------------------- GRS Test --------------------------------

GRS Statistics: 1.477 GRS p_value: 0.222
-----------------------------------------------------------------------------
```



