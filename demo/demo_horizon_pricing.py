'''
    Horizon Pricing
'''

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

# %% add time
import datetime as dt
import numpy as np

# TimeStamp the tradetime 
month_return['Time'] = pd.to_datetime(month_return['Trdmnt'])
# convert the time string to the timestamp
month_return['numberTime'] = month_return['Time'].apply(dt.datetime.timestamp)
# forward the monthly return for each stock
# emrwd is the return including dividend
month_return['emrwd'] = month_return.groupby(['Stkcd'])['Mretwd'].shift(-1)
# emrnd is the return including no dividend
month_return['emrnd'] = month_return.groupby(['Stkcd'])['Mretnd'].shift(-1)
# select the A share stock
month_return = month_return[month_return['Markettype'].isin([1, 4, 16])]

# distinguish the stocks whose size is among the up 30% stocks in each month
def percentile(stocks) :
    return stocks >= stocks.quantile(q=.3)

month_return['cap'] = month_return.groupby(['Trdmnt'])['Msmvttl'].apply(percentile)

# construct Size
month_return['Size'] = np.log(month_return['Msmvttl'])

# %% merge data
from pandas.tseries.offsets import *

month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
#month_return['Date_merge'] += MonthEnd()

company_data['Stkcd_merge'] = company_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
company_data['Date_merge'] = pd.to_datetime(company_data['TradingDate'])
company_data['Date_merge'] += MonthBegin()

# dataset starts from '2000-01'
company_data = company_data[company_data['Date_merge'] >= '2000-01']
month_return = month_return[month_return['Date_merge'] >= '2000-01']
return_company = pd.merge(company_data, month_return, on=['Stkcd_merge', 'Date_merge'])

# %% construct horizontal pricing factors
from portfolio_analysis import Bivariate
from fama_macbeth import Factor_mimicking_portfolio as fmp
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1 = test_data_1[['emrwd', 'Size', 'PE1A', 'Date_merge', 'Size']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '2000-01-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

# Independent-sort Bivariate analysis
#bi_1 = Bivariate(np.array(test_data_1), number=4)
#bi_1.fit()
#bi_1.print_summary()

# %% factor mimicking portfolio
fmp1 = fmp(test_data_1)

SMB_horizon_1, HML_horizon_1 = fmp1.portfolio_return_horizon(period=1, ret=True)
SMB_horizon_1 = - SMB_horizon_1

SMB_horizon_2, HML_horizon_2 = fmp1.portfolio_return_horizon(period=2, ret=True)
SMB_horizon_2 = - SMB_horizon_2

SMB_horizon_3, HML_horizon_3 = fmp1.portfolio_return_horizon(period=3, ret=True)
SMB_horizon_3 = - SMB_horizon_3

SMB_horizon_6, HML_horizon_6 = fmp1.portfolio_return_horizon(period=6, ret=True)
SMB_horizon_6 = - SMB_horizon_6

SMB_horizon_12, HML_horizon_12 = fmp1.portfolio_return_horizon(period=12, ret=True)
SMB_horizon_12 = - SMB_horizon_12

SMB_horizon_24, HML_horizon_24 = fmp1.portfolio_return_horizon(period=24, ret=True)
SMB_horizon_24 = - SMB_horizon_24

SMB_horizon_36, HML_horizon_36 = fmp1.portfolio_return_horizon(period=36, ret=True)
SMB_horizon_36 = - SMB_horizon_36

SMB_horizon_48, HML_horizon_48 = fmp1.portfolio_return_horizon(period=48, ret=True)
SMB_horizon_48 = - SMB_horizon_48

SMB_horizon_60, HML_horizon_60 = fmp1.portfolio_return_horizon(period=60, ret=True)
SMB_horizon_60 = -SMB_horizon_60

# %% Construct Variance Ratio
def variance_ratio(series_k, series_1, k, summary=True):
    import numpy as np
    from scipy import stats as sts
    import pandas as pd

    series_merge = pd.concat([series_1, series_k], axis=1).dropna()
    vr = np.var(series_merge.iloc[:,1])/(np.var(series_merge.iloc[:,0])*k)
    
    n = len(series_merge)
    vr_stat = (vr-1)*((n*k)**0.5)/((2*(k-1))**0.5)
    t = vr_stat
    if t > 0:
        p = (1 - sts.norm.cdf(t))*2
    elif t < 0:
        p = (sts.norm.cdf(t))*2

    if summary == True:
        print('VR '+str(k)+' :', vr)
        print('t_value :', t)
        print('p_value :', p)
    
    return vr, t, p

# %% SMB Variance Ratio
vr, t, p = variance_ratio(SMB_horizon_2, SMB_horizon_1, 2)
vr, t, p = variance_ratio(SMB_horizon_3, SMB_horizon_1, 3)
vr, t, p = variance_ratio(SMB_horizon_6, SMB_horizon_1, 6)
vr, t, p = variance_ratio(SMB_horizon_12, SMB_horizon_1, 12)
vr, t, p = variance_ratio(SMB_horizon_24, SMB_horizon_1, 24)
vr, t, p = variance_ratio(SMB_horizon_48, SMB_horizon_1, 48)
vr, t, p = variance_ratio(SMB_horizon_60, SMB_horizon_1, 60)

# %% HML
vr, t, p = variance_ratio(HML_horizon_2, HML_horizon_1, 2)
vr, t, p = variance_ratio(HML_horizon_3, HML_horizon_1, 3)
vr, t, p = variance_ratio(HML_horizon_6, HML_horizon_1, 6)
vr, t, p = variance_ratio(HML_horizon_12, HML_horizon_1, 12)
vr, t, p = variance_ratio(HML_horizon_24, HML_horizon_1, 24)
vr, t, p = variance_ratio(HML_horizon_48, HML_horizon_1, 48)
vr, t, p = variance_ratio(HML_horizon_60, HML_horizon_1, 60)

# %%





