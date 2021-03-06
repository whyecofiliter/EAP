'''
Factor mimicking portfolio in China market
'''
#%%
from cProfile import label
import sys,os
sys.path.append(os.path.abspath(".."))

# %% import data
# Monthly return of stocks in China security market
import pandas as pd

month_return = pd.read_hdf('.\data\month_return.h5', key='month_return')
company_data = pd.read_hdf('.\data\last_filter_pe.h5', key='data')
trade_data = pd.read_hdf('.\data\mean_filter_trade.h5', key='data')

# %% data preprocessing
# select the A share stock
month_return = month_return[month_return['Markettype'].isin([1, 4, 16])]

# % distinguish the stocks whose size is among the up 30% stocks in each month
def percentile(stocks) :
    return stocks >= stocks.quantile(q=.3)

month_return['cap'] = month_return.groupby(['Trdmnt'])['Msmvttl'].apply(percentile)

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

# %% SMB and HML
from fama_macbeth import Factor_mimicking_portfolio as fmp
import numpy as np
import matplotlib.pyplot as plt

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

# %%



