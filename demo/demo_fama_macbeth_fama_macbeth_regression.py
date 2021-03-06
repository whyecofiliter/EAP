'''
Fama-Macbeth regression
'''

# %% set system path
import sys,os
sys.path.append(os.path.abspath(".."))

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

# %% Fama-Macbeth regression
# dataset : #2
# include tail stocks
# range from 2000-01 ~ 2019-12-01
from fama_macbeth import Fama_macbeth_regress

test_data_2 = return_company[(return_company['Ndaytrd']>=10)]
test_data_2 = test_data_2[['emrwd', 'beta', 'Size', 'BM', 'ROE(TTM)', 'asset_growth_rate', 'momentum', 'specific_Turnover', 'Date_merge']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2000-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]

model = Fama_macbeth_regress(test_data_2)
result = model.fit(add_constant=True)
model.summary_by_time()
model.summary()

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
model.summary()
model.summary_by_time()

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
model.summary()
model.summary_by_time()

# %% Fama-Macbeth regression
# dataset : #2
# include tail stocks
# range from 2000-01 ~ 2019-12-01
from fama_macbeth import Fama_macbeth_regress

test_data_2 = return_company[(return_company['Ndaytrd']>=10)]
test_data_2 = test_data_2[['emrwd', 'beta', 'Size', 'BM', 'ROE(TTM)', 'asset_growth_rate', 'Date_merge']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2000-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]

model = Fama_macbeth_regress(test_data_2)
result = model.fit(add_constant=True)
model.summary_by_time()
model.summary()

# %%
