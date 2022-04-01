'''
Anomaly Portfolio
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

# merge data
data = pd.concat([ret_pcf, ret_inv, ret_abtr1mon, risk_premium], axis=1)
data = data['2004':'2019'].dropna()

# %% construct model
# Fama-French 3 factors model
# Without Newey-West Test
from time_series_regress import TS_regress

list_data = [np.array(data.iloc[:, i]) for i in range(3)]
factor = data.iloc[:, 3:6]
model = TS_regress(list_y=list_data, factor=np.array(factor))
model.fit(newey_west=False)
model.summary()

# %% Fama-French 3 factors model
# With Newey-West adjustment
model.fit(newey_west=True)
model.summary()

# %% Fama-French 3 factors model
# Without Newey-West adjustment
# import data type: Dataframe

list_data = data.iloc[:, :3]
factor = data.iloc[:, 3:6]
model = TS_regress(list_y=list_data, factor=factor)
model.fit(newey_west=False)
model.summary()

# %% Fama-French 3 factors model
# With Newey-West adjustment
# import data type: Dataframe
model = TS_regress(list_y=data.iloc[:, :3], factor=data.iloc[:, 3:6])
model.fit(newey_west=True)
model.summary()

# %% Fama-French 5 factors model
# With Newey-West adjustment
# import data type: Dataframe
model = TS_regress(list_y=data.iloc[:, :3], factor=data.iloc[:, 3:])
model.fit(newey_west=True)
model.summary()

# %%
