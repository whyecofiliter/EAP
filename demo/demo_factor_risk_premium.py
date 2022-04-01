'''
Factor risk premium
'''
# %% set system path
import sys,os
sys.path.append(os.path.abspath(".."))

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

# merge data
return_company = pd.merge(month_return, company_data, on=['Stkcd_merge', 'Date_merge'])
return_company = pd.merge(return_company, trade_data, on=['Stkcd_merge', 'Date_merge'])

# beta
return_company = return_company.set_index(['Stkcd', 'Trdmnt'])
return_company = pd.merge(return_company, beta, left_index=True, right_index=True)

# %% generate factor risk premium
from fama_macbeth import Factor_mimicking_portfolio
import numpy as np

# Size and Value factor risk premium
# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
size_bm = return_company[(return_company['Ndaytrd']>=10)]
size_bm = size_bm[['emrwd', 'Size', 'BM', 'Date_merge', 'Size']].dropna()
#size_bm = size_bm[(size_bm['Date_merge'] >= '1991-01-01') & (size_bm['Date_merge'] <= '2019-12-01')]
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
#size_cma = size_cma[(size_cma['Date_merge'] >= '2000-01-01') & (size_cma['Date_merge'] <= '2019-12-01')]
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
CNBETA = CNBETA.rename('MKT')

# %% merge data
risk_premium = pd.concat([CNBETA, CNSMB, CNHML, CNRMW, CNCMA], axis=1).shift(1)
