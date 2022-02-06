# %% import package
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(".."))

# %% import data
# Monthly return of stocks in China security market
month_return = pd.read_hdf('.\data\month_return.h5', key='month_return')
trade_data = pd.read_hdf('.\data\mean_filter_trade.h5', key='data')

# %% preprocessing data
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

# %% constrcut variable
import numpy as np
trade_data['rolling_Turnover'] = np.array(trade_data['Turnover'].groupby('Symbol').rolling(12).mean())
trade_data['specific_Turnover'] = trade_data['Turnover'] / trade_data['rolling_Turnover']

# %% prepare merge data
from pandas.tseries.offsets import *

month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
#month_return['Yearmonth'] = month_return['Date_merge'].map(lambda x : 1000*x.year + x.month)
#month_return['Date_merge'] += MonthEnd()

trade_data['Stkcd_merge'] = trade_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
trade_data['TradingDate'] = trade_data.index.map(lambda x : x[1])
trade_data['Date_merge'] = pd.to_datetime(trade_data['TradingDate'])
#company_data['Yearmonth'] = company_data['Date_merge'].map(lambda x : 1000*x.year + x.month)
trade_data['Date_merge'] += MonthBegin()

# %% dataset starts from '2000-01'
trade_data = trade_data[trade_data['Date_merge'] >= '2000-01']
month_return = month_return[month_return['Date_merge'] >= '2000-01']
return_company = pd.merge(trade_data, month_return, on=['Stkcd_merge', 'Date_merge'])

# %% construct test_data for bivariate analysis
# dataset 1 : 
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1 = test_data_1[['emrwd', 'Msmvttl', 'specific_Turnover', 'Date_merge']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '2000-01-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_1 = Univariate(np.array(test_data_1[['emrwd', 'specific_Turnover', 'Date_merge']]), number=9)
uni_1.summary_and_test()
uni_1.print_summary_by_time()
uni_1.print_summary()

# Bivariate analysis
bi_1 = Bivariate(np.array(test_data_1), number=4)
bi_1.average_by_time()
bi_1.summary_and_test()
bi_1.print_summary_by_time()
bi_1.print_summary()

# %% construct test_data for bivariate analysis
# dataset 2 : 
from portfolio_analysis import Bivariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_2 = return_company[return_company['Ndaytrd']>=10]
test_data_2 = test_data_2[['emrwd', 'Msmvttl', 'specific_Turnover', 'Date_merge']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2004-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_2 = Univariate(np.array(test_data_2[['emrwd', 'specific_Turnover', 'Date_merge']]), number=9)
uni_2.summary_and_test()
uni_2.print_summary_by_time()
uni_2.print_summary()

# Bivariate analysis
bi_2 = Bivariate(np.array(test_data_2), number=4)
bi_2.average_by_time()
bi_2.summary_and_test()
bi_2.print_summary_by_time()
bi_2.print_summary()

# %%
