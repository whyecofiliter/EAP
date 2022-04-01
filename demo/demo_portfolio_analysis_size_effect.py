# %% import package
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(".."))

# %% import data
# Monthly return of Stocks in China Security Market
month_return = pd.read_hdf('.\data\month_return.h5', key='month_return')
risk_premium = pd.read_hdf('.\data\\risk_premium.h5', key='data')

# %% add time
import datetime as dt

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

# %% construct test_data for univariate analysis
# dataset 1
from portfolio_analysis import Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_1 = month_return[(month_return['cap']==True) & (month_return['Ndaytrd']>=10)]
# construct data for univariate analysis
test_data_1 = test_data_1[['emrwd', 'Msmvttl', 'Time']].dropna()
test_data_1 = test_data_1[(test_data_1['Time'] >= '2000-01-01') & (test_data_1['Time'] <= '2019-12-01')]

# Univariate analysis
uni_1 = Univariate(np.array(test_data_1), number=9)
uni_1.average_by_time()
uni_1.summary_and_test()
uni_1.print_summary_by_time()
uni_1.print_summary()

# %% dataset 2
# select stocks whose trading days are more than or equal to 10 days
test_data_2 = month_return[month_return['Ndaytrd']>=10]
# construct data for univariate analysis
test_data_2 = test_data_2[['emrwd', 'Msmvttl', 'Time']].dropna()
# time interval between 2000-01-01 to 2019-12-01
test_data_2 = test_data_2[(test_data_2['Time'] >= '2000-01-01') & (test_data_2['Time'] <= '2019-12-01')]
# using Univariate class to conduct analysis
uni_2 = Univariate(np.array(test_data_2), number=9)
uni_2.average_by_time()
uni_2.summary_and_test()
uni_2.print_summary_by_time()
uni_2.print_summary()

# %% dataset 3
# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_3 = month_return[(month_return['cap']==True) & (month_return['Ndaytrd']>=10)]
# construct data for univariate analysis
test_data_3 = test_data_3[['emrwd', 'Msmvttl', 'Time']].dropna()
# time interval between 2000-01-01 to 2016-12-01
test_data_3 = test_data_3[(test_data_3['Time'] >= '2000-01-01') & (test_data_3['Time'] <= '2016-12-01')]
# analysis
uni_3 = Univariate(np.array(test_data_3), number=9)
uni_3.average_by_time()
uni_3.summary_and_test()
uni_3.print_summary_by_time()
uni_3.print_summary()

# %% dataset 4
# select stocks whose trading days are more than or equal to 10 days
test_data_4 = month_return[month_return['Ndaytrd']>=10]
# construct data for univariate analysis
test_data_4 = test_data_4[['emrwd', 'Msmvttl', 'Time']].dropna()
# time interval between 2000-01-01 to 2016-12-01
test_data_4 = test_data_4[(test_data_4['Time'] >= '2000-01-01') & (test_data_4['Time'] <= '2016-12-01')]
# analysis
uni_4 = Univariate(np.array(test_data_4), number=9)
uni_4.average_by_time()
uni_4.summary_and_test()
uni_4.print_summary_by_time()
uni_4.print_summary()

# %%
# %% Persistence Analysis
from portfolio_analysis import Persistence as perse

test_data_1_per = month_return[['Stkcd', 'Time', 'Msmvttl']]
perse_1 = perse(test_data_1_per)
perse_1.fit(lags=[1, 2, 3])
perse_1.summary(periodic=True)
perse_1.summary(periodic=False)
