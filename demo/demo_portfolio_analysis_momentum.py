# %% import package
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(".."))

# %% import data
# Monthly return of stocks in China security market
month_return = pd.read_hdf('.\data\month_return.h5', key='month_return')
#month_return = month_return.set_index(['Stkcd', 'Trdmnt'], drop=False)

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

# %% construct momentum variable
import numpy as np

month_return['rolling_12'] = np.array(month_return.groupby(['Stkcd'])['Mretwd'].rolling(12).sum())
month_return['momentum'] = month_return['rolling_12'] - month_return['Mretwd']

# %% construct label
month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
# data starts from 2000-01
return_company = month_return[month_return['Date_merge']>='2000-01']

# %% construct test_data for bivariate analysis
# dataset 1
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1 = test_data_1[['emrwd', 'Msmvttl', 'momentum', 'Date_merge']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '2000-01-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_1 = Univariate(np.array(test_data_1[['emrwd', 'momentum', 'Date_merge']]), number=9)
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
# dataset 2
from portfolio_analysis import Bivariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_2 = return_company[return_company['Ndaytrd']>=10]
test_data_2 = test_data_2[['emrwd', 'Msmvttl', 'momentum', 'Date_merge']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2005-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_2 = Univariate(np.array(test_data_1[['emrwd', 'momentum', 'Date_merge']]), number=9)
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
