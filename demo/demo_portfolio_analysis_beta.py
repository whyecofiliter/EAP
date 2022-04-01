# %% import package
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(".."))

# %% import data
# Monthly return of stocks in China security market
month_return = pd.read_hdf('.\\data\\month_return.h5', key='month_return')
beta = pd.read_hdf('.\\data\\beta.h5', key='data')

# %% data preprocessing
# select the A share stock
month_return = month_return[month_return['Markettype'].isin([1, 4, 16])]
# forward the monthly return for each stock
# emrwd is the return including dividend
month_return['emrwd'] = month_return.groupby(['Stkcd'])['Mretwd'].shift(-1)
# emrnd is the return including no dividend
month_return['emrnd'] = month_return.groupby(['Stkcd'])['Mretnd'].shift(-1)

# % distinguish the stocks whose size is among the up 30% stocks in each month
def percentile(stocks) :
    return stocks >= stocks.quantile(q=.3)

month_return['cap'] = month_return.groupby(['Trdmnt'])['Msmvttl'].apply(percentile)

# %% construct market portfolio return
import numpy as np
market_portfolio_return = month_return['Mretwd'].groupby(month_return['Trdmnt']).apply(np.mean)
market_portfolio_return.name = 'market_portfolio_return'
month_return = pd.merge(month_return, market_portfolio_return, on=['Trdmnt'])
month_return = month_return.sort_values(['Stkcd', 'Trdmnt'])

# %% calculate beta
import statsmodels.api as sm
import numpy as np

reg = month_return.set_index(['Stkcd', 'Trdmnt'])[['Mretwd', 'market_portfolio_return']].dropna()
beta = pd.Series(index=reg.index, dtype=float, name='beta')
for i in reg.groupby('Stkcd'):
    row, col = np.shape(i[1])
    for j in range(row-12):
        model = sm.OLS(i[1].iloc[j:j+12, 0], sm.add_constant(i[1].iloc[j:j+12, 1])).fit()
        beta.loc[i[1].index[j+11]] = model.params[1]

reg = pd.merge(reg, beta, left_index=True, right_index=True)

# %% Data preprocessing: merge data
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
month_return['Stkcd_merge'] = pd.to_datetime(month_return['Stkcd'])
month_return = month_return.set_index(['Stkcd', 'Trdmnt'])
month_return = pd.merge(month_return, beta, left_index=True, right_index=True)
# construct label
# data starts from 2000-01
return_company = month_return[month_return['Date_merge']>='2000-01']

# %% construct test_data for bivariate analysis
# dataset 1
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1 = test_data_1[['emrwd', 'Msmvttl', 'beta', 'Date_merge']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '2000-01-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_1 = Univariate(np.array(test_data_1[['emrwd', 'beta', 'Date_merge']]), number=9)
uni_1.fit()
uni_1.print_summary_by_time()
uni_1.print_summary()
# summary statistics
uni_1.summary_statistics()

# %% Persistence Analysis
from portfolio_analysis import Persistence as perse

test_data_1_per = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1_per = test_data_1_per[['Stkcd_merge', 'Date_merge', 'beta']]
perse_1 = perse(test_data_1_per)
perse_1.fit(lags=[1, 2, 3])
perse_1.summary(periodic=True)
perse_1.summary(periodic=False)

# %% Independent-sort Bivariate analysis
bi_1 = Bivariate(np.array(test_data_1), number=4)
bi_1.average_by_time()
bi_1.summary_and_test()
bi_1.print_summary_by_time()
bi_1.print_summary()

# Dependent-sort Bivariate analysis
bi_1_de = Bivariate(test_data_1, number=4)
bi_1_de.fit(conditional=True)
bi_1_de.print_summary_by_time()
bi_1_de.print_summary()

# %% construct test_data for bivariate analysis
# dataset 2
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_2 = return_company[(return_company['Ndaytrd']>=10)]
test_data_2 = test_data_2[['emrwd', 'Msmvttl', 'beta', 'Date_merge']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2000-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_2 = Univariate(np.array(test_data_2[['emrwd', 'beta', 'Date_merge']]), number=9)
uni_2.summary_and_test()
uni_2.print_summary_by_time()
uni_2.print_summary()

# Bivariate analysis
bi_2 = Bivariate(np.array(test_data_2), number=4)
bi_2.average_by_time()
bi_2.summary_and_test()
bi_2.print_summary_by_time()
bi_2.print_summary()

# Dependent-sort Biviraite analysis
bi_2_de = Bivariate(test_data_2, number=4)
bi_2_de.fit(conditional=True)
bi_2_de.print_summary_by_time()
bi_2_de.print_summary()

# %%
