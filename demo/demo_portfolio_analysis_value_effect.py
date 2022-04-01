# %% import package
from linearmodels import test
from numpy import dtype
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(".."))

# %% import data
# Monthly return of stocks in China security market
month_return = pd.read_hdf('.\data\month_return.h5', key='month_return')
company_data = pd.read_hdf('.\data\last_filter_pe.h5', key='data')
risk_premium = pd.read_hdf('.\data\\risk_premium.h5', key='data')

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

# %% merge data
from pandas.tseries.offsets import *

month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
#month_return['Date_merge'] += MonthEnd()

company_data['Stkcd_merge'] = company_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
company_data['Date_merge'] = pd.to_datetime(company_data['TradingDate'])
company_data['Date_merge'] += MonthBegin()

# %% dataset starts from '2000-01'
company_data = company_data[company_data['Date_merge'] >= '2000-01']
month_return = month_return[month_return['Date_merge'] >= '2000-01']
return_company = pd.merge(company_data, month_return, on=['Stkcd_merge', 'Date_merge'])

# %% construct test_data for bivariate analysis
# dataset 1 : PE
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1 = test_data_1[['emrwd', 'Msmvttl', 'PE1A', 'Date_merge']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '2000-01-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

# Univariate Analysis
uni_1 = Univariate(test_data_1[['emrwd', 'PE1A', 'Date_merge']], number=10)
uni_1.fit()
uni_1.summary_statistics(variables=test_data_1[['Msmvttl', 'emrwd']], periodic=True)
uni_1.print_summary()

# correlation 
uni_1.correlation(variables=np.array(test_data_1[['PE1A', 'emrwd']]), periodic=False)

# Independent-sort Bivariate analysis
bi_1 = Bivariate(np.array(test_data_1), number=4)
bi_1.average_by_time()
bi_1.summary_and_test()
bi_1.print_summary_by_time()
bi_1.print_summary()

# risk adjustment
print('--------------------------------------- RISK ADJUSTMET --------------------------------')
risk_model = risk_premium['MKT']
risk_model = risk_model['2000':'2019']
bi_1.factor_adjustment(risk_model)
bi_1.print_summary()

print('-------------------------------------- Dependent-Sort ---------------------------------')
# Dependent-sort Bivariate analysis
bi_1_de = Bivariate(test_data_1, number=4)
bi_1_de.fit(conditional=True)
bi_1_de.print_summary()

print('---------------------------------------Dependent-Sort RISK ADJUSTMENT ------------------')
bi_1_de.factor_adjustment(risk_model)
bi_1_de.print_summary()

# %% construct test_data for bivariate analysis
# dataset 2 : PB
from portfolio_analysis import Bivariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_2 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_2 = test_data_2[['emrwd', 'Msmvttl', 'PBV1A', 'Date_merge']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2000-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]
# Independent-sort Bivariate analysis
bi_2 = Bivariate(np.array(test_data_2), number=4)
bi_2.average_by_time()
bi_2.summary_and_test()
bi_2.print_summary_by_time()
bi_2.print_summary()

# Risk Adjustment
print('--------------------------------------- RISK ADJUSTMET --------------------------------')
risk_model = risk_premium['MKT']
risk_model = risk_model['2000':'2019']
bi_2.factor_adjustment(risk_model)
bi_2.print_summary()

# Dependent-sort Bivariate Analysis
print('-------------------------------------- Dependent-Sort ---------------------------------')
bi_2_de = Bivariate(test_data_2, number=4)
bi_2_de.fit(conditional=True)
bi_2_de.print_summary()

# Risk Adjustment
print('---------------------------------------Dependent-Sort RISK ADJUSTMENT ------------------')
bi_2_de.factor_adjustment(risk_model)
bi_2_de.print_summary()

# %% construct test_data for bivariate analysis
# dataset 3 : PE 
from portfolio_analysis import Bivariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_3 = return_company[return_company['Ndaytrd']>=10]
test_data_3 = test_data_3[['emrwd', 'Msmvttl', 'PE1A', 'Date_merge']].dropna()
test_data_3 = test_data_3[(test_data_3['Date_merge'] >= '2000-01-01') & (test_data_3['Date_merge'] <= '2019-12-01')]

# Independent-sort Bivariate analysis
bi_3 = Bivariate(np.array(test_data_3), number=4)
bi_3.average_by_time()
bi_3.summary_and_test()
bi_3.print_summary_by_time()
bi_3.print_summary()

# Risk Adjustment
print('--------------------------------------- RISK ADJUSTMET --------------------------------')
risk_model = risk_premium['MKT']
risk_model = risk_model['2000':'2019']
bi_3.factor_adjustment(risk_model)
bi_3.print_summary()

# Dependent-sort Bivariate Analysis
print('-------------------------------------- Dependent-Sort ---------------------------------')
bi_3_de = Bivariate(test_data_3, number=4)
bi_3_de.fit(conditional=True)
bi_3_de.print_summary()

# Risk Adjustment
print('---------------------------------------Dependent-Sort RISK ADJUSTMENT ------------------')
bi_3_de.factor_adjustment(risk_model)
bi_3_de.print_summary()

# %% construct test_data for bivariate analysis
# dataset 4 : PB
from portfolio_analysis import Bivariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_4 = return_company[(return_company['Ndaytrd']>=10)]
test_data_4 = test_data_4[['emrwd', 'Msmvttl', 'PBV1A', 'Date_merge']].dropna()
test_data_4 = test_data_4[(test_data_4['Date_merge'] >= '2000-01-01') & (test_data_4['Date_merge'] <= '2019-12-01')]
# Independent-sort Bivariate Analysis
bi_4 = Bivariate(np.array(test_data_4), number=4)
bi_4.average_by_time()
bi_4.summary_and_test()
bi_4.print_summary_by_time()
bi_4.print_summary()

# Risk Adjustment
print('--------------------------------------- RISK ADJUSTMET --------------------------------')
risk_model = risk_premium['MKT']
risk_model = risk_model['2000':'2019']
bi_4.factor_adjustment(risk_model)
bi_4.print_summary()

# Dependent-sort Bivariate Analysis
print('-------------------------------------- Dependent-Sort ---------------------------------')
bi_4_de = Bivariate(test_data_4, number=4)
bi_4_de.fit(conditional=True)
bi_4_de.print_summary()

# Risk Adjustment
print('---------------------------------------Dependent-Sort RISK ADJUSTMENT ------------------')
bi_4_de.factor_adjustment(risk_model)
bi_4_de.print_summary()

# %% Persistence Analysis
from portfolio_analysis import Persistence as perse

test_data_1_per = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1_per = test_data_1_per[['Stkcd_merge', 'Date_merge', 'PE1A']]
perse_1 = perse(test_data_1_per)
perse_1.fit(lags=[1, 2, 3])
perse_1.summary(periodic=True)
perse_1.summary(periodic=False)

# %%
