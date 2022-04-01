'''
F SCORE
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
debt = pd.read_hdf('.\\data\\debt.h5', key='data')
profit = pd.read_hdf('.\\data\\profit.h5', key='data')
cash = pd.read_hdf('.\\data\\cash.h5', key='data')
operation = pd.read_hdf('.\\data\\operation.h5', key='data')
eq_offer = pd.read_hdf('.\\data\\eq_offer.h5', key='data')

# %% construct variable
# ROA
profit['ROA(TTM)'] = profit['F050204C']
# diff ROA
profit['delta_ROA'] = profit.groupby(['Stkcd'])['ROA(TTM)'].diff(4)
# CFOA = operation cash flow / total asset
profit['CFOA'] = profit['ROA(TTM)'] * profit['F052901C'] / profit['F050901B']
# Accrued profit 
profit['accrued_profit'] = profit['F051101B'] - profit['CFOA']

# long term debt
debt['delta_LEVER'] = debt.groupby(['Stkcd'])['F011901A'].diff(4)
# diff liquidity
debt['delta_LIQUID'] = debt.groupby(['Stkcd'])['F010101A'].diff(4)
# EQ_OFFER already have 

# margin profit
profit['delta_MARGIN'] = profit['F050201B'] / profit['F050901B']
# capital asset
operation['delta_TURN'] = operation.groupby(['Stkcd'])['F041701B'].diff(4)


def compare(x):
    if x > 0 :
        return 1
    else: 
        return 0
def compare_re(x):
    if x < 0 :
        return 1
    else:
        return 0

def compare_eq(x):
    if x <= 0:
        return 1
    else:
        return 0

# count 
profit['count_ROA(TTM)'] = profit['ROA(TTM)'].apply(compare)
profit['count_delta_ROA'] = profit['delta_ROA'].apply(compare)
profit['count_CFOA'] = profit['CFOA'].apply(compare)
profit['count_accrued_profit'] = profit['accrued_profit'].apply(compare_re)

debt['count_delta_LEVER'] = debt['delta_LEVER'].apply(compare_re)
debt['count_delta_LIQUID'] = debt['delta_LIQUID'].apply(compare_re)
eq_offer['count_EQ_OFFER'] = eq_offer['EQ_OFFER'].apply(compare_eq)

profit['count_delta_MARGIN'] = profit['delta_MARGIN'].apply(compare)
operation['count_delta_TURN'] = operation['delta_TURN'].apply(compare)

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

profit = profit[profit['Typrep']=='A']
debt = debt[debt['Typrep']=='A']
eq_offer = eq_offer[eq_offer['Markettype'].isin([1,4,16])]
operation = operation[operation['Typrep']=='A']

# %%construct variable
import numpy as np
# BM
company_data['BM'] = 1 / company_data['PBV1A']

# Size
month_return['Size'] = np.log(month_return['Msmvttl'])

# %% merge data
from pandas.tseries.offsets import *

month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
#month_return['Date_merge'] += MonthEnd()

profit['Stkcd_merge'] = profit['Stkcd'].astype(dtype='string')
profit['Date_merge'] = pd.to_datetime(profit['Accper'])
profit['Date_merge'] += MonthBegin()

debt['Stkcd_merge'] = debt['Stkcd'].astype(dtype='string')
debt['Date_merge'] = pd.to_datetime(debt['Accper'])
debt['Date_merge'] += MonthBegin()

eq_offer['Stkcd_merge'] = eq_offer['Stkcd'].astype(dtype='string')
eq_offer['Date_merge'] = pd.to_datetime(eq_offer['Trdmnt'])

operation['Stkcd_merge'] = operation['Stkcd'].astype(dtype='string')
operation['Date_merge'] = pd.to_datetime(operation['Accper'])
operation['Date_merge'] += MonthBegin()

company_data['Stkcd_merge'] = company_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')
company_data['Date_merge'] = pd.to_datetime(company_data['TradingDate'])
company_data['Date_merge'] += MonthBegin()

# merge 
return_company = pd.merge(month_return, profit, on=['Stkcd_merge', 'Date_merge'], how='left')
return_company = pd.merge(return_company, debt, on=['Stkcd_merge', 'Date_merge'], how='left')
return_company = pd.merge(return_company, eq_offer, on=['Stkcd_merge', 'Date_merge'], how='left')
return_company = pd.merge(return_company, operation, on=['Stkcd_merge', 'Date_merge'], how='left')
return_company = pd.merge(return_company, company_data, on=['Stkcd_merge', 'Date_merge'], how='left')

return_company['Stkcd_fillna'] = return_company['Stkcd_merge'] 
return_company = return_company.groupby(['Stkcd_fillna']).fillna(method='pad')
return_company = return_company.fillna(0)

# %% construct F score
return_company['F_SCORE'] = return_company[['count_ROA(TTM)', 'count_delta_ROA', 'count_CFOA', 'count_accrued_profit', 'count_delta_LEVER', 'count_delta_LIQUID', 'count_EQ_OFFER', 'count_delta_MARGIN', 'count_delta_TURN']].sum(axis=1)

# %% construct test_data for bivariate analysis
# dataset 1
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1 = test_data_1[['emrwd', 'F_SCORE', 'BM', 'Date_merge']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '2000-01-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_1 = Univariate(test_data_1[['emrwd', 'F_SCORE', 'Date_merge','BM']], number=2, weight=True)
uni_1.fit()
uni_1.print_summary()

# Bivariate analysis
bi_1 = Bivariate(test_data_1, number=2)
bi_1.fit()
bi_1.print_summary()

# %% risk adjustment
risk_model = risk_premium[['MKT', 'SMB', 'HML']]
risk_model = risk_model['2000':'2019']
bi_1.factor_adjustment(risk_model)
bi_1.print_summary()

# %% construct test_data for bivariate analysis
# dataset 2: Size weighted
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_2 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_2 = test_data_2[['emrwd', 'F_SCORE', 'BM', 'Date_merge', 'Size']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2000-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_2 = Univariate(test_data_2[['emrwd', 'F_SCORE', 'Date_merge']], number=2)
uni_2.fit()
uni_2.print_summary()

# Bivariate analysis
bi_2 = Bivariate(test_data_2, number=2, weight=True)
bi_2.fit()
bi_2.print_summary()

# %% risk adjustment
risk_model = risk_premium[['MKT', 'SMB', 'HML']]
risk_model = risk_model['2000':'2019']
bi_2.factor_adjustment(risk_model)
bi_2.print_summary()
