# %% import package
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(".."))

# %% import data
# Monthly return of stocks in China security market
month_return = pd.read_hdf('.\\data\\month_return.h5', key='month_return')
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

# %% prepare merge data
from pandas.tseries.offsets import *

month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
month_return['Yearmonth'] = month_return['Date_merge'].map(lambda x : 1000*x.year + x.month)
#month_return['Date_merge'] += MonthEnd()

# in this demo, the ROE(TTM) are used
# ROE(TTM) = PBV1B/PE(TTM) 
company_data['ROE(TTM)'] = company_data['PBV1B']/company_data['PE1TTM']
company_data['Stkcd_merge'] = company_data['Symbol'].dropna().astype(dtype='int').astype(dtype='string')

company_data['Date_merge'] = pd.to_datetime(company_data['TradingDate'])
company_data['Yearmonth'] = company_data['Date_merge'].map(lambda x : 1000*x.year + x.month)
company_data['Date_merge'] += MonthBegin()

# %% dataset starts from '2000-01'
company_data = company_data[company_data['Date_merge'] >= '2000-01']
month_return = month_return[month_return['Date_merge'] >= '2000-01']
return_company = pd.merge(company_data, month_return, on=['Stkcd_merge', 'Date_merge'])

# %% construct test_data for bivariate analysis
# dataset 1 : no tail stocks & ROE Bivariate
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1 = test_data_1[['emrwd', 'Msmvttl', 'ROE(TTM)', 'Date_merge', 'Mnshrtrd', 'Mnvaltrd']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '1999-12-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_1 = Univariate(np.array(test_data_1[['emrwd', 'ROE(TTM)', 'Date_merge']]), number=9)
uni_1.summary_and_test()
uni_1.print_summary_by_time()
uni_1.print_summary()

# %%
uni_1.summary_statistics()
uni_1.correlation(variables=np.array(test_data_1[['emrwd', 'ROE(TTM)']]), periodic=True)

# %% Risk adjustment
risk_model = risk_premium[['MKT', 'SMB', 'HML']]
risk_model = risk_model['2000':'2019']
uni_1.factor_adjustment(risk_model)
uni_1.print_summary()

# %% Test autocorrelation of factor momentum
import statsmodels.api as sm
import matplotlib.pyplot as plt

fac_mom = uni_1.difference(uni_1.average_by_time())[-1, :]
fac_acf = sm.tsa.stattools.acf(fac_mom, qstat=True, nlags=12)
print(fac_acf)
plt.plot(fac_acf[0])

# %% build AR model
ar = sm.tsa.arima.ARIMA(fac_mom, order=([4,5,12], 0, 0), trend='n')
res = ar.fit()
print(res.summary())

pre_fac = res.predict()

# %%
uni_1.summary_statistics(test_data_1[['Mnshrtrd', 'Mnvaltrd']], periodic=True)
vol = uni_1.average_variable_period
vol_1 = np.log(np.mean(vol[[0,9], 1, :], axis=0))
vol_2 = np.mean(np.log(vol[[0,9], 1, :]), axis=0)
vol_3 = np.mean(np.log(vol[[0,9], 2, :]), axis=0)

vol_acf = sm.tsa.stattools.acf(vol_1, qstat=True, nlags=12)
print(vol_acf)
plt.plot(vol_acf[0])

# %% build AR model
ar_vol = sm.tsa.arima.ARIMA(vol_1, order=([1,3,6,7], 0, 0), trend='c')
res_vol = ar_vol.fit()
print(res_vol.summary())

pre_vol_1 = res_vol.predict()

# %% fit the fac_mom and vol_2
model = sm.OLS(pre_fac, sm.add_constant(vol_1)).fit()
print(model.summary())
model_pre = sm.OLS(pre_fac, sm.add_constant(pre_vol_1)).fit()
print(model_pre.summary())

# %% build regression model
resid_acf = sm.tsa.stattools.acf(model.resid, qstat=True, nlags=12)
print(resid_acf)
plt.plot(resid_acf[0])

ar_resid = sm.tsa.arima.ARIMA(model.resid, order=[[4,5,12],0,0], trend='n')
res_resid = ar_resid.fit()
print(res_resid.summary())

# %% Independent-sort Bivariate analysis
bi_1 = Bivariate(np.array(test_data_1), number=4)
bi_1.average_by_time()
bi_1.summary_and_test()
bi_1.print_summary_by_time()
bi_1.print_summary()

# Risk adjustment
risk_model = risk_premium[['MKT', 'SMB', 'HML']]
risk_model = risk_model['2000':'2019']
bi_1.factor_adjustment(risk_model)
bi_1.print_summary()

# Dependent-sort Bivariate Analysis
bi_1_de = Bivariate(test_data_1, number=4)
bi_1_de.fit(conditional=True)
bi_1_de.print_summary()

# Risk Adjustment
bi_1_de.factor_adjustment(risk_model)
bi_1_de.print_summary()

# %% construct test_data for bivariate analysis
# dataset 2 : tail stocks & ROE Bivariate  
from portfolio_analysis import Bivariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_2 = return_company[return_company['Ndaytrd']>=10]
test_data_2 = test_data_2[['emrwd', 'Msmvttl', 'ROE(TTM)', 'Date_merge']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2000-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_2 = Univariate(np.array(test_data_2[['emrwd', 'ROE(TTM)', 'Date_merge']]), number=9)
uni_2.summary_and_test()
uni_2.print_summary_by_time()
uni_2.print_summary()

# analysis
bi_2 = Bivariate(np.array(test_data_2), number=4)
bi_2.average_by_time()
bi_2.summary_and_test()
bi_2.print_summary_by_time()
bi_2.print_summary()

# Risk adjustment
risk_model = risk_premium[['MKT', 'SMB', 'HML']]
risk_model = risk_model['2000':'2019']
bi_2.factor_adjustment(risk_model)
bi_2.print_summary()

# Dependent-sort Bivariate Analysis
bi_2_de = Bivariate(test_data_2, number=4)
bi_2_de.fit(conditional=True)
bi_2_de.print_summary()

# Risk Adjustment
bi_2_de.factor_adjustment(risk_model)
bi_2_de.print_summary()

# %%
