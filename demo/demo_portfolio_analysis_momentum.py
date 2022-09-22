# %% import package
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(".."))

# %% import data
# Monthly return of stocks in China security market
month_return = pd.read_hdf('.\data\month_return.h5', key='month_return')
#month_return = month_return.set_index(['Stkcd', 'Trdmnt'], drop=False)
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

# %% construct momentum variable
import numpy as np

month_return['rolling_12'] = np.array(month_return.groupby(['Stkcd'])['Mretwd'].rolling(12).sum())
month_return['momentum'] = month_return['rolling_12'] - month_return['Mretwd']

# %% construct label
month_return['Stkcd_merge'] = month_return['Stkcd'].astype(dtype='string')
month_return['Date_merge'] = pd.to_datetime(month_return['Trdmnt'])
# data starts from 2000-01
return_company = month_return[month_return['Date_merge']>='2000-01']
return_company['Size'] = np.log(return_company['Msmvttl'])
return_company['Turnover'] = return_company['Mnshrtrd'] / (return_company['Msmvosd']/return_company['Mclsprc'])

# %% construct test_data for bivariate analysis
# dataset 1
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1 = test_data_1[['emrwd', 'Msmvttl', 'momentum', 'Date_merge', 'Size', 'Mnshrtrd', 'Mnvaltrd', 'Turnover']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '2000-01-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

# Risk Adjustment
risk_model = risk_premium[['MKT', 'SMB', 'HML']]
risk_model = risk_model['2005':'2019']

# Univariate analysis
uni_1 = Univariate(np.array(test_data_1[['emrwd', 'momentum', 'Date_merge', 'Size']]), number=9, weight=True)
uni_1.fit()
uni_1.factor_adjustment(risk_model)
uni_1.print_summary(explicit=True)

# %% Test autocorrelation of factor momentum
import statsmodels.api as sm
import matplotlib.pyplot as plt

fac_mom = uni_1.difference(uni_1.average_by_time())[-1, :]
fac_acf = sm.tsa.stattools.acf(fac_mom, qstat=True, nlags=12)
print(fac_acf)
plt.plot(fac_acf[0])

# %% build AR model
ar = sm.tsa.arima.ARIMA(fac_mom, order=([6,9], 0, 0), trend='n')
res = ar.fit()
print(res.summary())

pre_fac = res.predict()

# %%
pre = res.predict()
plt.plot(fac_mom, label='factor')
plt.plot(pre, label='predict')
plt.legend()

# %% Test the trading volume
uni_1.summary_statistics(test_data_1[['Mnshrtrd', 'Mnvaltrd', 'Turnover']], periodic=True)
vol = uni_1.average_variable_period
vol_1 = np.log(np.mean(vol[[0,9], 1, :], axis=0))
vol_2 = np.mean(np.log(vol[[0,9], 1, :]), axis=0)
vol_3 = np.mean(np.log(vol[[0,9], 2, :]), axis=0)
vol_4 = np.log(np.mean(vol[[0,9], 3, :], axis=0))

vol_acf = sm.tsa.stattools.acf(vol_4, qstat=True, nlags=12)
print(vol_acf)
plt.plot(vol_acf[0])

# %% build AR model
ar_vol = sm.tsa.arima.ARIMA(vol_4, order=([1,4,6,7], 0, 0), trend='c')
res_vol = ar_vol.fit()
print(res_vol.summary())

pre_vol_4 = res_vol.predict()

# %% fit the fac_mom and vol_2
model = sm.OLS(pre_fac, sm.add_constant(vol_4)).fit()
print(model.summary())
model_pre = sm.OLS(pre_fac, sm.add_constant(pre_vol_4)).fit()
print(model_pre.summary())

# %% build regression model
resid_acf = sm.tsa.stattools.acf(model.resid, qstat=True, nlags=12)
print(resid_acf)
plt.plot(resid_acf[0])

ar_resid = sm.tsa.arima.ARIMA(model.resid, order=[[6,9],0,0], trend='n')
res_resid = ar_resid.fit()
print(res_resid.summary())

# %% Independent-sort Bivariate Analysis
bi_1 = Bivariate(np.array(test_data_1), number=4)
bi_1.average_by_time()
bi_1.summary_and_test()
bi_1.print_summary_by_time()
bi_1.print_summary()

# Risk Adjustment
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
bi_1_de.print_summary(explicit=True)

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

# Risk Adjustment
risk_model = risk_premium[['MKT', 'SMB', 'HML']]
risk_model = risk_model['2005':'2019']
bi_2.factor_adjustment(risk_model)
bi_2.print_summary()

# Dependent-sort Bivariate Analysis
bi_2_de = Bivariate(test_data_2, number=4)
bi_2_de.fit(conditional=True)
bi_2_de.print_summary()

# Risk Adjustment
bi_2_de.factor_adjustment(risk_model)
bi_2_de.print_summary()

# %% Persistence Analysis
from portfolio_analysis import Persistence as prse

test_data_1_per = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
model = prse(test_data_1_per[['Stkcd_merge', 'Date_merge', 'momentum']])
model.fit(lags=[1,2,3,5])
model.summary(periodic=True)
model.summary()

# %% Test autocorrelation of factor momentum
import statsmodels.api as sm
import matplotlib.pyplot as plt

fac_mom = uni_1.difference(uni_1.average_by_time())[-1, :]
fac_acf = sm.tsa.stattools.acf(fac_mom, qstat=True, nlags=12)
print(fac_acf)
plt.plot(fac_acf[0])

# %% build AR model
ar = sm.tsa.arima.ARIMA(fac_mom, order=([6,9], 0, 0), trend='n')
res = ar.fit()
print(res.summary())

# %%
