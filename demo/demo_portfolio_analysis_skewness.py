'''
Portfolio Analysis : Skewness 
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

# merge data
data = month_return[['Stkcd', 'Trdmnt', 'Mretwd']]
data['Date_merge'] = pd.to_datetime(data['Trdmnt'])
risk_premium['Date_merge'] = risk_premium.index
data = pd.merge(data, risk_premium[['MKT', 'SMB', 'HML', 'Date_merge']], on=['Date_merge'], how='outer').dropna()

# %% construct proxy variable
import numpy as np
import statsmodels.api as sm
from scipy.stats import skew

skewness = pd.Series(index=data.index, dtype=float, name='skewness')
coskewness = pd.Series(index=data.index, dtype=float, name='coskewness')
idioskewness = pd.Series(index=data.index, dtype=float, name='idioskewness')
for i in data.groupby('Stkcd'):
    row, col = np.shape(i[1])
    for j in range(row-12):
        skewness.loc[i[1].index[j+11]] = skew(i[1].iloc[j:j+12, 2])

        endog = np.array([i[1].iloc[j:j+12, 4], i[1].iloc[j:j+12, 4]**2]).T
        model_coskew = sm.OLS(i[1].iloc[j:j+12, 2], sm.add_constant(endog)).fit()
        coskewness.loc[i[1].index[j+11]] = model_coskew.params[2]
        
        model_idioskew = sm.OLS(i[1].iloc[j:j+12, 2], sm.add_constant(i[1].iloc[j:j+12, 4:7])).fit()
        idioskewness.loc[i[1].index[j+11]] = skew(model_idioskew.resid)

return_company = pd.concat([data, skewness, coskewness, idioskewness, month_return[['cap', 'Msmvttl', 'Ndaytrd', 'emrwd']]], axis=1)

# %% construct test_data for bivariate analysis
# dataset 1
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
#test_data_1 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_1 = return_company[(return_company['cap']==True)]
test_data_1 = test_data_1[['emrwd', 'Msmvttl', 'skewness', 'Date_merge']].dropna()
test_data_1 = test_data_1[(test_data_1['Date_merge'] >= '2000-01-01') & (test_data_1['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_1 = Univariate(np.array(test_data_1[['emrwd', 'skewness', 'Date_merge']]), number=9)
uni_1.summary_and_test()
uni_1.print_summary_by_time()
uni_1.print_summary()

# Independent-sort Bivariate analysis
bi_1 = Bivariate(np.array(test_data_1), number=4)
bi_1.average_by_time()
bi_1.summary_and_test()
bi_1.print_summary_by_time()
bi_1.print_summary()

# Risk Adjustment
risk_model = risk_premium[['MKT', 'SMB', 'HML']]
risk_model = risk_model.loc[test_data_1['Date_merge'].unique()]
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
# dataset 2
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_2 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_2 = test_data_2[['emrwd', 'Msmvttl', 'coskewness', 'Date_merge']].dropna()
test_data_2 = test_data_2[(test_data_2['Date_merge'] >= '2000-01-01') & (test_data_2['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_2 = Univariate(np.array(test_data_2[['emrwd', 'coskewness', 'Date_merge']]), number=9)
uni_2.summary_and_test()
uni_2.print_summary_by_time()
uni_2.print_summary()

# Independent-sort Bivariate analysis
bi_2 = Bivariate(np.array(test_data_2), number=4)
bi_2.average_by_time()
bi_2.summary_and_test()
bi_2.print_summary_by_time()
bi_2.print_summary()

# Risk Adjustment
risk_model = risk_premium[['MKT', 'SMB', 'HML']]
risk_model = risk_model.loc[test_data_1['Date_merge'].unique()]
bi_2.factor_adjustment(risk_model)
bi_2.print_summary()

# Dependent-sort Bivariate Analysis
bi_2_de = Bivariate(test_data_2, number=4)
bi_2_de.fit(conditional=True)
bi_2_de.print_summary()

# Risk Adjustment
bi_2_de.factor_adjustment(risk_model)
bi_2_de.print_summary()

# %% construct test_data for bivariate analysis
# dataset 3
from portfolio_analysis import Bivariate, Univariate
import numpy as np

# select stocks whose size is among the up 30% stocks in each month and whose trading 
# days are more than or equal to 10 days
test_data_3 = return_company[(return_company['cap']==True) & (return_company['Ndaytrd']>=10)]
test_data_3 = test_data_3[['emrwd', 'Msmvttl', 'idioskewness', 'Date_merge']].dropna()
test_data_3 = test_data_3[(test_data_3['Date_merge'] >= '2000-01-01') & (test_data_3['Date_merge'] <= '2019-12-01')]

# Univariate analysis
uni_3 = Univariate(np.array(test_data_3[['emrwd', 'idioskewness', 'Date_merge']]), number=9)
uni_3.summary_and_test()
uni_3.print_summary_by_time()
uni_3.print_summary()

# Bivariate analysis
bi_3 = Bivariate(np.array(test_data_3), number=4)
bi_3.average_by_time()
bi_3.summary_and_test()
bi_3.print_summary_by_time()
bi_3.print_summary()

# Risk Adjustment
risk_model = risk_premium[['MKT', 'SMB', 'HML']]
risk_model = risk_model.loc[test_data_1['Date_merge'].unique()]
bi_3.factor_adjustment(risk_model)
bi_3.print_summary()

# Dependent-sort Bivariate Analysis
bi_3_de = Bivariate(test_data_3, number=4)
bi_3_de.fit(conditional=True)
bi_3_de.print_summary()

# Risk Adjustment
bi_3_de.factor_adjustment(risk_model)
bi_3_de.print_summary()

# %%
