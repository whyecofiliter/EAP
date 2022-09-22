# %%
from operator import truediv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,os

sys.path.append(os.path.abspath(".."))

# %% import data
risk_premium = pd.read_hdf('.\data\\risk_premium.h5', key='data')

# %% wavelet decomposition
from time_frequency import wavelet_pricing as wlp

data = risk_premium[['MKT', 'SMB', 'HML', 'CMA']].dropna()

# %%
model1 = wlp(data['SMB'], data['MKT'])
model1.fit('haar', mode='multi', level=6)
model1.summary()

# %%
model2 = wlp(data['HML'], data['MKT'])
model2.fit('haar', mode='multi', level=6)
model2.summary()

# %%
model3 = wlp(data['CMA'], data[['MKT', 'SMB', 'HML']])
model3.fit('haar', mode='multi', level=4)
model3.summary()

# %%
import matplotlib.pyplot as plt

rmodel1 = wlp(data['SMB'], data['MKT'])
result = rmodel1.fit('haar', mode='multi', level=6, win=60, robust=True, params_only=True)

plt.plot(result[1].params[:, 1])

# %% 
import pandas as pd

data = pd.read_hdf('G:\文件\金融学\论文\谱因子模型\程序\data.hdf5', key='month_data')
risk_premium = pd.read_csv('G:\文件\金融学\论文\谱因子模型\数据\STK_MKT_THRFACMONTH.csv')
mom_pre = pd.read_csv('G:\文件\金融学\论文\谱因子模型\数据\STK_MKT_CARHARTFOURFACTORS.csv')
trade_data = pd.read_hdf('.\data\mean_filter_trade.h5', key='data')

# %%
data['ClosePrice'] = pd.to_numeric(data['ClosePrice']).astype('float')

# %%
data['month_return'] = data.groupby(['Syb'])['ClosePrice'].diff(1) / data.groupby(['Syb'])['ClosePrice'].shift(1)
data['next_month_return'] = data.groupby(['Syb'])['month_return'].shift(-1)
data_shift = data.groupby(['Syb']).shift(1).dropna()

# %%
import matplotlib.pyplot as plt

plt.plot(data['next_month_return'].loc['5'])

# %%
mkt = risk_premium[['MarkettypeID', 'TradingMonth', 'RiskPremium1']]
mkt = mkt[mkt['MarkettypeID']=='P9707']
mkt.index = pd.to_datetime(mkt['TradingMonth'])
mkt = mkt['RiskPremium1']

smb = risk_premium[['MarkettypeID', 'TradingMonth', 'SMB1']]
smb = smb[smb['MarkettypeID']=='P9707']
smb.index = pd.to_datetime(smb['TradingMonth'])
smb = smb['SMB1']

hml = risk_premium[['MarkettypeID', 'TradingMonth', 'HML1']]
hml = hml[hml['MarkettypeID']=='P9707']
hml.index = pd.to_datetime(hml['TradingMonth'])
hml = hml['HML1']

mom = mom_pre[['MarkettypeID', 'TradingMonth', 'UMD1']]
mom = mom[mom['MarkettypeID']=='P9707']
mom.index = pd.to_datetime(mom['TradingMonth'])
mom = mom['UMD1']

# %%
from time_frequency import wavelet_pricing as wlp

def wlt(x, level_num, base):
    import pandas as pd
    import numpy as np
    #level = 0
    return_series = pd.Series(index=x.index, dtype=float)

    temp = pd.merge(x, base, left_index=True, right_index=True).dropna()
    if len(temp) % 2 == 0:
        model = wlp(temp.iloc[:, 0], temp.iloc[:, 1])
        d = 0
    else: 
        #temp = temp.iloc[:-1, :]
        temp.loc[len(temp)] = temp.iloc[-1, :]
        model = wlp(temp.iloc[:, 0], temp.iloc[:, 1])
        d = 1

    if len(temp) >= 120:
        result = model.fit('haar', mode='multi', level=6, win=60, robust=True, params_only=True)
        if d == 0:
            return_series[temp.index] = result[level_num].params[:, 1]
        else:
            return_series[temp.index[:-1]] = result[level_num].params[:-1, 1]
        
        return return_series
    else:
        return return_series

data['beta0'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=0, base=mkt.shift(1).dropna())
data['beta1'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=1, base=mkt.shift(1).dropna())
data['beta2'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=2, base=mkt.shift(1).dropna())
data['beta3'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=3, base=mkt.shift(1).dropna())
data['beta4'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=4, base=mkt.shift(1).dropna())
data['beta5'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=5, base=mkt.shift(1).dropna())
data['beta6'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=6, base=mkt.shift(1).dropna())

data['smb_beta0'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=0, base=smb.shift(1).dropna())
data['smb_beta1'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=1, base=smb.shift(1).dropna())
data['smb_beta2'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=2, base=smb.shift(1).dropna())
data['smb_beta3'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=3, base=smb.shift(1).dropna())
data['smb_beta4'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=4, base=smb.shift(1).dropna())
data['smb_beta5'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=5, base=smb.shift(1).dropna())
data['smb_beta6'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=6, base=smb.shift(1).dropna())

# hml
data['hml_beta0'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=0, base=hml.shift(1).dropna())
data['hml_beta1'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=1, base=hml.shift(1).dropna())
data['hml_beta2'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=2, base=hml.shift(1).dropna())
data['hml_beta3'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=3, base=hml.shift(1).dropna())
data['hml_beta4'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=4, base=hml.shift(1).dropna())
data['hml_beta5'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=5, base=hml.shift(1).dropna())
data['hml_beta6'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=6, base=hml.shift(1).dropna())

# mom
data['mom_beta0'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=0, base=mom.shift(1).dropna())
data['mom_beta1'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=1, base=mom.shift(1).dropna())
data['mom_beta2'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=2, base=mom.shift(1).dropna())
data['mom_beta3'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=3, base=mom.shift(1).dropna())
data['mom_beta4'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=4, base=mom.shift(1).dropna())
data['mom_beta5'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=5, base=mom.shift(1).dropna())
data['mom_beta6'] = data['month_return'].groupby(['Symbol']).apply(wlt, level_num=6, base=mom.shift(1).dropna())

# %% beta 0
from portfolio_analysis import Univariate

data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'beta0','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 1
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'beta1','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']
# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 2
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'beta2','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 3
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'beta3','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'beta4','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'beta5','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'beta6','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
def original(x, base):
    import pandas as pd
    import numpy as np
    from statsmodels.regression.rolling import RollingOLS
    import statsmodels.api as sm
    
    return_series = pd.Series(index=x.index, dtype=float)

    temp = pd.merge(x, base, left_index=True, right_index=True).dropna()

    if len(temp) >= 120:
        model = RollingOLS(temp.iloc[:, 0], sm.add_constant(temp.iloc[:, 1]), window=60)
        result = model.fit(method='pinv', params_only=True)
        #print(result.params.iloc[:,1])
        return_series[temp.index] = result.params.iloc[:, 1]
        
        return return_series
    else:
        return return_series

data['beta'] = data['month_return'].groupby(['Symbol']).apply(original, base=mkt.shift(1).dropna())

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'beta','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()


# %%
'''
SMB
'''

# % beta 0
from portfolio_analysis import Univariate

data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'smb_beta0','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 1
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'smb_beta1','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']
# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 2
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'smb_beta2','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 3
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'smb_beta3','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'smb_beta4','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'smb_beta5','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'smb_beta6','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['smb_beta'] = data['month_return'].groupby(['Symbol']).apply(original, base=smb.shift(1).dropna())

data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'smb_beta','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %%
from portfolio_analysis import Univariate

model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 0
'''
HML
'''

from portfolio_analysis import Univariate

data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'hml_beta0','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 1
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'hml_beta1','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']
# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 2
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'hml_beta2','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 3
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'hml_beta3','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'hml_beta4','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'hml_beta5','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'hml_beta6','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['hml_beta'] = data['month_return'].groupby(['Symbol']).apply(original, base=hml.shift(1).dropna())

data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'hml_beta','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
from fama_macbeth import Fama_macbeth_regress

return_company = data[['next_month_return', 'smb_beta0', 'smb_beta1', 'smb_beta2', 'smb_beta3', 'smb_beta4', 'smb_beta5', 'smb_beta6', 'TradingMonth']].dropna()
return_company= return_company[return_company['TradingMonth']>='2000']

model = Fama_macbeth_regress(return_company)
result = model.fit(add_constant=True)
model.summary()

# %%
return_company = data[['next_month_return', 'hml_beta0', 'hml_beta1', 'hml_beta2', 'hml_beta3', 'hml_beta4', 'hml_beta5', 'hml_beta6', 'TradingMonth']].dropna()
return_company= return_company[return_company['TradingMonth']>='2000']

model = Fama_macbeth_regress(return_company)
result = model.fit(add_constant=True)
model.summary()

# %%
return_company = data[['next_month_return', 'beta0', 'beta1', 'beta2', 'beta3', 'beta4', 'beta5', 'beta6', 'TradingMonth']].dropna()
return_company= return_company[return_company['TradingMonth']>='2000']

model = Fama_macbeth_regress(return_company)
result = model.fit(add_constant=True)
model.summary()

# %%
return_company = data[['next_month_return', 'beta', 'smb_beta', 'hml_beta', 'TradingMonth']].dropna()
return_company= return_company[return_company['TradingMonth']>='2000']

model = Fama_macbeth_regress(return_company)
result = model.fit(add_constant=True)
model.summary()

# %%
'''
MOM
'''

from portfolio_analysis import Univariate

data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'mom_beta0','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 1
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'mom_beta1','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']
# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 2
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'mom_beta2','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %% beta 3
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'mom_beta3','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'mom_beta4','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'mom_beta5','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'mom_beta6','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
data['mom_beta'] = data['month_return'].groupby(['Symbol']).apply(original, base=mom.shift(1).dropna())

data['Size'] = np.log(pd.to_numeric(data['CirculatedMarketValue']))
return_company = data[['next_month_return', 'mom_beta','TradingMonth', 'Size']].dropna()
return_company = return_company[return_company['TradingMonth']>='2000']

# %
model = Univariate(return_company, number=9, weight=True)
model.fit()
model.print_summary_by_time()
model.print_summary()

# %%
from fama_macbeth import Fama_macbeth_regress

return_company = data[['next_month_return', 'mom_beta0', 'mom_beta1', 'mom_beta2', 'mom_beta3', 'mom_beta4', 'mom_beta5', 'mom_beta6', 'TradingMonth']].dropna()
return_company= return_company[return_company['TradingMonth']>='2000']

model = Fama_macbeth_regress(return_company)
result = model.fit(add_constant=True)
model.summary()

# %%











