'''
Fama-Macbeth Regerssion Fama-Macbeth 回归
Following Two step 分为两步
1. specify the model and take cross-sectional regression 确定模型，进行截面回归 
2. take the time-series average of regress coefficient 对系数在时间序列上取平均

For more academic reference: 
Empirical Asset Pricing: The Cross Section of Stock Returns. Bali, Engle, Murray. 2016.
'''


from numpy.core.defchararray import add
from statsmodels.tools.tools import add_constant

from .portfolio_analysis import Bivariate


class Fama_macbeth_regress():
    '''
    fama_macbeth_regress follwing two steps
    Need Package: numpy,statsmodels,scipy,prettytable
    '''
    def __init__(self, sample):
        '''
        input: sample: data used for analysis 用于回归数据
        The strucure of the sample：
        the first column is dependent variable/ Test Portfolio Return
        the second to the last-1 columns is independent variable/ factor loadings
        the last coolumn is time label
        '''
        import numpy as np

        if type(sample).__name__ == 'DataFrame':
            self.sample = np.array(sample)
            self._input_type = 'DataFrame'
            self._columns = list(sample.columns)
        elif type(sample).__name__ == 'ndarray':
            self.sample = sample
            self._input_type = 'ndarray'
        # self.time_series_average()
    
    def divide_by_time(self, sample):
        '''
        devide the sample by time into the groups 将样本按时间分组
        output: groups by time： sample groups split by time 按时间分成的样本组
        '''
        import numpy as np
        
        # extracting the sample time label 提取样板时间戳
        time = np.sort(np.unique(sample[:,-1]))
        # group the sample by time 按时间对样本中的数据进行分类
        groups_by_time = list()
        for i in range(len(time)):
            groups_by_time.append(sample[np.where(sample[:,-1]==time[i])])
        
        # return grouped sample 返回被分类的样本组
        return groups_by_time
            
    def cross_sectional_regress(self, add_constant=True, normalization=True):
        '''
        The #1 step of Fama-Macbeth Regression Fama-Macbeth 回归第一步
        take the cross_sectional regress for each period 在每个时间段分别进行截面数据回归
        input : **kwargs: 
                add_constant: whether add_constant when take CS regression 截面回归时是否添加常数
        output: parameters: 
                regression coefficient/factor risk premium: 回归参数/因子风险溢价
                rows: group coefficient 每组回归参数
                columns: regression variable 各因子风险溢价（回归参数）
                tvalue: t value for the coefficient 回归参数的t值
                rsquare: R-square 回归R方
                adjrsq: Adjust R-square 调整回归R方
                n: sample number in each group 每组的样本个数
        '''

        import statsmodels.api as sm
        import numpy as np
        
        # get sample shape 获取样本行列数
        r,c = np.shape(self.sample)
        # get sample groups split by time 获取分组样本 
        groups = self.divide_by_time(self.sample)
        # initiate the variable 初始化变量
        
        if add_constant == True :
            params = np.zeros((len(groups), c-1))
            tvalue = np.zeros((len(groups), c-1))
        elif add_constant == False :
            params = np.zeros((len(groups), c-2))
            tvalue = np.zeros((len(groups), c-2))
        rsquare = list()
        adjrsq = list()
        n = list()
        
        # cross section regression by groups 分组截面回归
        for i in range(len(groups)) :
            # group size 组的大小
            r, c = np.shape(groups[i])
            temp_group = groups[i]
            # regression 回归
            if add_constant == True :
                if normalization == True:
                    temp_endog = (temp_group[:, 0]-np.mean(temp_group[:, 0]))/np.std(temp_group[:, 0])
                    temp_exog = (temp_group[:, 1:-1]-np.mean(temp_group[:, 1:-1], axis=0))/(np.var(temp_group[:, 1:-1], axis=0)**0.5)
                    result = sm.OLS(temp_endog.astype(float), sm.add_constant(temp_exog.astype(float))).fit()
                
                elif normalization == False:
                    result = sm.OLS(temp_group[:, 0].astype(float), sm.add_constant(temp_group[:, 1:-1].astype(float))).fit()    
                
                self.constant = True
            elif add_constant == False :
                if normalization == True:
                    temp_endog = (temp_group[:, 0]-np.mean(temp_group[:, 0]))/np.std(temp_group[:, 0])
                    temp_exog = (temp_group[:, 1:-1]-np.mean(temp_group[:, 1:-1], axis=0))/(np.var(temp_group[:, 1:-1], axis=0)**0.5)
                    result = sm.OLS(temp_endog.astype(float), temp_exog.astype(float)).fit()
  
                elif normalization == False:
                    result = sm.OLS(temp_group[:, 0].astype(float), temp_group[:, 1:-1].astype(float)).fit()
                
                self.constant = False
            # params of each group
            params[i,:] = result.params
            # t_vlaue of each group
            tvalue[i,:] = result.tvalues
            # R square of each group
            rsquare.append(result.rsquared)
            # adjust R square of each group
            adjrsq.append(result.rsquared_adj)
            # sample number of each group
            n.append(r)
        
        return params,tvalue,rsquare,adjrsq,n
    
    def time_series_average(self,**kwargs) :
        '''
        The #2 step of Fama-French regression Fama-French 回归第二步 
        Time series average for cross section regression  对截面数据在时间上取均值
        '''
        import scipy.stats as sts
        import numpy as np
        
        # the params of cross-section regression 截面回归相关参数 
        self.result_cross = self.cross_sectional_regress(**kwargs)
        # the average of the regression coefficeint 参数均值
        self.coefficient_average = np.mean(self.result_cross[0], axis=0)
        # t_test for the regression coefficient 回归系数T检验
        self.tvalue = sts.ttest_1samp(self.result_cross[0], 0.0, axis=0)[0]
        # the average of the R square R方均值
        self.rsquare_average = np.mean(self.result_cross[2])
        # the average of the adjust R square 调整R方均值
        self.adjrsq_average = np.mean(self.result_cross[3])
        # the average sample numbers in each group 各组样本均值
        self.n_average = np.mean(self.result_cross[4])
        # print the result 打印结果
        print('para_average:', self.coefficient_average)
        print('tvalue:', self.tvalue)
        print('R:', self.rsquare_average)
        print('ADJ_R:', self.adjrsq_average)
        print('sample number N:', self.n_average)
        # print(np.vstack([self.para_average,self.tvalue,np.array(self.rsquare_average)]))
    
    def fit(self, **kwargs) :
        '''
        Fit model 估计模型
        run function: time_series_average 运行函数： time_series_average()
        '''

        self.time_series_average(**kwargs) 

    def summary_by_time(self):
        '''
        summary the cross-section result of each time 总结每一时刻的结果
        package needed : prettytable
        '''
        from prettytable import PrettyTable
        import numpy as np
        
        r,c = np.shape(self.sample)
        time = np.sort(np.unique(self.sample[:, -1]))
        result_cross = self.result_cross
        
        table = PrettyTable()
        table.add_column('Year', time)
        table.add_column('Param', result_cross[0])
        table.add_column('R Square', np.round(result_cross[2], decimals=2))
        table.add_column('Adj R Square', np.round(result_cross[3], decimals=2))
        table.add_column('Sample Number', result_cross[4])
        
        # coefficient decimals: 3, R & adj_R decimals: 2
        np.set_printoptions(formatter={'float':'{:0.3f}'.format})
        print(table)
        
    def summary(self,charactername=None) :
        '''
        summary the time-series average 总结时间序列平均
        '''
        from prettytable import PrettyTable        
        import numpy as np
        
        r,c = np.shape(self.sample)
        table = PrettyTable()
        if charactername == None :
            charactername = list()
            for i in range(c-2):
                charactername.extend(str(i+1))
        if self.constant == True and self._input_type == 'ndarray' :
            table.field_names = ['Intercept','Intercept Tvalue'] + ['Param','Param Tvalue'] + ['Average R','Average adj R','Average n']
            table.add_row([np.around(self.coefficient_average[0], decimals=4), np.around(self.tvalue[0], decimals=3), self.coefficient_average[1:],\
                           self.tvalue[1:], np.around(self.rsquare_average, decimals=3), np.around(self.adjrsq_average, decimals=3),\
                           self.n_average])
        elif self.constant == True and self._input_type == 'DataFrame' :
            table.field_names = ['Intercept'] + self._columns[1:-1] + ['Average R','Average adj R','Average n']
            table.add_row(np.around([self.coefficient_average[0]] + list(self.coefficient_average[1:]) + [np.around(self.rsquare_average, decimals=3), np.around(self.adjrsq_average, decimals=3),\
                           np.around(self.n_average, decimals=2)], decimals=4))
            table.add_row([np.around(self.tvalue[0], decimals=3)] + list(np.around(self.tvalue[1:], decimals=3)) + ['-', '-', '-'])

        elif self.constant == False and self._input_type == 'ndarray' :
            table.field_names = ['Param','Param Tvalue']+['Average R','Average adj R','Average n']
            table.add_row([self.coefficient_average, self.tvalue, np.around(self.rsquare_average, decimals=3), np.around(self.adjrsq_average, decimals=3),\
                           self.n_average])
        elif self.constant == False and self._input_type == 'DataFrame' :
            table.field_names = self._columns[1:-1] + ['Average R','Average adj R','Average n']
            table.add_row(np.around(list(self.coefficient_average), decimals=4) + [np.around(self.rsquare_average, decimals=3), np.around(self.adjrsq_average, decimals=3),\
                           np.around(self.n_average, decimals=2)])
            table.add_row(list(np.around(self.tvalue, decimals=3)) + ['-', '-', '-'])

        
        print('\n')
        print(table)
        
class Factor_mimicking_portfolio():
    '''
    Factor_mimicking_portfolio:
    Following Fama-French(1993), generate factor mimicking portfolio and calculate factor risk premium.
    '''

    def __init__(self, sample, perc_row=[0, 50, 100], perc_col=[0, 30, 70, 100], weight=True):
        import numpy as np
        
        self.sample = sample
        self.perc_row = perc_row
        self.perc_col = perc_col
        self.weight = weight
        if type(sample).__name__ == 'DataFrame':
            self._time = np.sort(np.unique(sample.iloc[:, 3])) 
        elif type(sample).__name__ == 'ndarray':
            self._time = np.sort(np.unique(sample[:, 3]))

    def portfolio_return_time(self):
        '''
        Construct portfolio and calculate the average return and difference matrix 
        '''
        from .portfolio_analysis import Bivariate
        
        bi = Bivariate(self.sample, perc_row=self.perc_row, perc_col=self.perc_col, weight=self.weight)
        diff = bi.difference(bi.average_by_time())
        
        return diff
    
    def portfolio_return(self):
        '''
        Construct factor risk premium
        '''
        import numpy as np
        import pandas as pd
        
        diff = self.portfolio_return_time()
        r, c, n = np.shape(diff)
        
        time = self._time

        return_row = pd.Series(np.mean(diff[-1, :c, :], axis=0), index=pd.to_datetime(time))
        return_col = pd.Series(np.mean(diff[:r, -1, :], axis=0), index=pd.to_datetime(time))
        
        return return_row, return_col
    
    def portfolio_return_horizon(self, period, ret=False):
        '''
        Construct horizon pricing factor
        input :
            period (int): period
        '''
        import numpy as np
        import pandas as pd

        diff = self.portfolio_return_time()
        r, c, n =np.shape(diff)

        time = self._time

        return_col_up = pd.Series(np.mean(diff[:-1, 0, :], axis=0), index=pd.to_datetime(time))
        return_col_down = pd.Series(np.mean(diff[:-1, -2, :], axis=0), index=pd.to_datetime(time))
        return_row_up = pd.Series(np.mean(diff[0, :c, :], axis=0), index=pd.to_datetime(time))
        return_row_down = pd.Series(np.mean(diff[-2, :c, :], axis=0), index=pd.to_datetime(time))

        return_row_up_plus = np.log(return_row_up + 1)
        return_row_down_plus = np.log(return_row_down + 1)
        return_col_up_plus = np.log(return_col_up + 1)
        return_col_down_plus = np.log(return_col_down + 1)
        
        if period > 1:
            return_row_up_multi = return_row_up_plus.rolling(window=period).sum()
            return_row_down_multi = return_row_down_plus.rolling(window=period).sum()
            return_col_up_multi = return_col_up_plus.rolling(window=period).sum()
            return_col_down_multi = return_col_down_plus.rolling(window=period).sum()
            
            if ret == False:
                return_row_multi = np.exp(return_row_down_multi) - np.exp(return_row_up_multi)
                return_col_multi = np.exp(return_col_down_multi) - np.exp(return_col_up_multi)
            elif ret == True:
                return_row_multi = return_row_down_multi - return_row_up_multi
                return_col_multi = return_col_down_multi - return_col_up_multi

        elif period == 1:
            if ret == False:
                return_row_multi = np.exp(return_row_down_plus) - np.exp(return_row_up_plus)
                return_col_multi = np.exp(return_col_down_plus) - np.exp(return_col_up_plus)
            elif ret == True:
                return_row_multi = return_row_down_plus - return_row_up_plus
                return_col_multi = return_col_down_plus - return_col_up_plus

        return return_row_multi, return_col_multi

