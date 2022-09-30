'''
Portfolio Analysis
This module is used for portfolio analysis by 4 steps:
    1. select breakpoints
    2. distribute the assets into groups
    3. calculate the average and difference of groups
    4. summary the result
'''

class ptf_analysis():
    def __init__(self) :
        pass
    
    def select_breakpoints(self, character, number, perc=None, percn=None):
        '''
        This function, corresponding to the step 1,  selects the breakpoints of the sample.
        input :  
        character (ndarray/Series):  The asset characteristics by which the assets are grouped. 
        number (int): The number of the breakpoints and the number of interval is *number+1*. Once the number is given, the assets would be grouped into number+1 groups by the average partition of the asset characteristic.  
        perc (list/ndarray):  perc are percentile points of the characteristics. Once it is set, then  *number* is overwritten. eg. perc = [0, 30, 70, 100] represents the percentiles at 0%, 30%, 70%, 100%.
        percn (list/ndarray): percn are percentiles of the characteristics. eg. the characteristics are divided by NYSE breakpoints.

        output : 
        breakpoint : The breakpoints are the percentiles of the asset characteristic, ranging from 0% to 100%, whose in length is number+2.

        Example
        from EAP.portfolio_analysis import Ptf_analysis as ptfa
        import matplotlib.pyplot as plt
        import numpy as np

        # generate characteristics
        character = np.random.normal(0, 100, 10000)
        # generate breakpoint
        breakpoint = ptfa().slect_breakpoint(character=character, number=9)
        print('Breakpoint:', breakpoint)
        # comapre with the true breakpoint
        for i in np.linspace(0, 100, 11):
            print('True breakpoint', i, '%:', np.percentile(character,i))

        '''
        import numpy as np
        # create percentiles
        if perc == None: 
            perc = np.linspace(0, 100, number+2, dtype=int)
        elif perc is not None:
            perc = perc

        if percn is None:    
            breakpoint = np.percentile(character, perc, axis=0)
        elif percn is not None :
            breakpoint = np.array(percn)

        return breakpoint
    
    def distribute(self, character, breakpoint):
        '''
        This function, corresponding to the step 2, distributes the assets into groups by characteristics  grouped by the breakpoint.
        input :  
        character (ndarray/Series): The characteristic by which the assets are grouped.
        breakpoint (list/array): The breakpoints of the characteristic.
        
        output : 
        label (ndarray): an array containing the group number of each asset.

        Example:
        # continue the previous code
        # generate the groups
             
        print('The Label of unique value:\n', np.sort(np.unique(ptfa().distribute(character, breakpoint))))
        # plot the histogram of the label 
        # each group have the same number of samples
        plt.hist(ptfa().distribute(character, breakpoint))
        label = ptfa().distribute(character, breakpoint)[:, 0]
        # print the label
        print('Label:\n', ptfa().distribute(character, breakpoint))
        '''
        import numpy as np
        
        r = len(character)
        label = np.zeros((r, 1))
        for i in range(len(breakpoint) - 1):
            label[np.where((character >= breakpoint[i]) & (character < breakpoint[i+1]) & (i+1 < len(breakpoint) - 1))] = i
            label[np.where((character >= breakpoint[i]) & (character <= breakpoint[i+1]) & (i+1 == len(breakpoint) - 1))] = i
        return label
    
    def average(self, sample_return, label, cond='uni', weight=None):
        '''
        This function, corresponding to the step 3, calculates the average return for each group.
        input :
        sample_return (ndarray): The return of each asset.
        label (ndarray): The group label of each asset.
        cond (str): If univariate analysis, then cond = 'uni'; if bivariate analysis, then cond = 'bi'.
        weight (None): The weight to calculate the weighted average group return.   

        output :
        average_return (ndarray): The average return of each group.

        Example:
        # continue the previous code
        # generate the future sample return
        sample_return = character/100 + np.random.normal()
        ave_ret = ptfa().average(sample_return, label)
        # print the groups return
        print('average return for groups:\n', ave_ret)
        '''
        import numpy as np

        if cond == 'uni' :
            # the whole group label, eg. 10 group lables: [1,2,3,4,5,6,7,8,9,10]  
            temp_label = np.sort(np.unique(label))
            # the average return of each group
            average_return = np.zeros((len(temp_label), 1))
            # calculate the average return of each group through matching the sample_return's 
            # label with the group label and the sample_return is Forecasted Return
            for i in range(len(temp_label)):
                if weight is None:
                    average_return[i, 0] = np.mean(sample_return[np.where(label==temp_label[i])])
                else:
                    standard_weight = weight[np.where(label==temp_label[i])] / np.sum(weight[np.where(label==temp_label[i])])
                    average_return[i, 0] = np.sum(standard_weight * sample_return[np.where(label==temp_label[i])])

            # return average value of each group
            return average_return
        
        if cond == 'bi' :
            # the whole group label, eg. 10 group labels: [1,2,3,4,5,6,7,8,9,10]
            temp_label_row = np.sort(np.unique(label[0]))
            temp_label_col = np.sort(np.unique(label[1]))
            # the average return of each group
            average_return = np.zeros((len(temp_label_row), len(temp_label_col)))
            # calculate the average return of each group through matching the sample_return's
            # label with the group label and the sample_return is Forecasted Return
            for i in range(len(temp_label_row)):
                for j in range(len(temp_label_col)):
                    if weight is None:
                        average_return[i, j] = np.mean(sample_return[np.where((label[0]==temp_label_row[i])&(label[1]==temp_label_col[j]))])
                    else:
                        standard_weight = weight[np.where((label[0]==temp_label_row[i])&(label[1]==temp_label_col[j]))] / np.sum(weight[np.where((label[0]==temp_label_row[i])&(label[1]==temp_label_col[j]))])
                        average_return[i, j] = np.sum(standard_weight * sample_return[np.where((label[0]==temp_label_row[i])&(label[1]==temp_label_col[j]))])
            
            # return average value of each group
            return average_return
    
    def statistics(self, variable, label, func, cond='uni'):
        '''
        This function is for summary statistics of groups.
        input :
        variable (ndarray): The variables of the groups.
        label (ndarray): The label of the groups for each stock.
        func (function): The operations to variables in each group, like numpy.mean, numpy.sum, etc.
        cond (str):  If univariate analysis, then cond = 'uni'; if bivariate analysis, then cond = 'bi'.

        output:
        average_statistics (ndarray):* The statistics of each group.
        '''
        import numpy as np
        
        # the whole group label, eg. 10 group labels: [1,2,3,4,5,6,7,9,10]
        temp_label = np.sort(np.unique(label))
        # initialize average value of variable
        try: 
            r, c = np.shape(variable)
        except:
            r = len(variable)
            c = 0
        
        if c  == 0 :
            average_statistics = np.zeros((len(temp_label)))
            # calculate the average variable value of each group through matching the variable's 
            # label with the group label.
            for i in range(len(temp_label)):
                average_statistics[i] = func(variable[np.where(label==temp_label[i])]) 

            return average_statistics
        else :
            average_statistics = np.zeros((len(temp_label), c))

            for i in range(len(temp_label)):
                for j in range(c):
                    average_statistics[i, j] = func(variable[np.where(label==temp_label[i]), j])

            return average_statistics
    
    def create_breakpoint(self, data, number, perc=None):
        '''
        This function is for creating breakpoints. In many researches, the special breakpoints are needed like NYSE breakpoints in common. This function is designed for this demand.

        input :
        data (ndarray/DataFrame): The characteristics for partition. 
            The structure is 
                first row to last two row: character.
                last row : time index.
        number (int): The number of breakpoints.
        perc (array/list): The percentile points of breakpoints.

        output :
        breakpoints array (list): The breakpoints of characteristics.

        Example :
        # generate variables
        character=np.random.normal(0,1,20*3000)
        # generate future return
        ret=character*-0.5+np.random.normal(0,1,20*3000)
        # create sample containing future return, character, time
        sample=np.array([character,year]).T
    
        breakpoints = ptfa().create_breakpoint(data=sample, number=4)
        print(breakpoints)
        '''
        import numpy as np
       
        if type(data).__name__ == 'DataFrame':
            data = np.array(data)
        
        r, c =np.shape(data)

        def divide_by_time(data):        
            time = np.sort(np.unique(data[:, -1]))
            groups_by_time = list()
            for i in range(len(time)):
                groups_by_time.append(data[np.where(data[:, -1]==time[i])])
        
            return groups_by_time
        
        # get the sample groups by time 得到按时间分组的样本
        groups_time = divide_by_time(data)
        # generate table of  average return for group and time 生成组和时间的表格
        # Rows: groups 行： 组  
        # Columns: Time 列： 时间
        if perc is not None:
            number = len(perc) - 2 
        
        breakpoint_array = list()
        for i in range(c-1):
            breakpoint_array.append(np.zeros((len(groups_time), number+2)))
        
        for i in range(len(groups_time)):
            # for each time, a group exist
            group = groups_time[i]
            # for each time, generate breakpoint
            for j in range(c-1):
                breakpoint_array[j][i, :] = self.select_breakpoints(character = group[:, j], number = number, perc = perc)
        
        return breakpoint_array


class Univariate(ptf_analysis):
    '''
    This class is designed for univariate portfolio analysis.
    '''
    def __init__(self, sample):
        '''
        input: sample: the samples to be analyzed 
            sample usually contain the future return, characteristics, time 
            the DEFAULT settting is the First column is the forecast return and 
            the second colunm is the characteristic
            the third colunm or the index is time label
            
            number: the breakpoint number
            perc: the breakpoint percentiles
            factor: risk adjust factor
            maxlag: maximum lag for Newey-West adjustment
        '''
        import numpy as np

        if type(sample).__name__ == 'DataFrame':
            self.sample = np.array(sample)
            self._sample_type = 'DataFrame'
            self._columns = list(sample.columns)
        elif type(sample).__name__ == 'ndarray':
            self.sample = sample
            self._sample_type = 'ndarray'
        else:
            raise IOError
             
    def divide_by_time(self, sample):
        '''
        This function groups the sample by time.
        split the sample by time into groups  
        output : 
        groups_by_time (list): The samples group by time.
        '''
        import numpy as np
        
        time = np.sort(np.unique(sample[:, 2]))
        groups_by_time = list()
        for i in range(len(time)):
            groups_by_time.append(sample[np.where(sample[:, 2]==time[i])])
        
        return groups_by_time
            
    def average_by_time(self):
        '''
        average of the groups at each time point
        This function, using the sample group by time from function *divide_by_time*, 
        groups the sample by the characteristic, 
        and then calculate average return of each group samples at every time point. 
        
        Example:
        import numpy as np
        from portfolio_analysis import Univariate as uni
    
        # generate time 
        year=np.ones((3000,1),dtype=int)*2020
        for i in range(19):
            year=np.append(year,(2019-i)*np.ones((3000,1),dtype=int))
    
        # generate character
        character=np.random.normal(0,1,20*3000)
        # generate future return
        ret=character*-0.5+np.random.normal(0,1,20*3000)
        # create sample containing future return, character, time
        sample=np.array([ret,character,year]).T
        # initializ the univariate object
        exper=uni(sample,9)
        # average by time
        data=exper.average_by_time()
        print(data)
        '''
        import numpy as np
        # get the sample groups by time 得到按时间分组的样本
        groups_time = self.divide_by_time(self.sample)
        # generate table of  average return for group and time 生成组和时间的表格
        # Rows: groups 行： 组  
        # Columns: Time 列： 时间
        average_group_time = np.zeros((self.number+1, len(groups_time)))
        self._label = list()
        for i in range(len(groups_time)):
            # for each time, a group exist
            group = groups_time[i]
            # for each time, generate breakpoint
            if self.percn is None:
                breakpoint = super().select_breakpoints(character = group[:, 1], number = self.number, perc = self.perc)
            elif self.percn is not None:
                breakpoint = super().select_breakpoints(character = group[:, 1], number = self.number, percn = self.percn[i, :])
            # for each time, generate label
            label = super().distribute(group[:, 1], breakpoint)
            # for each group in each time, calculate the average future return
            if self._weight == False:
                try:
                    average_group_time[:, i] = super().average(group[:, 0], label[:, 0]).reshape((self.number+1))
                except:
                    average_group_time[:, i] = np.full([self.number+1], np.nan)
            elif self._weight == True:
                try:
                    average_group_time[:, i] = super().average(group[:, 0], label[:, 0], weight=group[:, 3]).reshape((self.number+1))
                except:
                    average_group_time[:, i] = np.full([self.number+1], np.nan)
            else: return IOError

            self._label.append(label[:, 0])
            
        # return the Table
        # Rows: groups in each time
        # Columns: Time 
        return average_group_time

    def difference(self,average_group):
        '''
        This functions calculates the difference of group return, which, in detail, 
        is the last group average return minus the first group average return. 
        input :
        average_group (ndarray): The average return of groups by each characteristic-time pair.

        output :
        result (ndarray): The matrix added with the difference of average group return.
        '''
        import numpy as np
        
        diff = average_group[-1, :] - average_group[0, :]
        diff = diff.reshape((1, len(diff)))        
        result = np.append(average_group, diff, axis=0)
        self.diff = diff

        return result

    def summary_and_test(self) :
        '''
        This function summarizes the result and take t-test.
        output : 
        self.average (ndarray): The average of the portfolio return across time.
        self.ttest (ndarray): The t-value of the portfolio return across time.
        '''
        import numpy as np
        from scipy import stats
        from .adjust import newey_west_t
        
        self.result = self.difference(self.average_by_time())
        r, c = np.shape(self.result)
        self.average = np.nanmean(self.result, axis=1)
        if self.maxlag == 0 :
            self.ttest = stats.ttest_1samp(self.result, 0.0, axis=1, nan_policy='omit')
        else:
            self.ttest = np.zeros((2, r))

            for i in range(r):
                temp_result = self.result[i, :]
                temp_result = temp_result[~np.isnan(temp_result)]
                temp_one = np.ones((len(temp_result), 1))
                tvalue, pvalue = newey_west_t(y=temp_result, X=temp_one, J=self.maxlag, constant=False)
                self.ttest[0, i] = tvalue
                self.ttest[1, i] = pvalue

        # if there is a facotr adjustment, then normal return result plus the anomaly result
#        if self.factor is not None  :
#            self.alpha, self.alpha_tvalue = self.factor_adjustment(self.result)
#            return self.average, self.ttest, self.alpha, self.alpha_tvalue

        return self.average, self.ttest
    
    def fit(self, number, perc=None, percn=None, maxlag=12, weight=False) :
        '''
        This function fit the model
        input :
        number (int): The breakpoint number.
        perc (list or array): The breakpoint percentile points.
        percn (list or array): The breakpoint percentiles.
        maxlag (int): The maximum lag for Newey-West adjustment.
        weight (boolean): If the weighted return is chosen, then weight is True. The **DEFAULT** is False.
        '''
        import numpy as np

        if perc is not None :
            self.number = len(perc) - 2
        elif percn is not None:
            r, c = np.shape(percn)
            self.number = c - 2
        else:
            self.number = number

        self.perc = perc
        self.percn = percn
        self.maxlag = maxlag
        self._factor = None
        self._time = np.sort(np.unique(self.sample[:, 2])) 
        self._weight = weight

        self.summary_and_test()

    def factor_adjustment(self, factor):
        '''
        This function calculates the group return adjusted by risk factors.
        input :
        factor (ndarray or DataFrame): The return table with difference sequence.

        output :
        alpha (array): The anomaly
        ttest (array): The t-value of the anomaly.

        Example:
        # generate factor
        factor=np.random.normal(0,1.0,(20,1))
        exper=uni(sample,9,factor=factor,maxlag=12)
        # print(exper.summary_and_test()) # if needed
        exper.print_summary()
        '''
        import statsmodels.api as sm
        import numpy as np
        import pandas as pd
        
        if type(factor).__name__ == 'DataFrame':
            self._factor = factor
        elif type(factor).__name__ == 'ndarray':
            self._factor = pd.DataFrame(factor)

        # take the inverse of the Table with difference
        # Rows: Time
        # Columns: Groups Return
        table = self.result.T
        row, col = np.shape(table)
        row_fac, col_fac = np.shape(self._factor)
        # generate anomaly: alpha 
        # generate tvalues: ttest
        alpha = np.zeros((col, 1))
        ttest = np.zeros((col, 1))
        beta = np.zeros((col, col_fac))
        beta_ttest = np.zeros((col, col_fac))
        r2 = np.zeros((col, 1))

        # factor adjusment
        for i in range(col):
            time = np.sort(np.unique(self.sample[:, 2]))
            temp_result = pd.DataFrame(table[:, i], index=time)
            temp_result.index = pd.to_datetime(temp_result.index)
            temp_mat = pd.merge(temp_result, factor, left_index=True, right_index=True).dropna()
            #model = sm.OLS(table[:,i], sm.add_constant(factor))
            # fit the model with the Newey-West Adjusment
            # lags=maxlag
            #re = model.fit()
            model = sm.OLS(temp_mat.iloc[:, 0], sm.add_constant(temp_mat.iloc[:, 1:]))
            re = model.fit()

            re = re.get_robustcov_results(cov_type='HAC', maxlags=self.maxlag, use_correction=True)
            #print(re.summary())
            # get anomaly and tvalues
            alpha[i] = re.params[0]
            ttest[i] = re.tvalues[0]
            beta[i, :] = re.params[1:]
            beta_ttest[i, :] = re.tvalues[1:]
            r2[i] = re.rsquared
        
        self.alpha = alpha
        self.alpha_tvalue = ttest
        self.beta = beta
        self.beta_ttest = beta_ttest
        self.r2 = r2
        self.params = re.params

        return alpha, ttest    
    
    def extractor(self, pos):
        '''
        This function extracts the return series
        input :
        pos (int): The position of the return series.

        output:
        series_ex (Series): The extracted Series.
        '''
        import numpy as np
        import pandas as pd
        
        if type(pos).__name__ != 'int':        
            return IOError 

        time = np.sort(np.unique(self.sample[:, 2]))
        series_ex = self.result[pos, :]
        r, c = np.shape(self.result)
        if pos in [r-1, -1]:
            series_ex = pd.DataFrame(np.array([time, series_ex]).T, columns=['Time', 'Diff'])
        else:
            series_ex = pd.DataFrame(np.array([time, series_ex]).T, columns=['Time', 'Ret'+str(pos)])

        return series_ex

    def summary_statistics(self, variables=None, periodic=False):
        '''
        This function is for summary statistics and outputs the group statistics and variables statistics. 
        input :
        variables (ndarray/DataFrame): variables, except sort variable, that need to be analyzed.
        periodic (boolean): whether print periodic results.

        Example:
        # summary statstics
        exper.summary_statistics()
        exper.summary_statistics(periodic=True)
        exper.summary_statistics(variables=np.array([variable_1, variable_2]).T, periodic=True)
        '''
        import numpy as np
        from scipy import stats as sts
        from prettytable import PrettyTable
        
        '''
        Group Statistics
        '''
        # Group Statistics
        if variables is None:
            groups_time = self.divide_by_time(self.sample)
            average_variable_period = np.zeros((self.number+1, 1, len(groups_time)))

        elif variables is not None:
            try:
                r, c = np.shape(variables)
            except:
                c = 1
            
            temp_sample = np.c_[self.sample, variables]
            groups_time = self.divide_by_time(temp_sample)
            average_variable_period = np.zeros((self.number+1, c+1, len(groups_time)))
        
        # ccalculate average variables
        for i in range(len(groups_time)):
            group = groups_time[i]
            if variables is None:
                average_variable_period[:, 0, i] = super().statistics(group[:, 1], self._label[i], np.mean)
            elif variables is not None:
                average_variable_period[:, 0, i] = super().statistics(group[:, 1], self._label[i], np.mean)
                average_variable_period[:, 1:, i] = super().statistics(group[:, -c:], self._label[i], np.mean)
                        
        # print the result
        table = PrettyTable()
        if periodic == True:
            table.field_names = ['Time'] + [str(i+1) for i in range(self.number+1)]
            for i in range(len(self._time)):    
                table.add_row([self._time[i]] + list(np.around(average_variable_period[:, 0, i], decimals=5)))
            
            if type(variables).__name__ == 'DataFrame' or type(variables).__name__ == 'Series':
                if type(variables).__name__ == 'DataFrame':
                    variables_col = list(variables.columns)
                elif type(variables).__name__ == 'Series':
                    variables_col = list(variables.name)

                for j in range(c):
                    table.add_row([variables_col[j]] + [' ' for i in range(self.number+1)])
                    for i in range(len(self._time)):
                        table.add_row([self._time[i]] + list(np.around(average_variable_period[:, j+1, i], decimals=5)))

            elif type(variables).__name__ == 'ndarray' :
                for j in range(c):
                    table.add_row(['Variable'+str(j+1)]+[' ' for i in range(self.number+1)])
                    for i in range(len(self._time)):
                        table.add_row([self._time[i]] + list(np.around(average_variable_period[:, j+1, i], decimals=5)))  

            self.average_variable_period = average_variable_period                  
                 
        elif periodic == False:
            average_variable = np.mean(average_variable_period, axis=2)
            table.field_names = ['Variable'] + [str(i+1) for i in range(self.number+1)]
            if self._sample_type == 'DataFrame':
                table.add_row([self._columns[1]] + list(np.around(average_variable[:, 0], decimals=5)))

            else: table.add_row(['Sort Variable'] + list(np.around(average_variable[:, 0], decimals=5)))

            if type(variables).__name__ == 'DataFrame' or type(variables).__name__ == 'Series':
                if type(variables).__name__ == 'DataFrame':
                    variables_col = list(variables.columns)
                elif type(variables).__name__ == 'Series':
                    variables_col = [variables.name]

                for j in range(c):
                    table.add_row([variables_col[j]] + list(np.around(average_variable[:, j+1], decimals=5)))
            elif type(variables).__name__ == 'ndarray' :
                for j in range(c):
                    table.add_row(['Variable'+str(j+1)] + list(np.around(average_variable[:, j+1], decimals=5)))

            self.average_variable = average_variable        
        
        else: 
            return IOError
        
        print('\nGroup Statistics')
        print(table)
        
        '''
        Variable Statistics
        '''
        # Variable Statistics
        table = PrettyTable()
        if periodic == True:
            table.field_names = ['Time', 'Mean', 'SD', 'Skew', 'Kurt', 'Min', 'P5', 'P25', 'Median', 'P75', 'P95', 'Max', 'n']
            for i in range(len(groups_time)):
                group = groups_time[i]
                stats_mean = np.mean(group[:, 1])
                stats_std = np.std(group[:, 1])
                stats_skew = sts.skew(group[:, 1])
                stats_kurt = sts.kurtosis(group[:, 1])
                stats_min = np.min(group[:, 1])
                stats_perc5 = np.percentile(group[:, 1] ,5)
                stats_perc25 = np.percentile(group[:, 1], 25)
                stats_perc50 = np.percentile(group[:, 1], 50)
                stats_perc75 = np.percentile(group[:, 1], 75)
                stats_perc95 = np.percentile(group[:, 1], 95)
                stats_max = np.max(group[:, 1])
                stats_n = len(group[:, 1])

                table.add_row([self._time[i]] + list(np.around([stats_mean, stats_std, stats_skew, stats_kurt, stats_min, stats_perc5, stats_perc25, stats_perc50, stats_perc75, stats_perc95, stats_max, stats_n], decimals=5)))               
            
            if type(variables).__name__ == 'DataFrame' or type(variables).__name__ == 'Series':
                if type(variables).__name__ == 'DataFrame':
                    variables_col = list(variables.columns)
                elif type(variables).__name__ == 'Series':
                    variables_col = list(variables.name)
                
                for j in range(c):
                    table.add_row([variables_col[j]] + [' ' for k in range(12)])
                    for i in range(len(groups_time)):
                        group = groups_time[i]
                        stats_mean = np.mean(group[:, -(c-j)])
                        stats_std = np.std(group[:, -(c-j)])
                        stats_skew = sts.skew(group[:, -(c-j)])
                        stats_kurt = sts.kurtosis(group[:, -(c-j)])
                        stats_min = np.min(group[:, -(c-j)])
                        stats_perc5 = np.percentile(group[:, -(c-j)] ,5)
                        stats_perc25 = np.percentile(group[:, -(c-j)], 25)
                        stats_perc50 = np.percentile(group[:, -(c-j)], 50)
                        stats_perc75 = np.percentile(group[:, -(c-j)], 75)
                        stats_perc95 = np.percentile(group[:, -(c-j)], 95)
                        stats_max = np.max(group[:, -(c-j)])
                        stats_n = len(group[:, -(c-j)])

                        table.add_row([self._time[i]] + list(np.around([stats_mean, stats_std, stats_skew, stats_kurt, stats_min, stats_perc5, stats_perc25, stats_perc50, stats_perc75, stats_perc95, stats_max, stats_n], decimals=5)))               
            
            elif type(variables).__name__ == 'ndarray':
                for j in range(c):
                    table.add_row(['Variable'+str(j+1)] + [' ' for k in range(12)])
                    for i in range(len(groups_time)):
                        group = groups_time[i]
                        stats_mean = np.mean(group[:, -(c-j)])
                        stats_std = np.std(group[:, -(c-j)])
                        stats_skew = sts.skew(group[:, -(c-j)])
                        stats_kurt = sts.kurtosis(group[:, -(c-j)])
                        stats_min = np.min(group[:, -(c-j)])
                        stats_perc5 = np.percentile(group[:, -(c-j)] ,5)
                        stats_perc25 = np.percentile(group[:, -(c-j)], 25)
                        stats_perc50 = np.percentile(group[:, -(c-j)], 50)
                        stats_perc75 = np.percentile(group[:, -(c-j)], 75)
                        stats_perc95 = np.percentile(group[:, -(c-j)], 95)
                        stats_max = np.max(group[:, -(c-j)])
                        stats_n = len(group[:, -(c-j)])

                        table.add_row([self._time[i]] + list(np.around([stats_mean, stats_std, stats_skew, stats_kurt, stats_min, stats_perc5, stats_perc25, stats_perc50, stats_perc75, stats_perc95, stats_max, stats_n], decimals=5)))               

        elif periodic == False:
            table.field_names = ['Variable', 'Mean', 'SD', 'Skew', 'Kurt', 'Min', 'P5', 'P25', 'Median', 'P75', 'P95', 'Max', 'n']
            stats_mean = np.mean(self.sample[:, 1])
            stats_std = np.std(self.sample[:, 1])
            stats_skew = sts.skew(self.sample[:, 1])
            stats_kurt = sts.kurtosis(self.sample[:, 1])
            stats_min = np.min(self.sample[:, 1])
            stats_perc5 = np.percentile(self.sample[:, 1] ,5)
            stats_perc25 = np.percentile(self.sample[:, 1], 25)
            stats_perc50 = np.percentile(self.sample[:, 1], 50)
            stats_perc75 = np.percentile(self.sample[:, 1], 75)
            stats_perc95 = np.percentile(self.sample[:, 1], 95)
            stats_max = np.max(self.sample[:, 1])
            stats_n = int(len(self.sample[:, 1]) / len(self._time))

            if self._sample_type == 'DataFrame':                
                table.add_row([self._columns[1]] + list(np.around([stats_mean, stats_std, stats_skew, stats_kurt, stats_min, stats_perc5, stats_perc25, stats_perc50, stats_perc75, stats_perc95, stats_max, stats_n], decimals=5)))
            else:
                table.add_row(['Sort Variable'] + list(np.around([stats_mean, stats_std, stats_skew, stats_kurt, stats_min, stats_perc5, stats_perc25, stats_perc50, stats_perc75, stats_perc95, stats_max, stats_n], decimals=5)))
            
            if type(variables).__name__ == 'DataFrame' or type(variables).__name__ == 'Series':
                if type(variables).__name__ == 'DataFrame':
                    variables_col = list(variables.columns)
                elif type(variables).__name__ == 'Series':
                    variables_col = list([variables.name])
                
                stats_mean = np.mean(temp_sample[:, -c:], axis=0)
                stats_std = np.std(temp_sample[:, -c:], axis=0, dtype=float)
                if c > 1: 
                    stats_skew = sts.skew(variables.iloc[:, -c:], axis=0)
                elif c == 1:
                    stats_skew = [sts.skew(variables)] 
                stats_kurt = sts.kurtosis(temp_sample[:, -c:], axis=0)
                stats_min = np.min(temp_sample[:, -c:], axis=0)
                stats_perc5 = np.percentile(temp_sample[:, -c:], 5, axis=0)
                stats_perc25 = np.percentile(temp_sample[:, -c:], 25, axis=0)
                stats_perc50 = np.percentile(temp_sample[:, -c:], 50, axis=0)
                stats_perc75 = np.percentile(temp_sample[:, -c:], 75, axis=0)
                stats_perc95 = np.percentile(temp_sample[:, -c:], 95, axis=0)
                stats_max = np.max(temp_sample[:, -c:], axis=0)
                stats_n = int(len(temp_sample[:, 1]) / len(self._time))
                for j in range(c):
                    table.add_row([variables_col[j]] + list(np.around([stats_mean[j], stats_std[j], stats_skew[j], stats_kurt[j], stats_min[j], stats_perc5[j], stats_perc25[j], stats_perc50[j], stats_perc75[j], stats_perc95[j], stats_max[j], stats_n], decimals=5)))

            elif type(variables).__name__ == 'ndarray':
                stats_mean = np.mean(temp_sample[:, -c:], axis=0)
                stats_std = np.std(temp_sample[:, -c:], axis=0, dtype=float)
                if c > 1:    
                    stats_skew = sts.skew(variables[:, -c:], axis=0)
                elif c == 1:
                    stats_skew = [sts.skew(variables)]
                stats_kurt = sts.kurtosis(temp_sample[:, -c:], axis=0)
                stats_min = np.min(temp_sample[:, -c:], axis=0)
                stats_perc5 = np.percentile(temp_sample[:, -c:], 5, axis=0)
                stats_perc25 = np.percentile(temp_sample[:, -c:], 25, axis=0)
                stats_perc50 = np.percentile(temp_sample[:, -c:], 50, axis=0)
                stats_perc75 = np.percentile(temp_sample[:, -c:], 75, axis=0)
                stats_perc95 = np.percentile(temp_sample[:, -c:], 95, axis=0)
                stats_max = np.max(temp_sample[:, -c:], axis=0)
                stats_n = int(len(temp_sample[:, 1]) / len(self._time))
                for j in range(c):
                    table.add_row(['Variable'+str(j+1)] + list(np.around([stats_mean[j], stats_std[j], stats_skew[j], stats_kurt[j], stats_min[j], stats_perc5[j], stats_perc25[j], stats_perc50[j], stats_perc75[j], stats_perc95[j], stats_max[j], stats_n], decimals=5)))

        else: 
            return IOError

        # print the result
        print('\nIndicator Statistics')
        print(table)
    
    def correlation(self, variables, periodic=False, export=False):
        '''
        This function is for calculating correlation coefficient of variables.
        input :
        variables (ndarray/DataFrame): The variables to be analyzed.
        periodic (boolean): whether prints the periodic result. The **DEFAULT** is False.
        export (boolean): whether exports the summary table. The **DEFAULT** is False.

        output :
        df (DataFrame): The summary table if export is True.

        Example :
        # correlation 
        variable_3 = np.random.normal(0, 1, 20*3000)
        variable_4 = np.random.normal(0, 1, 20*3000)
        print('-------------------------------------- Correlation ---------------------------------')
        exper.correlation(variables=np.array([variable_1, variable_2, variable_3, variable_4]).T, periodic=True)
        exper.correlation(variables=np.array([variable_1, variable_2, variable_3, variable_4]).T)
        '''
        # Variable Statistics
        # input:
        # variables (ndarray\DataFrame)
        import numpy as np
        from prettytable import PrettyTable 
        from scipy import stats as sts
        
        r, c = np.shape(variables)
        temp_sample = np.c_[self.sample, variables]
        groups_time = self.divide_by_time(temp_sample)

        table = PrettyTable()
        table_spear = PrettyTable()
        # create field name
        if type(variables).__name__ == 'DataFrame':
            variables_col = list(variables.columns)
            field_name = list()
            for i in range(len(variables_col)):
                for j in range(len(variables_col)):
                    if j > i:
                        field_name.append(variables_col[i]+'&'+variables_col[j])
        
        elif type(variables).__name__ == 'ndarray':
            r , c =np.shape(variables)
            variables_col = ['Variable_' + str(i) for i in range(c)]
            field_name = list()
            for i in range(len(variables_col)):
                for j in range(len(variables_col)):
                    if j > i :
                        field_name.append(variables_col[i]+' & '+variables_col[j])
        
        if periodic == True:
            table.field_names = ['Time'] + field_name
            table_spear.field_names = ['Time'] + field_name
        elif periodic == False:
            table.field_names = ['Variable'] + field_name
        
        # calculate correlation coefficient
        corr_maxtrix = np.zeros((len(groups_time), len(field_name)))
        corr_maxtrix_spearman = np.zeros((len(groups_time), len(field_name)))
        for i in range(len(groups_time)):
            group = groups_time[i]
            temp_variables = group[:, -c:]
            corr = list()
            corr_spearman = list()
            for j in range(c):
                for k in range(c):
                    if k > j :
                        try:
                            corr.append(np.around(sts.pearsonr(temp_variables[:, j], temp_variables[:, k])[0], decimals=5))
                            corr_spearman.append(np.around(sts.spearmanr(temp_variables[:, j], temp_variables[:, k])[0], decimals=5))
                        except:
                            corr.append(np.nan)
                            corr_spearman.append(np.nan)
            corr_maxtrix[i, :] = corr
            corr_maxtrix_spearman[i, :] = corr_spearman
            
            if periodic == True:
                table.add_row([str(self._time[i]) + ' '] + corr)
                table_spear.add_row([str(self._time[i]) + ' '] + corr_spearman)
                
        if periodic == False:
            table.add_rows([['Pearson'] + list(np.around(np.nanmean(corr_maxtrix, axis=0), decimals=5))])
            table.add_row(['Spearman'] + list(np.around(np.nanmean(corr_maxtrix_spearman, axis=0), decimals=5)))
            print(table)

            if export == True :
                import pandas as pd
                try:
                    from StringIO import StringIO
                except ImportError:
                    from io import StringIO
            
                csv_string = table.get_csv_string()
                with StringIO(csv_string) as f:
                    df = pd.read_csv(f)
            
                return df


        elif periodic == True:  
            print('Spearman Correlation')          
            print(table_spear)
        
            print('Pearson Correlation')
            print(table)

            if export == True :
                import pandas as pd
                try:
                    from StringIO import StringIO
                except ImportError:
                    from io import StringIO
            
                csv_string = table.get_csv_string()
                csv_string_spear = table_spear.get_csv_string()
                with StringIO(csv_string) as f:
                    df = pd.read_csv(f)
                
                with StringIO(csv_string_spear) as f_spear:
                    df_spear = pd.read_csv(f_spear)
            
                return df, df_spear

    def print_summary_by_time(self, export=False) :
        '''
        This function print the summary grouped by time.
        input :
        export (boolean): Export the table or not. The table is exported in form of Dataframe. The default setting is False.

        output :
        df (DataFrame): The table exported in form of Dataframe.
        '''
        import numpy as np
        from prettytable import PrettyTable
        
        r, c = np.shape(self.result)
        table = PrettyTable()
        time = np.sort(np.unique(self.sample[:, 2]))
        table.add_column('Time', time)
        for i in range(r-1):
            table.add_column(str(i+1), np.around(self.result[i, :], decimals=3))
        table.add_column('diff', np.around(self.result[r-1,:], decimals=3))
        print(table)

        if export == True :
            import pandas as pd
            try:
                from StringIO import StringIO
            except ImportError:
                from io import StringIO
            
            csv_string = table.get_csv_string()
            with StringIO(csv_string) as f:
                df = pd.read_csv(f)
            
            return df
        
    def print_summary(self, explicit = False, export=False, percentage=False):
        '''
        This function print the summary grouped by characteristic and averaged by time.
        input :
        explicit (boolean): Whether presents the explicit result. The default is **False**.
        export (boolean): Export the table or not. The table is exported in form of Dataframe. The default setting is **False.**
        percentage (boolean): Whether presents the percentage average return. The default is **False**.

        output :
        df (DataFrame): The table exported in form of Dataframe.

        Example :
        exper.summary_and_test()
        exper.print_summary_by_time()
        exper.print_summary()
        '''
        import numpy as np
        from prettytable import PrettyTable
        # generate Table if no factor
        table = PrettyTable()
        table.add_column('Group', ['Average', 'T-Test'])
        for i in range(self.number+1):
            if percentage == False:
                table.add_column(str(i+1), np.around([self.average[i], self.ttest[0][i]], decimals=3))
            elif percentage == True:
                table.add_column(str(i+1), np.around([self.average[i]*100, self.ttest[0][i]], decimals=3))
        if percentage == False:
            table.add_column('Diff', np.around([self.average[-1], self.ttest[0][-1]], decimals=3))
        elif percentage == True:
            table.add_column('Diff', np.around([self.average[-1]*100, self.ttest[0][-1]], decimals=3))
        
        if self._factor is not None :
            table = PrettyTable()
            fac_name = list()
            row_fac, col_fac = np.shape(self._factor)
            if type(self._factor).__name__ == 'DataFrame':
                for i in range(len(self._factor.columns)):
                    fac_name.append(self._factor.columns[i])
                    fac_name.append('T-Test')                
            else: 
                for i in range(col_fac):
                    fac_name.append('factor'+str(i+1))
                    fac_name.append('T-Test')
            fac_name.append('R2')

            if explicit == False:
                table.add_column('Group', ['Average', 'T-Test', 'Alpha', 'Alpha-T'])
            elif explicit == True:
                temp_name = ['Average', 'T-Test', 'Alpha', 'Alpha-T']+fac_name
                table.add_column('Group', temp_name)
            
            for i in range(self.number+1):
                fac_re = list()
                for j in range(col_fac):
                    fac_re.append(self.beta[i][j])
                    fac_re.append(self.beta_ttest[i][j])
                fac_re.append(self.r2[i][0])

                if explicit == False:
                    if percentage == False:
                        table.add_column(str(i+1), np.around([self.average[i], self.ttest[0][i], self.alpha[i][0], self.alpha_tvalue[i][0]], decimals=3))
                    elif percentage == True:
                        table.add_column(str(i+1), np.around([self.average[i]*100, self.ttest[0][i], self.alpha[i][0]*100, self.alpha_tvalue[i][0]], decimals=3))
                elif explicit == True:
                    if percentage == False:
                        temp_re = [self.average[i], self.ttest[0][i], self.alpha[i][0], self.alpha_tvalue[i][0]] + fac_re
                    elif percentage == True:
                        temp_re = [self.average[i]*100, self.ttest[0][i], self.alpha[i][0]*100, self.alpha_tvalue[i][0]] + fac_re
                    table.add_column(str(i+1), np.around(temp_re, decimals=3))
            
            fac_re = list()
            for j in range(col_fac):
                if percentage == False:
                    fac_re.append(self.beta[-1][j])
                elif percentage == True:
                    fac_re.append(self.beta[-1][j]*100)
                fac_re.append(self.beta_ttest[-1][j])
            fac_re.append(self.r2[-1][0])

            if explicit == False:
                if percentage == False:
                    table.add_column('Diff', np.around([self.average[-1], self.ttest[0][-1], self.alpha[-1][0], self.alpha_tvalue[-1][0]], decimals=3))
                elif percentage == True:
                    table.add_column('Diff', np.around([self.average[-1]*100, self.ttest[0][-1], self.alpha[-1][0]*100, self.alpha_tvalue[-1][0]], decimals=3))
            elif explicit == True:
                if percentage == False:
                    table.add_column('Diff', np.around([self.average[-1], self.ttest[0][-1], self.alpha[-1][0], self.alpha_tvalue[-1][0]] + fac_re, decimals=3))
                elif percentage == True:
                    table.add_column('Diff', np.around([self.average[-1]*100, self.ttest[0][-1], self.alpha[-1][0]*100, self.alpha_tvalue[-1][0]] + fac_re, decimals=3))
        np.set_printoptions(formatter={'float':'{:0.3f}'.format})
        print(table)

        if export == True :
            import pandas as pd
            try:
                from StringIO import StringIO
            except ImportError:
                from io import StringIO
            
            csv_string = table.get_csv_string()
            with StringIO(csv_string) as f:
                df = pd.read_csv(f)
            
            return df

    
class Bivariate(ptf_analysis):
    '''
    This module is for Bivariate analysis
    '''
    def __init__(self, sample):
        '''
        input: sample: the samples to be analyzed 
            sample usually contain the future return, characteristics, time 
            the DEFAULT settting:
            the First column is the forecast return and 
            the second column is the row characteristic
            the third column is the column characteristic
            the fourth colunm or the index is time label
            
            number: the breakpoint number
            perc: the breakpoint percentiles
            factor: risk adjust factor
            maxlag: maximum lag for Newey-West adjustment
        '''
        import numpy as np

        if type(sample).__name__ == 'DataFrame' :
            self.sample = np.array(sample)
            self._sample_type = 'DataFrame'
            self._columns = list(sample.columns)
        elif type(sample).__name__ == 'ndarray' :
            self.sample = sample
            self._sample_type = 'ndarray'
        else:
            raise IOError
        
        
    def divide_by_time(self):
        '''
        This function groups the sample by time.
        output :
        groups_by_time (list): The samples group by time.
        '''
        import numpy as np
        
        time=np.sort(np.unique(self.sample[:, 3]))
        groups_by_time=list()
        for i in range(len(time)):
            groups_by_time.append(self.sample[np.where(self.sample[:, 3]==time[i])])
        
        return groups_by_time
        
    def average_by_time(self, conditional=False):
        '''
        average of the groups at each time point
        This function, using the sample group by time from function *divide_by_time*, 
        groups the sample by the characteristic, 
        and then calculate average return of each group samples at every time point. 
        input :
        conditional (boolean): The way of sorting. 
            If true, it is dependent-sort analysis; 
            if false, it is independent sort analysis. 
            The Default setting is **False**. 

        output : 
        average_group_time(matrix: N_N_T): The average return of groups by each characteristic pair indexed by time.

        Example :
        import numpy as np
        from portfolio_analysis import Bivariate as bi
    
        # generate time 
        year = np.ones((3000,1), dtype=int)*2020
        for i in range(19):
            year = np.append(year, (2019-i)*np.ones((3000,1), dtype=int))
    
        # generate character
        character_1 = np.random.normal(0, 1, 20*3000)
        character_2 = np.random.normal(0, 1, 20*3000)

        # generate future return
        ret=character_1*-0.5 + character_2*0.5 + np.random.normal(0,1,20*3000)
        # create sample containing future return, character, time
        sample=np.array([ret,character_1, character_2, year]).T
        print(sample)
        # generate the Univiriate Class
        exper=bi(sample,9)
        # test function divide_by_time
        group_by_time = exper.divide_by_time()
        print(group_by_time)
        # test function average_by_time
        average_group_time = exper.average_by_time()
        print(average_group_time)
        print(np.shape(average_group_time))
        '''
        import numpy as np
        # get the sample groups by time 得到按时间分组的样本
        groups_time = self.divide_by_time()
        # generate table of  average return for group and time 生成组和时间的表格
        # Rows: groups 行： 组  
        # Columns: Time 列： 时间
        average_group_time = np.zeros((self.num_row, self.num_col, len(groups_time)))
                                      
        for i in range(len(groups_time)):
            # for each time, there exists a group 
            group = groups_time[i]
            # for each time, generate breakpoint
            if self.percn_row is None:
                breakpoint_row = super().select_breakpoints(group[:, 1], self.num_row - 1, perc=self.perc_row)
            else:
                breakpoint_row = super().select_breakpoints(group[:, 1], self.num_row - 1, percn=self.percn_row[i, :])
            # for each time, generate label
            label_row = super().distribute(group[:, 1], breakpoint_row)[:, 0]
            
            if conditional == False:
                if self.percn_col is None:
                    breakpoint_col = super().select_breakpoints(group[:, 2], self.num_col - 1, perc=self.perc_col)
                else:
                    breakpoint_col = super().select_breakpoints(group[:, 2], self.num_col - 1, percn=self.percn_col[i, :])
                label_col = super().distribute(group[:, 2], breakpoint_col)[:, 0]
            elif conditional == True:
                label_row_unique = list(np.unique(label_row))
                label_col = - np.ones(len(group[:, 2]))
                for j in range(len(label_row_unique)):
                    if self.percn_col is None:
                        breakpoint_col = super().select_breakpoints(group[:, 2][np.where(label_row==label_row_unique[j])], self.num_col - 1, perc=self.perc_col)
                    else:
                        breakpoint_col = super().select_breakpoints(group[:, 2][np.where(label_row==label_row_unique[j])], self.num_col - 1, percn=self.percn_col[i, :])

                    label_col[np.where(label_row==label_row_unique[j])] = super().distribute(group[:, 2][np.where(label_row==label_row_unique[j])], breakpoint_col)[:, 0]

            # for each group in each time, calculate the average future return
            label = [label_row, label_col]

            if self.perc_sign == False:
                if self.weight == False:
                    try:
                        average_group_time[:,:,i] = super().average(group[:, 0], label, cond='bi').reshape((self.num_row, self.num_col))
                    except:
                        average_group_time[:,:,i] = np.full([self.num_row, self.num_col], np.nan)
                else:
                    try:
                        average_group_time[:,:,i] = super().average(group[:, 0], label, cond='bi', weight=group[:, 4]).reshape((self.num_row, self.num_col))
                    except:
                        average_group_time[:,:,i] = np.full([self.num_row, self.num_col], np.nan)
            elif self.perc_sign == True:
                if self.weight == False:
                    try:
                        average_group_time[:,:,i] = super().average(group[:, 0], label, cond='bi').reshape((self.num_row, self.num_col))
                    except:
                        average_group_time[:,:,i] = np.full([self.num_row, self.num_col], np.nan)
                else:
                    try:
                        average_group_time[:,:,i] = super().average(group[:, 0], label, cond='bi', weight=group[:, 4]).reshape((self.num_row, self.num_col))
                    except:
                        average_group_time[:,:,i] = np.full([self.num_row, self.num_col], np.nan)

        # return the Table
        # Rows: groups in each time
        # Columns: Time 

        return average_group_time
            
    def difference(self, average_group):
        '''
        calculate the difference group return
        This functions calculates the difference of group return, which, in detail, is the last group average return minus the first group average return. 
        input : 
        average_group (ndarray): The average return of groups by each characteristic-time pair.

        output :
        result (ndarray): The matrix added with the difference of average group return.
        '''
        import numpy as np
        
        a, b, c= np.shape(average_group)
        diff_row = average_group[-1, :, :] - average_group[0, :, :]
        diff_row = diff_row.reshape((1, b, c))        
        result = np.append(average_group, diff_row, axis=0)

        diff_col = result[:, -1, :] - result[:, 0, :]
        diff_col = diff_col.reshape((a+1, 1, c))
        result = np.append(result, diff_col, axis=1)
        return result
    
    def factor_adjustment(self, factor):
        '''
        This function calculates the group return adjusted by risk factors.

        input :
        factor (ndarray or DataFrame): The return table with difference sequence.

        output :
        alpha (ndarray): The anomaly
        ttest (ndarray): The t-value of the anomaly.
        '''
        import statsmodels.api as sm
        import numpy as np
        import pandas as pd
        
        self._factor = factor
        # result: r * c * n 
        r, c, n = np.shape(self.result)
        row_fac, col_fac = np.shape(self._factor)
        # generate anomaly: alpha 
        # generate tvalues: ttest
        alpha = np.zeros((r, c))
        ttest = np.zeros((r, c))
        time = np.sort(np.unique(self.sample[:, 3]))
        beta = np.zeros((r, c, col_fac))
        beta_test = np.zeros((r, c, col_fac))
        r2 = np.zeros((r, c))

        # factor adjusment
        for i in range(r):
            for j in range(c):
                temp_result = pd.DataFrame(self.result[i, j, :], index=time)
                temp_result.index = pd.to_datetime(temp_result.index)
                temp_mat = pd.merge(temp_result, factor, left_index=True, right_index=True).dropna()
                #model = sm.OLS(self.result[i, j, :], sm.add_constant(factor))
                # fit the model with the Newey-West Adjusment
                # lags=maxlag
                #re = model.fit()
                model = sm.OLS(temp_mat.iloc[:, 0], sm.add_constant(temp_mat.iloc[:, 1:]))
                re = model.fit()

                re = re.get_robustcov_results(cov_type='HAC', maxlags=self.maxlag, use_correction=True)
                #print(re.summary())
                # get anomaly and tvalues
                alpha[i, j] = re.params[0]
                ttest[i, j] = re.tvalues[0]
                beta[i, j, :] = re.params[1:]
                beta_test[i, j, :] = re.tvalues[1:]
                r2[i, j] = re.rsquared
        
        self.alpha = alpha
        self.alpha_tvalue = ttest
        self.beta = beta
        self.beta_test = beta_test
        self.r2 = r2

        return alpha, ttest    
    
    def summary_and_test(self, **kwargs) :
        '''
        This function summarizes the result and take t-test.
        input :
        export (boolean): Export the table or not. The table is exported in form of Dataframe. The default setting is **False.**

        output :
        self.average (array): The average of the portfolio return across time.
        self.ttest (array): The t-value of the portfolio return across time.
        '''
        import numpy as np
        from scipy import stats
        from .adjust import newey_west_t
        
        self.result = self.difference(self.average_by_time(**kwargs))
        r, c, z = np.shape(self.result)
        self.average = np.nanmean(self.result, axis=2)
        
        if self.maxlag == 0 :
            self.ttest = stats.ttest_1samp(self.result, 0.0, axis=2, nan_policy='omit')
        else:
            self.ttest = np.zeros((2, r, c))

            for i in range(r):
                for j in range(c):
                    temp_result = self.result[i, j, :]
                    temp_result = temp_result[~np.isnan(temp_result)]
                    temp_one = np.ones((len(temp_result), 1))
                    tvalue, pvalue = newey_west_t(y=temp_result, X=temp_one, J=self.maxlag, constant=False)
                    self.ttest[0, i, j] = tvalue
                    self.ttest[1, i, j] = pvalue
#        if self.maxlag == 0 :
#            self.ttest = stats.ttest_1samp(self.result, 0.0, axis=1, nan_policy='omit')
#        else:
#            self.ttest = np.zeros((2, r))

#            for i in range(r):
#                temp_result = self.result[i, :]
#                temp_result = temp_result[~np.isnan(temp_result)]
#                temp_one = np.ones((len(temp_result), 1))
#                tvalue, pvalue = newey_west_t(y=temp_result, X=temp_one, J=self.maxlag, constant=False)
#                self.ttest[0, i] = tvalue
#                self.ttest[1, i] = pvalue

        # if there is a facotr adjustment, then normal return result plus the anomaly result
#        if self.factor is not None  :
#            self.alpha, self.alpha_tvalue = self.factor_adjustment(self.result)
#            return self.average, self.ttest, self.alpha, self.alpha_tvalue

        return self.average, self.ttest

    def fit(self, number=4, perc_row=None, perc_col=None, percn_row=None, percn_col=None, weight=False, maxlag=12, **kwargs):
        '''
        This function run the function **summary_and_test().**
        input :
        number (int): The breakpoint number.
        perc_row (list or array): The breakpoint percentile points of row characteristics.
        perc_col (list or array): The breakpoint percentile points of column characteristics.
        percn_row (list or array): The breakpoints percentiles of row characteristics.
        percn_col (list or array): The breakpoints percentiles of column characteristics.
        weight (boolean): Whether calculate the weighted average return.
        maxlag (int):  The maximum lag for Newey-West adjustment.
        kwargs : kwargs include settings like conditional, etc. 
        '''
        import numpy as np
        # the number of groups

        self._factor = None
        self.maxlag = maxlag
        self.weight = weight
        self.perc_row = perc_row
        self.perc_col = perc_col
        self.percn_row = percn_row
        self.percn_col = percn_col
        if all([perc_row is None, perc_col is None, percn_row is None, percn_col is None]):
            self.num_row = number + 1
            self.num_col = number + 1
            self.perc_sign = False
        else:
            self.perc_sign = True
            if all([perc_row is not None, perc_col is not None]):
                self.num_row = len(perc_row) - 1 
                self.num_col = len(perc_col) - 1
            elif all([percn_row is not None, percn_col is not None]):
                r_row, c_row = np.shape(percn_row)
                r_col, c_col = np.shape(percn_col)
                self.num_row = c_row - 1
                self.num_col = c_col - 1

        self.summary_and_test(**kwargs)
    
    def extractor(self, r_pos, c_pos):
        '''
        This function extracts the return series.
        input :
        r_pos (int): The row position of the return matrix.
        c_pos (int): The column position of the return matrix.

        output :
        series_ex (Series): The extracted Series.
        '''
        import numpy as np
        import pandas as pd
        
        if type(r_pos).__name__ != 'int':        
            raise IOError 
        elif type(c_pos).__name__ != 'int':
            raise IOError
            
        time = np.sort(np.unique(self.sample[:, 3]))
        series_ex = self.result[r_pos, c_pos, :]
        r, c, z = np.shape(self.result)
        if r_pos in [r-1, -1] :
            if c_pos in [c-1, -1]:
                series_ex = pd.DataFrame(np.array([time, series_ex]).T, columns=['Time', 'Diff'+'Diff'])
            else:
                series_ex = pd.DataFrame(np.array([time, series_ex]).T, columns=['Time', 'Diff'+'Ret'+str(c_pos)])            
        else:
            if c_pos in [c-1, -1]:
                series_ex = pd.DataFrame(np.array([time, series_ex]).T, columns=['Time', 'Ret'+str(r_pos)+'Diff'])
            else:
                series_ex = pd.DataFrame(np.array([time, series_ex]).T, columns=['Time', 'Ret'+str(r_pos)+'Ret'+str(c_pos)])

        return series_ex


    def print_summary_by_time(self, export=False) :
        '''
        This function print the summary grouped by time.
        input : 
        export (boolean): Export the table or not. The table is exported in form of Dataframe. The default setting is **False.**

        output :
        df (DataFrame): The table exported in form of DataFrame.
        '''
        import numpy as np
        from prettytable import PrettyTable
        
        r, c, n = np.shape(self.result)
        table = PrettyTable()
        time = np.sort(np.unique(self.sample[:, 3]))
        table.field_names = ['Time', 'Group'] + [str(i+1) for i in range(self.num_row)] + ['Diff']
        for i in range(n):
            for j in range(r):
                if j == 0 :
                    temp = [time[i], j+1]
                elif j == r - 1 :
                    temp = [' ', 'Diff']
                else :
                    temp = [' ', j+1]
                for k in range(c): 
                    temp.append(np.round(self.result[j, k, i], decimals=3))

                table.add_row(temp)
        print(table)

        if export == True :
            import pandas as pd
            try:
                from StringIO import StringIO
            except ImportError:
                from io import StringIO
            
            csv_string = table.get_csv_string()
            with StringIO(csv_string) as f:
                df = pd.read_csv(f)
            
            return df
        
    def print_summary(self, explicit=False, export=False, percentage=False):
        '''
        print summary
        This function print the summary grouped by characteristic and averaged by time.
        input :
        explicit (boolean): Whether presents the explicit result. The default is **False**.
        export (boolean): Export the table or not. The table is exported in form of Dataframe. The default setting is **False.**
        percentage (boolean): Whether presents the percentage return. The default is **False**.

        output :
        df (DataFrame): The table exported in form of Dataframe.

        Example:
        result = exper.difference(average_group_time)
        print('result :\n', result)
        print('difference matrix :\n', np.shape(result))
        # test function summary_and_test
        average, ttest = exper.summary_and_test()
        print('average :\n', average)
        print(' shape of average :', np.shape(average))
        print('ttest :\n', ttest)
        print('shape of ttest :', np.shape(ttest)) 
        # test function print_summary_by_time()
        exper.print_summary_by_time()
        # test function print_summary
        exper.print_summary()
    
        # generate factor
        factor=np.random.normal(0,1.0,(20,1))
        exper=bi(sample,9,factor=factor,maxlag=12)
        exper.fit()
        exper.print_summary()
        '''
        import numpy as np
        from prettytable import PrettyTable
        # generate Table if no factor
        if self._factor is None :
            table=PrettyTable()
            if self._sample_type == 'ndarray':
                table.field_names = ['Group'] + [i+1 for i in range(self.num_col)] + ['Diff']
            elif self._sample_type == 'DataFrame':
                table.field_names = ['Group'] + [self._columns[2] + str(i+1) for i in range(self.num_col)] + ['Diff']

            for i in range(self.num_row+1):
                if i == self.num_row :
                    temp = ['Diff']
                    temp_tvalue = [' ']
                else:
                    if self._sample_type == 'ndarray':
                        temp = [str(i+1)]
                    elif self._sample_type == 'DataFrame':
                        temp = [self._columns[1] + str(i+1)]
                    temp_tvalue = [' ']
                for j in range(self.num_col+1):
                    if percentage == False:
                        temp.append(np.around(self.average[i, j], decimals=3))
                    elif percentage == True:
                        temp.append(np.around(self.average[i, j]*100, decimals=3))
                    
                    temp_tvalue.append(np.around(self.ttest[0][i, j], decimals=3))

                table.add_row(temp)
                table.add_row(temp_tvalue)

        elif self._factor is not None :
            table = PrettyTable()
            row_fac, col_fac = np.shape(self._factor)
            if self._sample_type == 'ndarray':
                table.field_names = ['Group'] + [i+1 for i in range(self.num_col)] + ['Diff']
            elif self._sample_type == 'DataFrame':
                table.field_names = ['Group'] + [self._columns[2] + str(i+1) for i in range(self.num_col)] + ['Diff']

            if type(self._factor).__name__ == 'DataFrame':
                fac_name = self._factor.columns
            elif type(self._factor).__name__ == 'ndarray':
                fac_name = ['factor' + str(i+1) for i in range(col_fac)]

            for i in range(self.num_row+1):
                if i == self.num_row:
                    temp = ['Diff']
                    temp_tvalue = [' ']
                    temp_fac = ['alpha']
                    temp_fac_tvalue = [' ']
                    temp_r2 = ['R2']
                else :
                    if self._sample_type == 'ndarray':
                        temp = [str(i+1)]
                    elif self._sample_type == 'DataFrame':
                        temp = [self._columns[1] + str(i+1)]
                    temp_tvalue = [' ']
                    temp_fac = ['alpha']
                    temp_fac_tvalue = [' ']
                    temp_r2 = ['R2']
                for j in range(self.num_col+1):
                    if percentage == False:
                        temp.append(np.around(self.average[i, j], decimals=3))
                    elif percentage == True:
                        temp.append(np.around(self.average[i, j]*100, decimals=3))
                    temp_tvalue.append(np.around(self.ttest[0][i, j], decimals=3))
                    if percentage == False:
                        temp_fac.append(np.around(self.alpha[i, j], decimals=3))
                    elif percentage == True:
                        temp_fac.append(np.around(self.alpha[i, j]*100, decimals=3))
                    temp_fac_tvalue.append(np.around(self.alpha_tvalue[i, j], decimals=3))
                    temp_r2.append(np.around(self.r2[i, j], decimals=3))

                table.add_row(temp)
                table.add_row(temp_tvalue)
                table.add_row(temp_fac)
                table.add_row(temp_fac_tvalue)

                if explicit == True:
                    for k in range(col_fac):
                        temp_fac_beta = [fac_name[k]]
                        temp_fac_betatvalue = ['t-value']
                        temp_blank = [' ']
                        for j in range(self.num_col+1):
                            if percentage == False:
                                temp_fac_beta.append(np.around(self.beta[i, j, k], decimals=3))
                            elif percentage == True:
                                temp_fac_beta.append(np.around(self.beta[i, j, k]*100, decimals=3))
                            
                            temp_fac_betatvalue.append(np.around(self.beta_test[i, j, k], decimals=3))
                            temp_blank.append(' ')
                    
                        table.add_row(temp_fac_beta)
                        table.add_row(temp_fac_betatvalue)
                    table.add_row(temp_r2)
                    table.add_row(temp_blank)

        np.set_printoptions(formatter={'float':'{:0.3f}'.format})
        print(table)

        if export == True :
            import pandas as pd
            try:
                from StringIO import StringIO
            except ImportError:
                from io import StringIO
            
            csv_string = table.get_csv_string()
            with StringIO(csv_string) as f:
                df = pd.read_csv(f)
            
            return df
        
class Persistence():
    '''
    This class is for persistence analysis
    '''

    def __init__(self, sample):
        '''
        This function makes initialization.
        input : 
        sample (DataFrame): Data for analysis. The structure of the sample:
                        The first column : sample indicator
                        The second column : timestamp
                        The higher order columns: the variables.
        '''
        import numpy as np

        self.sample = sample
        self._columns = sample.columns 
        self._r, self._c = np.shape(self.sample)
    
    def _shift(self, series, lag):
        '''
        This private function shift the time series with lags.
        input :
        series (Series): The series need to be shifted with lags.
        lag (int): The lag order.
        '''
        
        lag_series = series.groupby([self._columns[0]]).shift(-lag)
        lag_series.name = series.name + str(lag)
        return lag_series
    
    def fit(self, lags):
        '''
        This function calculate the persistence with lags.
        input :
            lags (list): the lags that need to be analyzed.
        '''
        import pandas as pd
        
        temp_sample = self.sample.set_index([self._columns[0], self._columns[1]]).sort_index()

        def autocorr(x):
            import numpy as np
        
            return np.corrcoef(x.iloc[:, 0], x.iloc[:, 1])[0, 1] 
        
        variable_autocorr = list()
        for lag_num in range(len(lags)):
            for i in range(self._c-2):
                temp_shift = self._shift(temp_sample.iloc[:, i], lags[lag_num])
                temp_merge = pd.merge(temp_sample.iloc[:, i], temp_shift, left_index=True, right_index=True).dropna()
                temp_autocorr = temp_merge.groupby([self._columns[1]])[self._columns[i+2], temp_shift.name].apply(autocorr)
                variable_autocorr.append(temp_autocorr)
        
        self._lag = lags
        self._variable_autocorr = pd.concat(variable_autocorr, axis=1)

    def summary(self, periodic=False, export=False):
        '''
        Print the Result
        This function prints the result summary and exports table. The Fisher coefficient and the Spearman coefficient are both calculated. 
        input :
            periodic (boolean): whether prints periodic result. The **DEFAULT** setting is False.
            export (boolean): whether export the summary table. The **DEFAULT** setting is False.

        output :
            df (DataFrame): If export is True, then output the summary table. 

        Example:
        import numpy as np
        import pandas as pd
        from portfolio_analysis import Persistence as pste
    
        # generate time 
        year = np.ones((3000,1), dtype=int)*2020
        id = np.linspace(1, 3000, 3000, dtype=int)
        for i in range(19):
            year = np.append(year, (2019-i)*np.ones((3000,1), dtype=int))
            id = np.append(id, np.linspace(1, 3000, 3000, dtype=int))
    
        # generate character
        character_1 = np.random.normal(0, 1, 20*3000)
        character_2 = np.random.normal(0, 1, 20*3000)
    
        # generate future return
        ret=character_1*-0.5 + character_2*0.5 + np.random.normal(0,1,20*3000)
        # create sample containing future return, character, time
        sample = np.array([id, year, ret, character_1, character_2]).T
        sample = pd.DataFrame(sample, columns=['id', 'year', 'ret', 'character_1', 'character_2'])
    
        exper = pste(sample)
        exper.fit(lags=[1, 2, 3])
        exper.summary(periodic=True)
        exper.summary()
        '''
        import numpy as np
        from prettytable import PrettyTable

        table = PrettyTable()
        
        if periodic == True:            
            field_name = ['Time']
        elif periodic == False:
            field_name = ['Variable']
        for lag_num in range(len(self._lag)):
            field_name = field_name + [self._columns[i+2] + '_lag_' + str(self._lag[lag_num]) for i in range(self._c-2)]

        table.field_names = field_name

        
        if periodic == True:
            #time = np.sort(np.unique(self.sample.iloc[:, 1]))
            for i in self._variable_autocorr.index:
                table.add_row([str(i)] + list(np.around(self._variable_autocorr.loc[i], decimals=5)))
        
        elif periodic == False:
            average = np.around(np.mean(self._variable_autocorr, axis=0), decimals=5)
            table.add_row(['Average']+list(average))

        print(table)

        if export == True :
            import pandas as pd
            try:
                from StringIO import StringIO
            except ImportError:
                from io import StringIO
            
            csv_string = table.get_csv_string()
            with StringIO(csv_string) as f:
                df = pd.read_csv(f)
            
            return df

class Tangency_portfolio():
    '''
    This class calculate the tangency portfolio
    The related content can be found in Financial Asset Pricing Theory by Munk(2010).
    '''
    def __init__(self, rf, mu, cov_mat):
        '''
        input : 
            rf (float): risk free rate
            mu (array or Series): stock rate of return
            cov_mat (matrix): covariance matrix of stock rate of return 
        '''
        import numpy as np

        self.rf = rf
        if type(mu).__name__ == 'Series':
            self.mu = np.array(mu)
            self.asset_name = list(mu.index)
        else: 
            self.mu = mu
            self.asset_name = None
        self.mu = np.reshape(self.mu, (len(self.mu), 1))
        self.cov_mat = np.mat(cov_mat)

    def _portfolio_weight(self):
        '''
        calculate the portfolio weight in tangency portfolio
        output : 
            weight (vector): the portfolio weight in tangency portfolio
        '''
        import numpy as np
        from numpy.linalg import inv
        
        excess_ret = np.mat((self.mu - self.rf))
        self.weight = inv(self.cov_mat).dot(excess_ret)/np.sum(inv(self.cov_mat).dot(excess_ret))

        return self.weight
    
    def _sharpe_ratio(self):
        '''
        calculate the Sharpe ratio of the tangency portfolio
        output :
            sr (float): the Sharpe ratio of the tangency portoflio
        '''
        import numpy as np
        
        sigma = (self.weight.T.dot(self.cov_mat).dot(self.weight))[0,0]
        sr = (self.weight.T.dot(self.mu)-self.rf)/sigma**0.5

        return sr[0, 0]

    def fit(self):
        '''
        this function run the portfolio_weight, sharpe_ratio
        '''

        return self._portfolio_weight(), self._sharpe_ratio()

    def print(self):
        '''
        this function print the result

        Example:
        import numpy as np
        from portfolio_analysis import Tangency_portfolio as tanport
    
        # construct the sample data 1
        mu = np.array([0.0427, 0.0015, 0.0285])
        cov_mat = np.mat([[0.01, 0.0018, 0.0011], [0.0018, 0.0109, 0.0026], [0.0011, 0.0026, 0.0199]])
        rf = 0.005
    
        # calculate the weight and the sharpe ratio
        portfolio = tanport(rf, mu, cov_mat)
        print(portfolio._portfolio_weight())
        print(portfolio.fit())
        portfolio.print()

        # construct the sample data 2
        mu = np.array([0.0427, 0.0015, 0.0285, 0.0028])
        cov_mat = np.mat([[0.01, 0.0018, 0.0011, 0], [0.0018, 0.0109, 0.0026, 0], [0.0011, 0.0026, 0.0199, 0], [0, 0, 0, 0.1]])
        rf = 0.005
    
        # calculate the weight and the sharpe ratio
        portfolio = tanport(rf, mu, cov_mat)
        print(portfolio._portfolio_weight())
        print(portfolio.fit())
        portfolio.print()
        '''
        from prettytable import PrettyTable 
        import numpy as np

        table = PrettyTable()
        if self.asset_name != None:
            table.field_names = ['Weight'] + self.asset_name
        else:
            table.field_names = ['Weight'] + ['asset' + str(i+1) for i in range(len(self.weight))]
        
        table.add_row([' '] + [np.around(j[0, 0], 5) for j in self.weight])

        print(table)

class Spanning_test():
    '''
    This module is designed for spanning test. Three asymptotic estimates and one small sample estimates are contained. The construction is based on 
        R. Kan, G. Zhou, Test of Mean-Variance Spanning, Annals of Economics and Finance, 2012, 13-1, 145-193.    
    '''
    def __init__(self, Rn, Rk):
        '''
        input:
            Rn (ndarray): T*N test assets. T: time length, N: asset numbers
            Rk (ndarray): T*K baseline assets. T: time length, K: assets numbers
        '''
        import numpy as np

        if type(Rn).__name__ == 'DataFrame':
            self.Rn = np.array(Rn)
            self.rn, self.cn =np.shape(Rn)
            self.Rn_name = Rn.columns
        elif type(Rn).__name__ == 'Series':
            self.Rn = np.array(Rn)
            self.rn = len(Rn)
            self.cn = 1
            self.Rn_name = Rn.name
        else :            
            self.Rn = Rn
            self.Rn_name = None
            try :
                self.rn, self.cn =np.shape(Rn)
            except:
                self.rn = len(Rn)
                self.cn = 1
        
        if type(Rk).__name__ == 'DataFrame':
            self.Rk = np.array(Rk)
            self.rk, self.ck =np.shape(Rk)
        elif type(Rk).__name__ == 'Series':
            self.Rk = np.array(Rk)
            self.rk = len(Rk)
            self.ck = 1
        else:            
            self.Rk = Rk
            try :
                self.rk, self.ck =np.shape(Rk)
            except:
                self.rk = len(Rk)
                self.ck = 1
     
    def _cov(self):
        '''
        This function calculates the covariance
        '''
        import numpy as np

        self.V_11 = np.correlate(self.Rn)
        self.V_12 = np.correlate(self.Rn, self.Rk)
        self.V_22 = np.correlate(self.Rk)


    def _regress(self):
        '''
        This function regresses Rn on Rk and return the eigen value and U statistics for building estimates of hypothesis tests.
        output :
            eigen1 (float) : the eigen value #1
            eigen2 (float) : the eigen value #2
            U (float) : the U statistics
        '''
        import numpy as np
        import statsmodels.api as sm
        from numpy.linalg import inv, det, eig
        
        vecn_one = np.ones((self.cn, 1))
        veck_one = np.ones((self.ck, 1))
        alpha = np.zeros((self.cn, 1))
        beta = np.zeros((self.cn, self.ck))
        B = np.zeros((self.ck+1, self.cn))
        residue = np.zeros((self.rn, self.cn))
        X = sm.add_constant(self.Rk)
        pvalues = np.zeros((self.ck+1, self.cn))

        for i in range(self.cn):
            model = sm.OLS(self.Rn[:, i], sm.add_constant(self.Rk)).fit()
            alpha[i, :] = model.params[0]
            beta[i, :] = model.params[1:]
            B[:, i] = model.params
            residue[:, i] = model.resid
            pvalues[:, i] = model.pvalues

        delta = vecn_one - beta.dot(veck_one)
        sigma = residue.T.dot(residue) / self.rn

        A = np.block([
            [1, np.zeros((1, self.ck))],
            [0, -np.ones((1, self.ck))]
        ])
        C = np.block([
            [np.zeros((1, self.cn))],
            [np.ones((1, self.cn))]
        ])

        theta = np.mat(A).dot(B) + C
        G = self.rn * np.mat(A).dot(inv(X.T.dot(X))).dot(np.mat(A).T)
        H = theta.dot(inv(sigma)).dot(theta.T)
        U = 1 / det(np.identity(2) + H.dot(inv(G)))
        eigen, eigenvec = eig(H.dot(inv(G)))
        eigen1 = eigen[0]
        eigen2 = eigen[1]

        self.alpha = alpha
        self.delta = delta
        self.pvalues = pvalues

        return eigen1, eigen2, U

    def _build_statistics(self):
        '''
        This function build three asymptotic estimates and one small sample estimate. 
        The asymptotic estimates include likelihood ratio (LR), Wald test (W), Lagrange multiplier test (LM). 
        The one small sample estimate is F-test corresponding to likelihood ratio. 
        The asymptotic estimates satisfy the chi-square distribution with freedom 2N, 
        where N is the number of test assets. 
        The small sample estimate satisfies the F-distribution with coefficient, 2N, 2(T-K-N) for N>1, and 2, (T-K-1) for N=1.
        
        This function build three statistics: 
        Likelihood Ratio: LR, 
        Wald test : W, 
        Lagrangianian test: LM.
        output :
        perc (array) : the quantiles of chi-square distribution at 90%, 95%, 99%. 
        perc_F (array) : the quantiles of F distribution at 90%, 95%, 99%.
        [LR, chi_LR] (float) : The LR estimate and p-value of test. 
        [W, chi_W] (float) : The Wald estimate and p-value of test.
        [LM, chi_LM] (float) : The LM estimate and p-value of test.
        [LR_F, f_LR] (float) : The F estimate and p-value of test.
        '''
        import numpy as np
        from scipy import stats
        
        eigen1, eigen2, U = self._regress()
        LR = self.rn * np.log((1 + eigen1) * (1 + eigen2))
        W = self.rn * (eigen1 + eigen2)
        LM = self.rn * (eigen1 / (1 + eigen1) + eigen2 / (1 + eigen2))
        
        perc = stats.chi2.ppf([0.9, 0.95, 0.99], df=2*self.cn)
        chi_LR = 1.0 - stats.chi2.cdf(LR, df=2*self.cn)
        chi_W = 1.0 - stats.chi2.cdf(W, df=2*self.cn)
        chi_LM = 1.0 - stats.chi2.cdf(LM, df=2*self.cn)
        
        if self.cn > 1: 
            LR_F = (1 / U ** 0.5 - 1) * (self.rn - self.ck - self.cn) / self.cn
            perc_F = stats.f.ppf([0.9, 0.95, 0.99], 2 * self.cn, 2 * (self.rn-self.ck-self.cn))
            f_LR = 1 - stats.f.cdf(LR_F, 2 * self.cn, 2 * (self.rn - self.ck - self.cn))
        elif self.cn == 1:
            LR_F = (1 / U - 1) * (self.rn - self.ck - self.cn) / 2
            perc_F = stats.f.ppf([0.9, 0.95, 0.99], 2 * self.cn, self.rn - self.ck - 1)
            f_LR = 1 - stats.f.cdf(LR_F, 2 * self.cn, self.rn - self.ck - 1)
        
        return perc, perc_F, [LR, chi_LR], [W, chi_W], [LM, chi_LM], [LR_F, f_LR]
    
    def fit(self):
        '''
        This function fits the model
        output :
            perc (array) : the quantiles of chi-square distribution at 90%, 95%, 99%. 
            perc_F (array) : the quantiles of F distribution at 90%, 95%, 99%.
            [LR, chi_LR] (float) : The LR estimate and p-value of test. 
            [W, chi_W] (float) : The Wald estimate and p-value of test.
            [LM, chi_LM] (float) : The LM estimate and p-value of test.
            [LR_F, f_LR] (float) : The F estimate and p-value of test.

        Example:
        import numpy as np
        from portfolio_analysis import Spanning_test as span

        factor1 = np.random.normal(loc=0.1, scale=1.0, size=(240, 1))
        factor2 = np.random.normal(loc=0.2, scale=2.0, size=(240, 1))
        factor3 = np.random.normal(loc=0.5, scale=4.5, size=(240, 1))

        factor4 = 0.1 * factor1 + 0.5 * factor2 + 0.4 * factor3
        factor5 = -0.2 * factor1 - 0.1 * factor2 + 1.3 * factor3
        factor6 = 1.0 * factor1 - 0.5 * factor2 + 0.5 * factor3
        factor7 = 0.2 * factor1 + 0.1 * factor2 + 0.7 * factor3
        factor8 = -0.1 * factor1 -0.1 * factor2 + 1.2 * factor3
        factor9 = -0.3 * factor1 - 0.2 * factor2 + 1.5 * factor3
        factor10 = 0.9 * factor1 - 0.5 * factor2 + 0.6 * factor3
        factor11 = 0.2 * factor1 - 0.1 * factor2 + 0.9 * factor3
    
        factornew1 = np.random.normal(loc=0.3, scale=2.0, size=(240, 1))

        factork = np.block([factor1, factor2, factor3, factor4, factor5, factor6, factor7, factor8, factor9])
        factorn = np.block([factor10, factor11])
  
        model1 = span(factorn, factork)
        model1._regress()
        model1._build_statistics()
        model1.fit()
        model1.summary()

        model2 = span(factornew1, factork)
        model2._regress()
        model2._build_statistics()
        model2.fit()
        model2.summary()
        '''
        perc, perc_F, [LR, chi_LR], [W, chi_W], [LM, chi_LM], [LR_F, f_LR] = self._build_statistics()
        self.perc = perc
        self.perc_F = perc_F
        self.LR = LR
        self.chi_LR = chi_LR
        self.W = W
        self.chi_W = chi_W
        self.LM = LM
        self.chi_LM = chi_LM
        self.LR_F = LR_F
        self.f_LR = f_LR
        return perc, perc_F, [LR, chi_LR], [W, chi_W], [LM, chi_LM], [LR_F, f_LR]
    
    def summary(self):
        '''
        print the result
        '''
        from prettytable import PrettyTable
        import numpy as np

        table = PrettyTable()
        table.field_names = ['asset', 'alpha', 'p-value', 'delta', 'F-test', 'p-value-F', 'LR', 'p-value-LR', 'W', 'p-value-W', 'LM', 'p-value-LM', 'T', 'N', 'K']
        if self.Rn_name == None:
            for i in range(self.cn):            
                table.add_row(['asset'+str(i), self.alpha[i, 0], self.pvalues[0, i], self.delta[i, 0], self.LR_F, self.f_LR, self.LR, self.chi_LR, self.W, self.chi_W, self.LM, self.chi_LM, self.rn, self.cn, self.ck])
        
        table.float_format = '.5'
        print(table)

