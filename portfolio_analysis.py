'''
Portfolio Analysis
This module is used for portfolio analysis
which is divided into 4 steps
1. select breakpoints
2. distribute the assets into groups
3. calculate the average and difference of groups
4. present the result
'''

from calendar import c
from msilib.schema import Error
from unicodedata import decimal
from unittest import expectedFailure

from click import group
from numpy import average, dtype
from pyrsistent import v
from sklearn.model_selection import GroupShuffleSplit
from typesentry import I


class ptf_analysis():
    def __init__(self) :
        pass
    
    def select_breakpoints(self,character,number,perc=None):
        '''
        select the breakpoints of the sample
        input : 
            character: used to be divided
            number: the number of the breakpoint and the number of interval is number+1
            perc(None): if perc is true and a list of number, it represents the percentage setted to divided the sample.
                        once it is setted, then the number is invalid         
        output: 
            the rows of samples is realized data
            the columns of sampple are characters
            breakpoint: the selected breakpoint 
        '''
        import numpy as np
        # create percentiles
        if perc == None: 
            perc = np.linspace(0, 100, number+2, dtype=int)
        elif perc is not None :
            perc = perc
            
        breakpoint = np.percentile(character, perc,axis=0)
        return breakpoint
    
    def distribute(self, character, breakpoint):
        '''
        split the character into groups
        input:
            character: character used to divided into groups
            breakpoint: the breakpoint for dividing samples
        output:
            label: return a label column for character
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
        calculate the average return for each group 
        input:  sample_return: sample forecasted return  
                label: group label
        output: average value of groups
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

    
class Univariate(ptf_analysis):
    def __init__(self, sample, number, perc=None, maxlag=12, weight=False):
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
            IOError
        
        self.number = number
        if perc is not None :
            self.number = len(perc) - 2
        self.perc = perc
        self.maxlag = maxlag
        self._factor = None
        self._time = np.sort(np.unique(self.sample[:, 2])) 
        self._weight = weight       
        
    def divide_by_time(self, sample):
        '''
        split the sample by time into groups  将样本按照时间分组
        output: groups_by_time (list) 按时间分组的样本
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
            breakpoint = super().select_breakpoints(group[:, 1], self.number, self.perc)
            # for each time, generate label
            label = super().distribute(group[:, 1], breakpoint)
            # for each group in each time, calculate the average future return
            if self._weight == False:
                average_group_time[:, i] = super().average(group[:, 0], label[:, 0]).reshape((self.number+1))
            elif self._weight == True:
                average_group_time[:, i] = super().average(group[:, 0], label[:, 0], weight=group[:, 3]).reshape((self.number+1))
            else: return IOError

            self._label.append(label[:, 0])
            
        # return the Table
        # Rows: groups in each time
        # Columns: Time 
        return average_group_time

    def difference(self,average_group):
        '''
        calculate the difference group return
        input : average_group : Average group at each time(MATRIX: N*T)
        output: the matrix added with the difference group return
        '''
        import numpy as np
        
        diff = average_group[-1, :] - average_group[0, :]
        diff = diff.reshape((1, len(diff)))        
        result = np.append(average_group, diff, axis=0)

        return result

    def summary_and_test(self) :
        '''
        summary the result and take t test
        '''
        import numpy as np
        from scipy import stats
        
        self.result = self.difference(self.average_by_time())
        self.average = np.mean(self.result, axis=1)
        self.ttest = stats.ttest_1samp(self.result, 0.0, axis=1)
        # if there is a facotr adjustment, then normal return result plus the anomaly result
#        if self.factor is not None  :
#            self.alpha, self.alpha_tvalue = self.factor_adjustment(self.result)
#            return self.average, self.ttest, self.alpha, self.alpha_tvalue

        return self.average, self.ttest
    
    def fit(self) :
        '''
        fit the model
        '''
        self.summary_and_test()

    def factor_adjustment(self, factor):
        '''
        factor adjustment 因子调整
        input: reuslt: Return Table with difference sequence
               factor: factor order by time
        output: alpha 超额收益
                ttest T统计量
        '''
        import statsmodels.api as sm
        import numpy as np
        
        self._factor = factor

        # take the inverse of the Table with difference
        # Rows: Time
        # Columns: Groups Return
        table = self.result.T
        row, col = np.shape(table)
        # generate anomaly: alpha 
        # generate tvalues: ttest
        alpha = np.zeros((col, 1))
        ttest = np.zeros((col, 1))
        

        # factor adjusment
        for i in range(col):
            model = sm.OLS(table[:,i], sm.add_constant(factor))
            # fit the model with the Newey-West Adjusment
            # lags=maxlag
            re = model.fit()
            re = re.get_robustcov_results(cov_type='HAC', maxlags=self.maxlag, use_correction=True)
            #print(re.summary())
            # get anomaly and tvalues
            alpha[i] = re.params[0]
            ttest[i] = re.tvalues[0]
        
        self.alpha = alpha
        self.alpha_tvalue = ttest
        return alpha, ttest    
    
    def summary_statistics(self, variables=None, periodic=False):
        '''
        Summary Statistics 描述性统计
        input : variables(ndarray/DataFrame) 除了排序变量之外的其他需要分组总结的数据
                periodic(boolean) 是否报告每一期数据
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
                elif type(variables).__namr__ == 'Series':
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
                        corr.append(np.around(sts.pearsonr(temp_variables[:, j], temp_variables[:, k])[0], decimals=5))
                        corr_spearman.append(np.around(sts.spearmanr(temp_variables[:, j], temp_variables[:, k])[0], decimals=5))
            
            corr_maxtrix[i, :] = corr
            corr_maxtrix_spearman[i, :] = corr_spearman
            
            if periodic == True:
                table.add_row([str(self._time[i]) + ' '] + corr)
                table_spear.add_row([str(self._time[i]) + ' '] + corr_spearman)
                
        if periodic == False:
            table.add_rows([['Pearson'] + list(np.around(np.mean(corr_maxtrix, axis=0), decimals=5))])
            table.add_row(['Spearman'] + list(np.around(np.mean(corr_maxtrix_spearman, axis=0), decimals=5)))
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
        print summary_by_time
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
        
    def print_summary(self, export=False):
        '''
        print summary
        '''
        import numpy as np
        from prettytable import PrettyTable
        # generate Table if no factor
        table = PrettyTable()
        table.add_column('Group', ['Average', 'T-Test'])
        for i in range(self.number+1):
            table.add_column(str(i+1), np.around([self.average[i], self.ttest[0][i]], decimals=3))
        table.add_column('Diff', np.around([self.average[-1], self.ttest[0][-1]], decimals=3))
        
        if self._factor is not None :
            table = PrettyTable()
            table.add_column('Group', ['Average', 'T-Test', 'Alpha', 'Alpha-T'])
            for i in range(self.number+1):
                table.add_column(str(i+1), np.around([self.average[i], self.ttest[0][i], self.alpha[i][0], self.alpha_tvalue[i][0]], decimals=3))
            table.add_column('Diff', np.around([self.average[-1], self.ttest[0][-1], self.alpha[-1][0], self.alpha_tvalue[-1][0]], decimals=3))
            
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
    def __init__(self, sample, number=5, perc_row=None, perc_col=None, weight=False, maxlag=12):
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
            IOError

        self.number = number
        self._factor = None
        self.maxlag = maxlag
        self.weight = weight
        self.perc_row = perc_row
        self.perc_col = perc_col
        if (perc_row is not None) and (perc_col is not None):
            self.perc_sign = True
        else:
            self.perc_sign = False

        
    def divide_by_time(self):
        '''
        split the sample by time into groups  将样本按照时间分组
        output: groups_by_time (list) 按时间分组的样本
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
        '''
        import numpy as np
        # get the sample groups by time 得到按时间分组的样本
        groups_time = self.divide_by_time()
        # generate table of  average return for group and time 生成组和时间的表格
        # Rows: groups 行： 组  
        # Columns: Time 列： 时间
        if self.perc_sign == False:
            average_group_time = np.zeros((self.number+1, self.number+1, len(groups_time)))
        elif self.perc_sign == True :
            average_group_time = np.zeros((len(self.perc_row)-1, len(self.perc_col)-1, len(groups_time)))       
        
        for i in range(len(groups_time)):
            # for each time, there exists a group 
            group = groups_time[i]
            # for each time, generate breakpoint
            breakpoint_row = super().select_breakpoints(group[:, 1], self.number, self.perc_row)
            # for each time, generate label
            label_row = super().distribute(group[:, 1], breakpoint_row)[:, 0]
            
            if conditional == False:
                breakpoint_col = super().select_breakpoints(group[:, 2], self.number, self.perc_col)
                label_col = super().distribute(group[:, 2], breakpoint_col)[:, 0]
            elif conditional == True:
                label_row_unique = list(np.unique(label_row))
                label_col = - np.ones(len(group[:, 2]))
                for j in range(len(label_row_unique)):
                    breakpoint_col = super().select_breakpoints(group[:, 2][np.where(label_row==label_row_unique[j])], self.number, self.perc_col)
                    label_col[np.where(label_row==label_row_unique[j])] = super().distribute(group[:, 2][np.where(label_row==label_row_unique[j])], breakpoint_col)[:, 0]

            # for each group in each time, calculate the average future return
            label = [label_row, label_col]

            if self.perc_sign == False:
                if self.weight == False:
                    average_group_time[:,:,i] = super().average(group[:, 0], label, cond='bi').reshape((self.number+1, self.number+1))
                else:
                    average_group_time[:,:,i] = super().average(group[:, 0], label, cond='bi', weight=group[:, -1]).reshape((self.number+1, self.number+1))
            elif self.perc_sign == True:
                if self.weight == False:
                    average_group_time[:,:,i] = super().average(group[:, 0], label, cond='bi').reshape((len(self.perc_row)-1, len(self.perc_col)-1))
                else:
                    average_group_time[:,:,i] = super().average(group[:, 0], label, cond='bi', weight=group[:, -1]).reshape((len(self.perc_row)-1, len(self.perc_col)-1))

        # return the Table
        # Rows: groups in each time
        # Columns: Time 

        return average_group_time
            
    def difference(self, average_group):
        '''
        calculate the difference group return
        input : average_group : Average group at each time(MATRIX: N*T)
        output: the matrix added with the difference group return
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
        factor adjustment 因子调整
        input: reuslt: Return Table with difference sequence
               factor: factor order by time
        output: alpha 超额收益
                ttest T统计量
        '''
        import statsmodels.api as sm
        import numpy as np
        
        self._factor = factor
        # result: r * c * n 
        r, c, n = np.shape(self.result)
        # generate anomaly: alpha 
        # generate tvalues: ttest
        alpha = np.zeros((r, c))
        ttest = np.zeros((r, c))
        
        # factor adjusment
        for i in range(r):
            for j in range(c):
                model = sm.OLS(self.result[i, j, :], sm.add_constant(factor))
                # fit the model with the Newey-West Adjusment
                # lags=maxlag
                re = model.fit()
                re = re.get_robustcov_results(cov_type='HAC', maxlags=self.maxlag, use_correction=True)
                #print(re.summary())
                # get anomaly and tvalues
                alpha[i, j] = re.params[0]
                ttest[i, j] = re.tvalues[0]
        
        self.alpha = alpha
        self.alpha_tvalue = ttest
        return alpha, ttest    
    
    def summary_and_test(self, **kwargs) :
        '''
        summary the result and take t test
        '''
        import numpy as np
        from scipy import stats
        
        self.result = self.difference(self.average_by_time(**kwargs))
        self.average = np.mean(self.result, axis=2)
        self.ttest = stats.ttest_1samp(self.result, 0.0, axis=2)
        # if there is a facotr adjustment, then normal return result plus the anomaly result
#        if self.factor is not None  :
#            self.alpha, self.alpha_tvalue = self.factor_adjustment(self.result)
#            return self.average, self.ttest, self.alpha, self.alpha_tvalue

        return self.average,self.ttest

    def fit(self, **kwargs):
        self.summary_and_test(**kwargs)
    
    def print_summary_by_time(self, export=False) :
        '''
        print summary_by_time
        '''
        import numpy as np
        from prettytable import PrettyTable
        
        r, c, n = np.shape(self.result)
        table = PrettyTable()
        time = np.sort(np.unique(self.sample[:, 3]))
        table.field_names = ['Time', 'Group'] + [str(i+1) for i in range(self.number+1)] + ['Diff']
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
        
    def print_summary(self, export=False):
        '''
        print summary
        '''
        import numpy as np
        from prettytable import PrettyTable
        
        # generate Table if no factor
        if self._factor is None :
            table=PrettyTable()
            if self._sample_type == 'ndarray':
                table.field_names = ['Group'] + [i+1 for i in range(self.number+1)] + ['Diff']
            elif self._sample_type == 'DataFrame':
                table.field_names = ['Group'] + [self._columns[2] + str(i+1) for i in range(self.number+1)] + ['Diff']

            for i in range(self.number+2):
                if i == self.number+1 :
                    temp = ['Diff']
                    temp_tvalue = [' ']
                else:
                    if self._sample_type == 'ndarray':
                        temp = [str(i+1)]
                    elif self._sample_type == 'DataFrame':
                        temp = [self._columns[1] + str(i+1)]
                    temp_tvalue = [' ']
                for j in range(self.number+2):
                    temp.append(np.around(self.average[i, j], decimals=3))
                    temp_tvalue.append(np.around(self.ttest[0][i, j], decimals=3))
                table.add_row(temp)
                table.add_row(temp_tvalue)

        elif self._factor is not None :
            table = PrettyTable()
            if self._sample_type == 'ndarray':
                table.field_names = ['Group'] + [i+1 for i in range(self.number+1)] + ['Diff']
            elif self._sample_type == 'DataFrame':
                table.field_names = ['Group'] + [self._columns[2] + str(i+1) for i in range(self.number+1)] + ['Diff']
            
            for i in range(self.number+2):
                if i == self.number+1:
                    temp = ['Diff']
                    temp_tvalue = [' ']
                    temp_fac = ['alpha']
                    temp_fac_tvalue = [' ']
                else :
                    if self._sample_type == 'ndarray':
                        temp = [str(i+1)]
                    elif self._sample_type == 'DataFrame':
                        temp = [self._columns[1] + str(i+1)]
                    temp_tvalue = [' ']
                    temp_fac = ['alpha']
                    temp_fac_tvalue = [' ']
                for j in range(self.number+2):
                    temp.append(np.around(self.average[i, j], decimals=3))
                    temp_tvalue.append(np.around(self.ttest[0][i, j], decimals=3))
                    temp_fac.append(np.around(self.alpha[i, j], decimals=3))
                    temp_fac_tvalue.append(np.around(self.alpha_tvalue[i, j], decimals=3))
                table.add_row(temp)
                table.add_row(temp_tvalue)
                table.add_row(temp_fac)
                table.add_row(temp_fac_tvalue)
            
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
        Input : 
            sample (DataFrame):
            first column : sample number
            second column : timestamp
            variables: other columns
        '''
        import numpy as np

        self.sample = sample
        self._columns = sample.columns 
        self._r, self._c = np.shape(self.sample)
    
    def _shift(self, series, lag):
        '''
        Input :
            series, lags
        '''
        
        lag_series = series.groupby([self._columns[0]]).shift(-lag)
        lag_series.name = series.name + str(lag)
        return lag_series
    
    def fit(self, lags):
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





        
        

        
        
        
        


        
        
