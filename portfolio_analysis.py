'''
Portfolio Analysis
This module is used for portfolio analysis
which is divided into 4 steps
1. select breakpoints
2. distribute the assets into groups
3. calculate the average and difference of groups
4. present the result
'''

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
    
    def average(self, sample_return,label):
        '''
        calculate the average return for each group 
        input: sample_return: sample forecasted return  
               label: group label
        output: average value of groups
        '''
        import numpy as np
        # the whole group label, eg. 10 group lables: [1,2,3,4,5,6,7,8,9,10]  
        temp_label = np.sort(np.unique(label))
        # the average return of each group
        average_return = np.zeros((len(temp_label), 1))
        # calculate the average return of each group throught matching the sample_return's 
        # label with the group label and the sample_return is Forecasted Return
        for i in range(len(temp_label)):
            average_return[i, 0]=np.mean(sample_return[np.where(label==temp_label[i])])
        
        # return average value of each group
        return average_return
            
    
class Univariate(ptf_analysis):
    def __init__(self, sample, number, perc=None, factor=None, maxlag=12):
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
        self.sample = sample
        self.number = number
        if perc is not None :
            self.number = len(perc) - 2
        self.perc = perc
        self.factor = factor
        self.maxlag = maxlag
        self.summary_and_test()
        
    def divide_by_time(self):
        '''
        split the sample by time into groups  将样本按照时间分组
        output: groups_by_time (list) 按时间分组的样本
        '''
        import numpy as np
        
        time=np.sort(np.unique(self.sample[:,2]))
        groups_by_time=list()
        for i in range(len(time)):
            groups_by_time.append(self.sample[np.where(self.sample[:,2]==time[i])])
        
        return groups_by_time
        
    def average_by_time(self):
        '''
        average of the groups at each time point
        '''
        import numpy as np
        # get the sample groups by time 得到按时间分组的样本
        groups_time=self.divide_by_time()
        # generate table of  average return for group and time 生成组和时间的表格
        # Rows: groups 行： 组  
        # Columns: Time 列： 时间
        average_group_time=np.zeros((self.number+1,len(groups_time)))
        for i in range(len(groups_time)):
            # for each time, a group exist
            group=groups_time[i]
            # for each time, generate breakpoint
            breakpoint=super().select_breakpoints(group[:,1],self.number,self.perc)
            # for each time, generate label
            label=super().distribute(group[:,1],breakpoint)
            # for each group in each time, calculate the average future return
            average_group_time[:,i]=super().average(group[:,0],label[:,0]).reshape((self.number+1))
            
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
        
        diff=average_group[-1,:]-average_group[0,:]
        diff=diff.reshape((1,len(diff)))        
        result=np.append(average_group,diff,axis=0)
        return result
    
    def factor_adjustment(self,result):
        '''
        factor adjustment 因子调整
        input: reuslt: Return Table with difference sequence
               factor: factor order by time
        output: alpha 超额收益
                ttest T统计量
        '''
        import statsmodels.api as sm
        import numpy as np
        
        # take the inverse of the Table with difference
        # Rows: Time
        # Columns: Groups Return
        table=result.T
        row,col=np.shape(table)
        # generate anomaly: alpha 
        # generate tvalues: ttest
        alpha=np.zeros((col,1))
        ttest=np.zeros((col,1))
        
        # factor adjusment
        for i in range(col):
            model=sm.OLS(table[:,i],sm.add_constant(self.factor))
            # fit the model with the Newey-West Adjusment
            # lags=maxlag
            re=model.fit()
            re=re.get_robustcov_results(cov_type='HAC',maxlags=self.maxlag,use_correction=True)
            #print(re.summary())
            # get anomaly and tvalues
            alpha[i]=re.params[0]
            ttest[i]=re.tvalues[0]
        
        return alpha,ttest    
    
    def summary_and_test(self) :
        '''
        summary the result and take t test
        '''
        import numpy as np
        from scipy import stats
        
        self.result=self.difference(self.average_by_time())
        self.average=np.mean(self.result,axis=1)
        self.ttest=stats.ttest_1samp(self.result,0.0,axis=1)
        # if there is a facotr adjustment, then normal return result plus the anomaly result
        if self.factor is not None  :
            self.alpha,self.alpha_tvalue=self.factor_adjustment(self.result)
            return self.average,self.ttest,self.alpha,self.alpha_tvalue

        return self.average,self.ttest
    
    def print_summary_by_time(self) :
        '''
        print summary_by_time
        '''
        import numpy as np
        from prettytable import PrettyTable
        
        r,c=np.shape(self.result)
        table=PrettyTable()
        time=np.sort(np.unique(self.sample[:,2]))
        table.add_column('Time',time)
        for i in range(r-1):
            table.add_column(str(i+1),np.around(self.result[i,:],decimals=3))
        table.add_column('diff',np.around(self.result[r-1,:],decimals=3))
        print(table)
        
    def print_summary(self):
        '''
        print summary
        '''
        import numpy as np
        from prettytable import PrettyTable
        # generate Table if no factor
        table=PrettyTable()
        table.add_column('Group',['Average','T-Test'])
        for i in range(self.number+1):
            table.add_column(str(i+1),np.around([self.average[i],self.ttest[0][i]],decimals=3))
        table.add_column('Diff',np.around([self.average[-1],self.ttest[0][-1]],decimals=3))
        if self.factor is not None :
            table=PrettyTable()
            table.add_column('Group',['Average','T-Test','Alpha','Alpha-T'])
            for i in range(self.number+1):
                table.add_column(str(i+1),np.around([self.average[i],self.ttest[0][i],self.alpha[i][0],self.alpha_tvalue[i][0]],decimals=3))
            table.add_column('Diff',np.around([self.average[-1],self.ttest[0][-1],self.alpha[-1][0],self.alpha_tvalue[-1][0]],decimals=3))
            
        np.set_printoptions(formatter={'float':'{:0.3f}'.format})
        print(table)

        